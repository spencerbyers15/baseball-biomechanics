"""Ingest MLB statsapi feed/live pitch data into pitch_labels.parquet.

Usage:
    python scripts/ingest_pitch_labels.py --game 823141
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Column order and types match the legacy SQLite pitch_label table so existing
# parquet files (migrated via scripts/migrate_sqlite_to_parquet.py) stay
# schema-compatible.
_COLUMNS: list[tuple[str, pa.DataType]] = [
    ("game_pk",         pa.int64()),
    ("play_id",         pa.string()),
    ("ab_index",        pa.int64()),
    ("pitch_number",    pa.int64()),
    ("inning",          pa.int64()),
    ("top_inning",      pa.int64()),
    ("batter_id",       pa.int64()),
    ("pitcher_id",      pa.int64()),
    ("batter_side",     pa.string()),
    ("pitcher_throws",  pa.string()),
    ("balls_before",    pa.int64()),
    ("strikes_before",  pa.int64()),
    ("outs_before",     pa.int64()),
    ("pitch_type",      pa.string()),
    ("pitch_type_desc", pa.string()),
    ("start_speed",     pa.float64()),
    ("end_speed",       pa.float64()),
    ("spin_rate",       pa.float64()),
    ("spin_direction",  pa.float64()),
    ("release_x",       pa.float64()),
    ("release_y",       pa.float64()),
    ("release_z",       pa.float64()),
    ("release_extension", pa.float64()),
    ("plate_x",         pa.float64()),
    ("plate_z",         pa.float64()),
    ("sz_top",          pa.float64()),
    ("sz_bot",          pa.float64()),
    ("result_call",     pa.string()),
    ("result_desc",     pa.string()),
    ("is_in_play",      pa.int64()),
    ("is_strike",       pa.int64()),
    ("is_ball",         pa.int64()),
    ("start_time",      pa.string()),
    ("end_time",        pa.string()),
    ("start_time_unix", pa.float64()),
    ("end_time_unix",   pa.float64()),
]
_SCHEMA = pa.schema(_COLUMNS)


def fetch_feed(game_pk: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh) FieldVision-pitch-label/1.0",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _iso_to_unix(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        d = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return d.timestamp()
    except Exception:
        return None


def feed_to_rows(game_pk: int, feed: dict) -> list[dict]:
    rows = []
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    for play in plays:
        ab_index = play.get("atBatIndex")
        inning = play.get("about", {}).get("inning")
        half = play.get("about", {}).get("halfInning")
        top_inning = 1 if half == "top" else (0 if half == "bottom" else None)
        matchup = play.get("matchup", {})
        batter_id = matchup.get("batter", {}).get("id")
        pitcher_id = matchup.get("pitcher", {}).get("id")
        batter_side = matchup.get("batSide", {}).get("code")
        pitcher_throws = matchup.get("pitcherHand", {}).get("code")

        for ev in play.get("playEvents", []):
            if not ev.get("isPitch"):
                continue
            play_id = ev.get("playId")
            if not play_id:
                continue
            pn = ev.get("pitchNumber")
            count = ev.get("count", {})
            details = ev.get("details", {})
            pdata = ev.get("pitchData", {})
            breaks = pdata.get("breaks", {})
            coords = pdata.get("coordinates", {})
            ptype = details.get("type", {}) or {}
            call = details.get("call", {}) or {}
            start_time = ev.get("startTime")
            end_time = ev.get("endTime")
            rows.append({
                "game_pk": game_pk,
                "play_id": play_id,
                "ab_index": ab_index,
                "pitch_number": pn,
                "inning": inning,
                "top_inning": top_inning,
                "batter_id": batter_id,
                "pitcher_id": pitcher_id,
                "batter_side": batter_side,
                "pitcher_throws": pitcher_throws,
                "balls_before": count.get("balls"),
                "strikes_before": count.get("strikes"),
                "outs_before": count.get("outs"),
                "pitch_type": ptype.get("code"),
                "pitch_type_desc": ptype.get("description"),
                "start_speed": pdata.get("startSpeed"),
                "end_speed": pdata.get("endSpeed"),
                "spin_rate": breaks.get("spinRate"),
                "spin_direction": breaks.get("spinDirection"),
                "release_x": coords.get("x0"),
                "release_y": coords.get("y0"),
                "release_z": coords.get("z0"),
                "release_extension": pdata.get("extension"),
                "plate_x": coords.get("px"),
                "plate_z": coords.get("pz"),
                "sz_top": pdata.get("strikeZoneTop"),
                "sz_bot": pdata.get("strikeZoneBottom"),
                "result_call": call.get("code"),
                "result_desc": details.get("description"),
                "is_in_play": 1 if details.get("isInPlay") else 0,
                "is_strike": 1 if details.get("isStrike") else 0,
                "is_ball": 1 if details.get("isBall") else 0,
                "start_time": start_time,
                "end_time": end_time,
                "start_time_unix": _iso_to_unix(start_time),
                "end_time_unix": _iso_to_unix(end_time),
            })
    return rows


def write_pitch_labels(game_pk: int, data_dir: Path, rows: list[dict]) -> int:
    """Idempotent write: merges with any existing pitch_labels.parquet,
    deduping on (game_pk, play_id) where new rows replace old."""
    out_dir = data_dir / str(game_pk)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pitch_labels.parquet"

    new_table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    if out_path.exists():
        old = pq.read_table(out_path)
        # Reorder/coerce old to match our canonical schema (handles minor
        # type drift from the migration's COPY).
        old = old.cast(_SCHEMA, safe=False) if old.schema != _SCHEMA else old
        new_ids = set(rows[i]["play_id"] for i in range(len(rows)))
        if new_ids:
            mask = [pid not in new_ids for pid in old.column("play_id").to_pylist()]
            kept = old.filter(pa.array(mask))
            combined = pa.concat_tables([kept, new_table])
        else:
            combined = old
    else:
        combined = new_table

    pq.write_table(combined, out_path, compression="zstd")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR", "data"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Fetching statsapi feed for game {args.game}...")
    feed = fetch_feed(args.game)
    rows = feed_to_rows(args.game, feed)
    n = write_pitch_labels(args.game, data_dir, rows)
    print(f"Wrote {n} pitch_label rows to {data_dir / str(args.game) / 'pitch_labels.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
