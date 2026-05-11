"""Ingest MLB statsapi feed/live pitch data into pitch_label.

Usage:
    python scripts/ingest_pitch_labels.py --game 823141
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fieldvision.storage import (_pitch_label_insert_sql, open_game_db,
                                  transaction)


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


def ingest_feed_dict(conn: sqlite3.Connection, game_pk: int, feed: dict) -> int:
    """Insert one pitch_label row per isPitch playEvent. Returns row count."""
    sql = _pitch_label_insert_sql()
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
            rows.append((
                game_pk, play_id,
                ab_index, pn,
                inning, top_inning,
                batter_id, pitcher_id,
                batter_side, pitcher_throws,
                count.get("balls"), count.get("strikes"), count.get("outs"),
                ptype.get("code"), ptype.get("description"),
                pdata.get("startSpeed"), pdata.get("endSpeed"),
                breaks.get("spinRate"), breaks.get("spinDirection"),
                coords.get("x0"), coords.get("y0"), coords.get("z0"),
                pdata.get("extension"),
                coords.get("px"), coords.get("pz"),
                pdata.get("strikeZoneTop"), pdata.get("strikeZoneBottom"),
                call.get("code"), details.get("description"),
                1 if details.get("isInPlay") else 0,
                1 if details.get("isStrike") else 0,
                1 if details.get("isBall") else 0,
                start_time, end_time,
                _iso_to_unix(start_time), _iso_to_unix(end_time),
            ))
    if rows:
        conn.executemany(sql, rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    conn = open_game_db(args.game, data_dir)
    print(f"Fetching statsapi feed for game {args.game}...")
    feed = fetch_feed(args.game)
    with transaction(conn):
        n = ingest_feed_dict(conn, args.game, feed)
    print(f"Inserted {n} pitch_label rows for game {args.game}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
