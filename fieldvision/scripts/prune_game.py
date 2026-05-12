"""Optional retention pruner for per-game Parquet stores.

No-op by default. Every cut is opt-in via flag. Designed for "the NAS
is filling up; condense the corpus" scenarios — not as part of the
standard ingest pipeline.

Each cut operates on actor_frames.parquet only (the biggest table).
Other tables (bat_frames, ball_frames, pitch_label, pitch_event, ...)
are untouched and still describe the full play, so dropped frames can
be re-scraped from MLB if their retention window is still open.

Cuts (all default OFF; can combine):

  --drop-umps-coaches
      Drop every actor row with actor_type in {umpire, plate-umpire,
      coach}. Always safe — these add no biomechanics signal.

  --drop-fielders-non-x
      Drop fielder rows whose timestamp falls inside a non-X pitch's
      [start_time_unix, end_time_unix] window. "X" = ball in play.
      Pitcher/catcher/batter are always kept.

  --drop-fielders-no-touch-post-release
      For X pitches only, drop fielder rows AFTER the pitch's release
      (start_time_unix) whose hand-to-ball minimum distance never
      drops below --touch-radius (default 2.0 ft). Pre-release
      positioning frames are preserved so defensive shifts remain
      analyzable.

Usage:
  python scripts/prune_game.py --game 823141 --drop-umps-coaches
  python scripts/prune_game.py --game 823141 --drop-umps-coaches \\
      --drop-fielders-non-x --drop-fielders-no-touch-post-release
  python scripts/prune_game.py --all-games --drop-umps-coaches --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games


SKELETAL_ROLES = ("pitcher", "catcher", "batter")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_filter(args, con: duckdb.DuckDBPyConnection) -> str:
    """Compose a single SQL WHERE clause that keeps the rows we want.
    Empty string means 'no filter, copy as-is'."""
    keep = []

    if args.drop_umps_coaches:
        # Keep anything that isn't an umpire/coach.
        keep.append(
            "actor_type NOT IN ('umpire', 'plate-umpire', 'coach')"
        )

    if args.drop_fielders_non_x:
        # Need pitch_label to know which time windows are non-X.
        non_x_exists = con.execute(
            "SELECT COUNT(*) FROM pitch_label WHERE result_call != 'X' "
            "AND start_time_unix IS NOT NULL AND end_time_unix IS NOT NULL"
        ).fetchone()[0]
        if non_x_exists == 0:
            _log("  --drop-fielders-non-x: no labeled non-X pitches in this game; "
                 "skipping (run ingest_pitch_labels.py first)")
        else:
            roles = ", ".join(f"'{r}'" for r in SKELETAL_ROLES)
            keep.append(
                f"NOT (actor_type = 'fielder' AND EXISTS ("
                f"  SELECT 1 FROM pitch_label pl "
                f"   WHERE pl.result_call != 'X' "
                f"     AND af.time_unix BETWEEN pl.start_time_unix AND pl.end_time_unix"
                f"))"
            )

    if args.drop_fielders_no_touch_post_release:
        # X-pitch + post-release + fielder + no hand-ball proximity → drop.
        # We compute the keep-set: fielders who DID touch the ball during
        # *some* X pitch. Anyone in the keep-set survives even on plays
        # they weren't involved in.
        r = con.execute(
            "SELECT COUNT(*) FROM pitch_label WHERE result_call = 'X' "
            "AND start_time_unix IS NOT NULL AND end_time_unix IS NOT NULL"
        ).fetchone()[0]
        if r == 0:
            _log("  --drop-fielders-no-touch-post-release: no labeled X pitches; "
                 "skipping (run ingest_pitch_labels.py first)")
        else:
            radius = float(args.touch_radius)
            # Build a temp set of (play_id, mlb_player_id) where the player
            # got within `radius` of the ball during the play window.
            con.execute("DROP TABLE IF EXISTS _touchers")
            con.execute(f"""
                CREATE TEMP TABLE _touchers AS
                SELECT DISTINCT pl.play_id, af.mlb_player_id
                  FROM pitch_label pl
                  JOIN actor_frame af
                    ON af.time_unix BETWEEN pl.start_time_unix AND pl.end_time_unix
                  JOIN ball_frame bf
                    ON bf.time_unix = af.time_unix
                 WHERE pl.result_call = 'X'
                   AND af.actor_type = 'fielder'
                   AND (
                     ((af.hand_lt_x - bf.ball_x) * (af.hand_lt_x - bf.ball_x)
                    + (af.hand_lt_y - bf.ball_y) * (af.hand_lt_y - bf.ball_y)
                    + (af.hand_lt_z - bf.ball_z) * (af.hand_lt_z - bf.ball_z)) < {radius * radius}
                     OR
                     ((af.hand_rt_x - bf.ball_x) * (af.hand_rt_x - bf.ball_x)
                    + (af.hand_rt_y - bf.ball_y) * (af.hand_rt_y - bf.ball_y)
                    + (af.hand_rt_z - bf.ball_z) * (af.hand_rt_z - bf.ball_z)) < {radius * radius}
                   )
            """)
            keep.append(
                "NOT (actor_type = 'fielder' "
                "  AND EXISTS ("
                "    SELECT 1 FROM pitch_label pl "
                "     WHERE pl.result_call = 'X' "
                "       AND af.time_unix BETWEEN pl.start_time_unix AND pl.end_time_unix "
                "       AND af.time_unix > pl.start_time_unix "  # post-release only
                "       AND NOT EXISTS ("
                "         SELECT 1 FROM _touchers t "
                "          WHERE t.play_id = pl.play_id "
                "            AND t.mlb_player_id = af.mlb_player_id"
                "       )"
                "  )"
                ")"
            )

    return " AND ".join(f"({c})" for c in keep)


def prune_one(game_pk: int, data_dir: Path, args) -> dict:
    gdir = data_dir / str(game_pk)
    af_path = gdir / "actor_frames.parquet"
    if not af_path.exists():
        return {"game_pk": game_pk, "skipped": "no actor_frames.parquet"}

    con = duckdb.connect()
    # Register every parquet that exists as a view so the filter SQL can
    # reference pitch_label, ball_frame, etc.
    for table, fname in [
        ("actor_frame",  "actor_frames"),
        ("pitch_label",  "pitch_labels"),
        ("ball_frame",   "ball_frames"),
    ]:
        p = gdir / f"{fname}.parquet"
        if p.exists():
            con.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{p.as_posix()}')"
            )

    where = _build_filter(args, con)
    before = con.execute("SELECT COUNT(*) FROM actor_frame").fetchone()[0]
    if not where:
        con.close()
        return {"game_pk": game_pk, "skipped": "no cuts requested",
                "rows_before": before}

    after = con.execute(
        f"SELECT COUNT(*) FROM actor_frame af WHERE {where}"
    ).fetchone()[0]
    pct = 100.0 * after / max(before, 1)

    out = {
        "game_pk": game_pk,
        "rows_before": before,
        "rows_after": after,
        "pct_kept": pct,
        "bytes_before": af_path.stat().st_size,
    }

    if args.dry_run:
        con.close()
        return out

    tmp = af_path.with_suffix(".parquet.tmp")
    con.execute(
        f"COPY (SELECT * FROM actor_frame af WHERE {where}) "
        f"TO '{tmp.as_posix()}' "
        f"(FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 100000)"
    )
    con.close()
    # Atomic-ish swap (rename within same filesystem). The old reader
    # connections will keep reading the new file via the same path.
    tmp.replace(af_path)
    out["bytes_after"] = af_path.stat().st_size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, help="Single gamePk")
    ap.add_argument("--all-games", action="store_true",
                    help="Run against every game in data-dir")
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR", "data"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be cut; don't rewrite anything")
    ap.add_argument("--drop-umps-coaches", action="store_true")
    ap.add_argument("--drop-fielders-non-x", action="store_true")
    ap.add_argument("--drop-fielders-no-touch-post-release", action="store_true")
    ap.add_argument("--touch-radius", type=float, default=2.0,
                    help="Hand-to-ball distance (ft) below which a fielder is "
                         "considered to have touched the ball")
    args = ap.parse_args()

    if not (args.game or args.all_games):
        ap.error("specify --game <pk> or --all-games")
    if args.game and args.all_games:
        ap.error("--game and --all-games are mutually exclusive")

    data_dir = Path(args.data_dir)
    targets = [args.game] if args.game else list_games(data_dir)
    if not targets:
        _log(f"no games found under {data_dir}")
        return

    _log(f"{'DRY RUN: ' if args.dry_run else ''}pruning {len(targets)} games")
    total_before = 0
    total_after = 0
    total_bytes_before = 0
    total_bytes_after = 0
    for pk in targets:
        try:
            r = prune_one(pk, data_dir, args)
        except Exception as e:
            _log(f"  pk={pk}: FAILED ({type(e).__name__}: {e})")
            continue
        if "skipped" in r:
            _log(f"  pk={pk}: skipped — {r['skipped']}")
            continue
        b, a = r["rows_before"], r["rows_after"]
        total_before += b
        total_after += a
        total_bytes_before += r["bytes_before"]
        if "bytes_after" in r:
            total_bytes_after += r["bytes_after"]
            _log(f"  pk={pk}: {b:,} → {a:,} rows ({r['pct_kept']:.1f}% kept), "
                 f"{r['bytes_before']/1e9:.2f} → {r['bytes_after']/1e9:.2f} GB")
        else:
            _log(f"  pk={pk}: {b:,} → {a:,} rows ({r['pct_kept']:.1f}% kept) [dry-run]")

    if total_before:
        pct = 100.0 * total_after / total_before
        _log("")
        _log(f"=== summary ===")
        _log(f"  rows: {total_before:,} → {total_after:,} ({pct:.1f}% kept, "
             f"{100 - pct:.1f}% pruned)")
        if total_bytes_after:
            _log(f"  bytes: {total_bytes_before/1e9:.1f} → {total_bytes_after/1e9:.1f} GB "
                 f"({(total_bytes_before - total_bytes_after)/1e9:.1f} GB freed)")


if __name__ == "__main__":
    main()
