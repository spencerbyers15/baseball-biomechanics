"""Backfill the new bat_frame table for an already-loaded game.

Faster than re-running load_to_db.py — only re-decodes inferredBat per
frame, doesn't redo the per-actor FK work.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldvision.storage import open_game_db, transaction
from fieldvision.wire_schemas import read_tracking_data


SCHEMA_BAT_FRAME = """
CREATE TABLE IF NOT EXISTS bat_frame (
    game_pk INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,
    frame_num INTEGER NOT NULL,
    time_unix REAL NOT NULL,
    head_x REAL, head_y REAL, head_z REAL,
    handle_x REAL, handle_y REAL, handle_z REAL,
    PRIMARY KEY (game_pk, segment_idx, frame_num)
);
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=823141)
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir) / f"binary_capture_{args.game}"
    bin_paths = sorted(samples_dir.glob(f"mlb_{args.game}_segment_*.bin"),
                       key=lambda p: int(p.stem.split("_")[-1]))
    print(f"Backfilling bat_frame for {len(bin_paths)} segments...")

    conn = open_game_db(args.game, Path(args.data_dir))
    conn.executescript(SCHEMA_BAT_FRAME)

    insert_sql = (
        "INSERT OR REPLACE INTO bat_frame "
        "(game_pk, segment_idx, frame_num, time_unix, "
        "head_x, head_y, head_z, handle_x, handle_y, handle_z) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    t0 = time.monotonic()
    total_rows = 0
    for i, p in enumerate(bin_paths):
        seg_idx = int(p.stem.split("_")[-1])
        td = read_tracking_data(p.read_bytes())
        rows = []
        for f in td.frames:
            ib = f.inferredBat
            if ib is None or ib.headPosition is None or ib.handlePosition is None:
                continue
            rows.append((
                args.game, seg_idx, f.num, f.time,
                ib.headPosition.x, ib.headPosition.y, ib.headPosition.z,
                ib.handlePosition.x, ib.handlePosition.y, ib.handlePosition.z,
            ))
        if rows:
            with transaction(conn):
                conn.executemany(insert_sql, rows)
            total_rows += len(rows)
        if i % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (len(bin_paths) - i - 1) / max(rate, 0.01)
            print(f"  {i + 1}/{len(bin_paths)}  ({rate:.0f} seg/s, "
                  f"eta {eta:.0f}s, {total_rows:,} bat rows)")

    print(f"\nDone in {time.monotonic() - t0:.1f}s")
    print(f"  bat_frame rows: {total_rows:,}")
    cur = conn.execute("SELECT COUNT(*) FROM bat_frame")
    print(f"  total in DB:    {cur.fetchone()[0]:,}")
    conn.close()


if __name__ == "__main__":
    main()
