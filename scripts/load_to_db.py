"""Bulk-decode all .bin segments for a game and load into SQLite.

Usage:
  python scripts/load_to_db.py --game 823141
  python scripts/load_to_db.py --game 823141 --rebuild   # drop + reload
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldvision.storage import (_actor_frame_insert_sql, ingest_segment,
                                 load_lookup_tables, open_game_db,
                                 open_registry, transaction, update_registry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=823141)
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--rebuild", action="store_true", help="Drop existing tables first")
    parser.add_argument("--limit", type=int, default=0, help="Only ingest N segments (debug)")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir) / f"binary_capture_{args.game}"
    data_dir = Path(args.data_dir)
    print(f"Loading game {args.game}")
    print(f"  source: {samples_dir.resolve()}")
    print(f"  output: {data_dir.resolve()}/fv_{args.game}.sqlite")

    metadata_path = samples_dir / f"mlb_{args.game}_metadata.json"
    labels_path = samples_dir / f"mlb_{args.game}_labels.json"
    if not metadata_path.exists() or not labels_path.exists():
        raise SystemExit(f"Missing metadata.json or labels.json in {samples_dir}")

    bin_paths = sorted(samples_dir.glob(f"mlb_{args.game}_segment_*.bin"),
                       key=lambda p: int(p.stem.split("_")[-1]))
    if not bin_paths:
        raise SystemExit("No .bin segments found")
    if args.limit:
        bin_paths = bin_paths[:args.limit]

    conn = open_game_db(args.game, data_dir)

    if args.rebuild:
        print("  dropping existing tables...")
        with transaction(conn):
            for table in ("actor_frame", "ball_frame", "game_event",
                          "labels", "bones", "players", "meta"):
                conn.execute(f"DELETE FROM {table}")

    print(f"\nLoading lookup tables...")
    labels_dict = load_lookup_tables(conn, metadata_path, labels_path)
    print(f"  labels: {len(labels_dict)} actor_uid → mlb_player_id mappings")

    print(f"\nIngesting {len(bin_paths)} segments...")
    insert_sql = _actor_frame_insert_sql()

    t0 = time.monotonic()
    total_actor_rows = 0
    total_ball_rows = 0
    last_print = t0
    for i, bin_path in enumerate(bin_paths):
        seg_idx = int(bin_path.stem.split("_")[-1])
        with transaction(conn):
            n_actor, n_ball = ingest_segment(conn, args.game, seg_idx,
                                             bin_path, labels_dict, insert_sql)
        total_actor_rows += n_actor
        total_ball_rows += n_ball
        now = time.monotonic()
        if now - last_print > 3 or i == len(bin_paths) - 1:
            elapsed = now - t0
            rate = (i + 1) / max(elapsed, 0.001)
            eta = (len(bin_paths) - i - 1) / max(rate, 0.001)
            print(f"  {i + 1}/{len(bin_paths)}  "
                  f"({rate:.1f} seg/s, eta {eta:.0f}s, "
                  f"{total_actor_rows:,} actor rows, {total_ball_rows:,} ball rows)")
            last_print = now

    print(f"\nDone in {time.monotonic() - t0:.1f}s")
    print(f"  actor_frame rows: {total_actor_rows:,}")
    print(f"  ball_frame rows:  {total_ball_rows:,}")

    print("\nUpdating registry...")
    db_path = data_dir / f"fv_{args.game}.sqlite"
    reg = open_registry(data_dir)
    update_registry(reg, conn, args.game, db_path)
    reg.close()

    # Quick example queries to validate
    print("\nSanity queries:")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT mlb_player_id) FROM actor_frame WHERE mlb_player_id IS NOT NULL")
    print(f"  distinct mlb_player_ids tracked: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(DISTINCT actor_uid) FROM actor_frame")
    print(f"  distinct actor_uids:             {cur.fetchone()[0]}")
    cur.execute("SELECT actor_type, COUNT(*) FROM actor_frame GROUP BY actor_type ORDER BY 2 DESC")
    print(f"  actor_type breakdown:")
    for row in cur.fetchall():
        print(f"    {row[0] or '(null)':<20} {row[1]:,} rows")
    cur.execute("SELECT MIN(time_unix), MAX(time_unix), MAX(time_unix) - MIN(time_unix) FROM actor_frame")
    mn, mx, span = cur.fetchone()
    print(f"  time span: {span / 60:.1f} minutes ({mn:.1f} → {mx:.1f})")
    cur.execute("SELECT mlb_player_id, COUNT(*) AS n FROM actor_frame "
                "WHERE mlb_player_id IS NOT NULL "
                "GROUP BY mlb_player_id ORDER BY n DESC LIMIT 5")
    print(f"  top tracked players (rows):")
    for row in cur.fetchall():
        print(f"    player_id={row[0]}  rows={row[1]:,}")

    conn.close()
    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"\nDB size: {size_mb:.1f} MB at {db_path}")


if __name__ == "__main__":
    main()
