"""Migrate fv_<gamePk>.sqlite databases to a per-game Parquet layout.

Old:  data/fv_<gamePk>.sqlite                    (one big SQLite per game)
New:  data/<gamePk>/{actor_frames, bat_frames, ball_frames,
                     pitch_events, pitch_labels, players,
                     labels, bones, meta}.parquet

Uses DuckDB's sqlite_scanner extension for the read side (fast parallel
read of SQLite tables, no SQLAlchemy needed). Each table dumps to its
own Parquet via DuckDB's COPY TO ... FORMAT PARQUET, which uses Snappy
compression by default and is way smaller than the raw SQLite (especially
for the float-heavy actor_frame).

Usage:
    python scripts/migrate_sqlite_to_parquet.py --src data/fv_823141.sqlite --dst data/823141
    python scripts/migrate_sqlite_to_parquet.py --src-dir data/ --dst-dir data/   # all games
    python scripts/migrate_sqlite_to_parquet.py --workers 4 ...                   # parallel
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb


# Each per-game SQLite has these tables; map them to Parquet filenames.
TABLES = [
    "actor_frame",
    "bat_frame",
    "ball_frame",
    "pitch_event",
    "pitch_label",
    "players",
    "labels",
    "bones",
    "meta",
]
PARQUET_NAMES = {
    "actor_frame":  "actor_frames",
    "bat_frame":    "bat_frames",
    "ball_frame":   "ball_frames",
    "pitch_event":  "pitch_events",
    "pitch_label":  "pitch_labels",
    "players":      "players",
    "labels":       "labels",
    "bones":        "bones",
    "meta":         "meta",
}


def migrate_one(sqlite_path: Path, parquet_dir: Path,
                compression: str = "zstd", row_group_size: int = 100_000) -> dict:
    """Convert one SQLite file to a directory of Parquet files."""
    sqlite_path = Path(sqlite_path)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(sqlite_path), "dest": str(parquet_dir),
               "tables_written": {}, "skipped": [], "error": None}
    t0 = time.monotonic()
    try:
        con = duckdb.connect()
        con.execute("INSTALL sqlite_scanner; LOAD sqlite_scanner;")
        con.execute(f"ATTACH '{sqlite_path.as_posix()}' AS src (TYPE SQLITE, READ_ONLY)")

        # discover tables actually present in this SQLite (old DBs may lack some)
        existing = {row[0] for row in con.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table'"
        ).fetchall()}

        for tbl in TABLES:
            if tbl not in existing:
                summary["skipped"].append(tbl)
                continue
            out = parquet_dir / f"{PARQUET_NAMES[tbl]}.parquet"
            n_rows = con.execute(f"SELECT COUNT(*) FROM src.{tbl}").fetchone()[0]
            if n_rows == 0:
                # write an empty parquet with schema so readers don't blow up
                con.execute(
                    f"COPY (SELECT * FROM src.{tbl} LIMIT 0) "
                    f"TO '{out.as_posix()}' "
                    f"(FORMAT PARQUET, COMPRESSION '{compression}')"
                )
            else:
                con.execute(
                    f"COPY (SELECT * FROM src.{tbl}) "
                    f"TO '{out.as_posix()}' "
                    f"(FORMAT PARQUET, COMPRESSION '{compression}', "
                    f"ROW_GROUP_SIZE {row_group_size})"
                )
            summary["tables_written"][tbl] = {
                "rows": n_rows,
                "bytes": out.stat().st_size,
            }
        con.close()
        summary["elapsed_sec"] = time.monotonic() - t0
        return summary
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["elapsed_sec"] = time.monotonic() - t0
        return summary


def _worker(args):
    sqlite_path, parquet_dir, compression, row_group_size = args
    return migrate_one(sqlite_path, parquet_dir, compression, row_group_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="Single SQLite to migrate")
    ap.add_argument("--dst", help="Destination Parquet dir (only with --src)")
    ap.add_argument("--src-dir", default=None,
                    help="Dir containing fv_<gamePk>.sqlite files")
    ap.add_argument("--dst-dir", default=None,
                    help="Parent dir for per-game Parquet dirs "
                         "(default: same as --src-dir)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel migrations")
    ap.add_argument("--compression", default="zstd",
                    choices=["snappy", "gzip", "zstd", "lz4", "none"])
    ap.add_argument("--row-group-size", type=int, default=100_000)
    ap.add_argument("--keep-sqlite", action="store_true",
                    help="Don't delete source SQLite after successful migration")
    args = ap.parse_args()

    jobs: list[tuple[Path, Path]] = []
    if args.src:
        if not args.dst:
            raise SystemExit("--src requires --dst")
        jobs.append((Path(args.src), Path(args.dst)))
    elif args.src_dir:
        src_dir = Path(args.src_dir)
        dst_dir = Path(args.dst_dir or args.src_dir)
        pat = re.compile(r"^fv_(\d+)\.sqlite$")
        for p in sorted(src_dir.glob("fv_*.sqlite")):
            m = pat.match(p.name)
            if not m: continue
            game_pk = m.group(1)
            jobs.append((p, dst_dir / game_pk))
    else:
        raise SystemExit("must specify --src or --src-dir")

    print(f"migrating {len(jobs)} games with {args.workers} workers, compression={args.compression}")

    worker_args = [(s, d, args.compression, args.row_group_size) for s, d in jobs]
    completed = 0
    total_in_bytes = 0
    total_out_bytes = 0
    failures = []
    started_at = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, wa): wa for wa in worker_args}
        for fut in as_completed(futures):
            wa = futures[fut]
            result = fut.result()
            completed += 1
            src = Path(wa[0])
            in_bytes = src.stat().st_size if src.exists() else 0
            out_bytes = sum(t["bytes"] for t in result.get("tables_written", {}).values())
            ratio = (in_bytes / max(out_bytes, 1))
            total_in_bytes += in_bytes
            total_out_bytes += out_bytes
            elapsed = time.monotonic() - started_at
            eta = (elapsed / completed) * (len(jobs) - completed) if completed else 0
            if result.get("error"):
                failures.append((src, result["error"]))
                print(f"  [{completed}/{len(jobs)}] ✗ {src.name}: {result['error']}")
                continue
            print(f"  [{completed}/{len(jobs)}] ✓ {src.name}  "
                  f"{in_bytes/1e9:.2f}→{out_bytes/1e9:.2f} GB "
                  f"({ratio:.1f}x compression, {result['elapsed_sec']:.0f}s)  "
                  f"eta {eta/60:.0f}m")
            if not args.keep_sqlite:
                # Also remove the -shm/-wal sidecars if present
                for ext in ("sqlite", "sqlite-shm", "sqlite-wal"):
                    sidecar = src.with_suffix(f".{ext}")
                    if sidecar.exists() and sidecar.suffix == ".sqlite":
                        sidecar.unlink()
                    elif sidecar.exists():
                        sidecar.unlink()

    print()
    print(f"=== done in {(time.monotonic() - started_at) / 60:.1f}m ===")
    print(f"  succeeded: {completed - len(failures)}")
    print(f"  failed:    {len(failures)}")
    if total_in_bytes:
        print(f"  total: {total_in_bytes/1e9:.1f} GB SQLite → {total_out_bytes/1e9:.1f} GB Parquet "
              f"({total_in_bytes/max(total_out_bytes,1):.1f}x smaller)")
    if failures:
        print("  failures:")
        for src, err in failures:
            print(f"    {src.name}: {err}")


if __name__ == "__main__":
    main()
