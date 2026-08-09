"""Compact a game's suffixed Parquet files into the canonical per-table file.

Live/resumed captures append `<table>-poll-*.parquet` / `<table>-resume-*.parquet`
beside the canonical `<table>.parquet` (the append_suffix mechanism that
avoids truncating an open file). Readers union them transparently, but
many small files slow queries and prune_game.py rewrites only the
canonical file. This tool merges each table's finalized files into one.

Safety:
- Only finalized files (PAR1 footer) are touched; an in-flight writer's
  file is left alone (so don't compact a game that's actively capturing).
- The merged file is written to a temp name, its row count verified
  against the sum of the inputs, and only then swapped in.
- Originals are MOVED to `<gamedir>/pre_compact/` (default), not deleted;
  pass --purge to delete them instead once you trust the tool.

Usage:
  python scripts/compact_game.py --game 824566
  python scripts/compact_game.py --all-games --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games  # noqa: E402

TABLES = ("actor_frames", "bat_frames", "ball_frames", "pitch_events")


def _is_finalized(p: Path) -> bool:
    try:
        if p.stat().st_size < 8:
            return False
        with p.open("rb") as f:
            f.seek(-4, 2)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def compact_table(gdir: Path, table: str, dry_run: bool, purge: bool) -> dict:
    files = sorted(p for p in gdir.glob(f"{table}*.parquet") if _is_finalized(p))
    if len(files) <= 1:
        return {"table": table, "files": len(files), "action": "skip"}

    con = duckdb.connect()
    files_sql = "[" + ", ".join(f"'{p.as_posix()}'" for p in files) + "]"
    expected = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({files_sql}, union_by_name=true)"
    ).fetchone()[0]
    if dry_run:
        con.close()
        return {"table": table, "files": len(files), "rows": expected,
                "action": "would-compact"}

    tmp = gdir / f"{table}.compact.tmp.parquet"
    con.execute(
        f"COPY (SELECT * FROM read_parquet({files_sql}, union_by_name=true) "
        f"      ORDER BY segment_idx) "
        f"TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    got = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{tmp.as_posix()}')").fetchone()[0]
    con.close()
    if got != expected:
        tmp.unlink()
        return {"table": table, "files": len(files), "action": "ABORT",
                "error": f"tmp has {got} rows, expected {expected}"}

    stash = gdir / "pre_compact"
    if not purge:
        stash.mkdir(exist_ok=True)
    for p in files:
        if purge:
            p.unlink()
        else:
            p.rename(stash / p.name)
    tmp.rename(gdir / f"{table}.parquet")
    return {"table": table, "files": len(files), "rows": expected,
            "action": "compacted"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int)
    ap.add_argument("--all-games", action="store_true")
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR", "data"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--purge", action="store_true",
                    help="Delete originals instead of moving to pre_compact/")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.game:
        pks = [args.game]
    elif args.all_games:
        pks = list_games(data_dir)
    else:
        raise SystemExit("need --game or --all-games")

    for pk in pks:
        gdir = data_dir / str(pk)
        if not gdir.is_dir():
            print(f"pk={pk}: no game dir")
            continue
        for table in TABLES:
            r = compact_table(gdir, table, args.dry_run, args.purge)
            if r["action"] != "skip":
                print(f"pk={pk} {r}")


if __name__ == "__main__":
    main()
