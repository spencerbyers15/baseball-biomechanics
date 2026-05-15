"""Audit: for each captured game, compare distinct segment_idx in
actor_frames against the segment count MLB's manifest claims exist.

Reports games with gaps so you know what to re-scrape.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fieldvision.parquet_readers import list_games  # noqa: E402

DATA_DIR = Path(os.environ.get("FV_DATA_DIR", REPO_ROOT / "data"))
TOKEN = (REPO_ROOT / ".fv_token.txt").read_text().strip()
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def manifest_segment_count(pk: int) -> int | None:
    url = f"https://fieldvision-hls.mlbinfra.com/mannequin/{pk}/1.6.2/manifest.json"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {TOKEN}",
        "x-mannequin-client": "gameday",
        "Origin": "https://www.mlb.com",
        "Referer": "https://www.mlb.com/",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        return len(data.get("records", []))
    except Exception as e:
        return None


def parquet_segment_count(pk: int) -> tuple[int, int]:
    """(distinct_count, max_idx) across all actor_frames*.parquet for the game."""
    files = sorted((DATA_DIR / str(pk)).glob("actor_frames*.parquet"))
    files = [f for f in files if f.stat().st_size > 8]
    if not files:
        return 0, -1
    files_sql = "[" + ", ".join(f"'{f.as_posix()}'" for f in files) + "]"
    con = duckdb.connect()
    n, m = con.execute(
        f"SELECT COUNT(DISTINCT segment_idx), MAX(segment_idx) "
        f"FROM read_parquet({files_sql}, union_by_name=true)"
    ).fetchone()
    con.close()
    return int(n), int(m) if m is not None else -1


def audit_one(pk: int) -> dict:
    expected = manifest_segment_count(pk)
    have, max_idx = parquet_segment_count(pk)
    if expected is None:
        return {"pk": pk, "have": have, "max_idx": max_idx,
                "expected": None, "missing": None, "expired": True}
    return {"pk": pk, "have": have, "max_idx": max_idx,
            "expected": expected, "missing": expected - have,
            "expired": False}


def main():
    games = list_games(DATA_DIR)
    print(f"Auditing {len(games)} games...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(audit_one, pk): pk for pk in games}
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            if i % 25 == 0:
                print(f"  ...{i}/{len(games)}", flush=True)

    results.sort(key=lambda r: r["pk"])

    # Summary
    expired = [r for r in results if r["expired"]]
    clean = [r for r in results if not r["expired"] and r["missing"] == 0]
    gappy = [r for r in results if not r["expired"] and (r["missing"] or 0) > 0]
    overshot = [r for r in results if not r["expired"] and (r["missing"] or 0) < 0]

    print()
    print("=" * 60)
    print(f"  Total games audited:   {len(results)}")
    print(f"  Manifest expired:      {len(expired)}  (no MLB data; can't audit)")
    print(f"  Clean (no gaps):       {len(clean)}")
    print(f"  Has missing segments:  {len(gappy)}")
    print(f"  Has more than expected: {len(overshot)}  (rare; manifest shrunk?)")
    print()

    if gappy:
        total_missing = sum(r["missing"] for r in gappy)
        print(f"=== Games with gaps ({total_missing:,} segments missing total) ===")
        # Sort by worst-affected first
        gappy.sort(key=lambda r: -r["missing"])
        for r in gappy[:50]:
            pct = 100 * r["have"] / r["expected"] if r["expected"] else 0
            print(f"  pk={r['pk']}: have {r['have']:>5,} / {r['expected']:>5,} "
                  f"({pct:.1f}%, missing {r['missing']:>5,}, max_idx {r['max_idx']})")
        if len(gappy) > 50:
            print(f"  ... and {len(gappy) - 50} more")

    if overshot:
        print()
        print("=== Games where we have MORE than manifest claims ===")
        for r in overshot:
            print(f"  pk={r['pk']}: have {r['have']:,} but manifest says {r['expected']:,}")


if __name__ == "__main__":
    main()
