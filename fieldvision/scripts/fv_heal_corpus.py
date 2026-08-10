"""Heal gaps in already-captured games while MLB's retention window is open.

The May-era corpus predates the 429-indefinite-retry fix and the gap-aware
resume, so many games silently dropped segments. For every game on disk,
this compares the ingested segment set against the manifest's non-gap
records and re-fetches exactly what's missing (scrape_one_game is
gap-aware, so the heavy lifting is one call per gappy game).

Oldest gamePk first: the retention horizon (~117 days) is actively rolling
past the corpus start, so the oldest gaps die first.

  FV_TOKEN_FILE=... FV_DATA_DIR=... FV_SAMPLES_DIR=... FV_STATE_DIR=... \
  python3 scripts/fv_heal_corpus.py --workers 3 [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fieldvision.mlb_api import resolve_manifest  # noqa: E402
from fieldvision.parquet_readers import list_games  # noqa: E402
from fieldvision.storage_parquet import ingested_segment_set  # noqa: E402
from scrape_team_history import http_get, scrape_one_game  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FV_DATA_DIR", REPO_ROOT / "data"))
TOKEN_FILE = Path(os.environ.get("FV_TOKEN_FILE", REPO_ROOT / ".fv_token.txt"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_one(pk: int, token: str) -> dict:
    """How many non-gap manifest segments are we missing for pk?"""
    version, status, body = resolve_manifest(
        pk, lambda u: http_get(u, token, timeout=15, max_retries=2))
    if status == 404 or version is None:
        return {"pk": pk, "expired": status == 404, "status": status}
    recs = json.loads(body).get("records", [])
    non_gap = {r.get("index", i) for i, r in enumerate(recs)
               if not r.get("isGap")}
    have = ingested_segment_set(DATA_DIR, pk)
    return {"pk": pk, "expired": False,
            "missing": len(non_gap - have), "non_gap": len(non_gap)}


def heal_one(pk: int, delete_bins: bool = True) -> dict:
    token = TOKEN_FILE.read_text().strip()
    return {"pk": pk, **scrape_one_game(pk, token, delete_bins)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help="heal at most N games (oldest first)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = TOKEN_FILE.read_text().strip()
    pks = sorted(list_games(DATA_DIR))
    log(f"scanning {len(pks)} games for gaps (oldest first)")

    todo, expired = [], []
    for pk in pks:
        try:
            r = scan_one(pk, token)
        except Exception as e:
            log(f"  pk={pk} scan failed: {e}")
            continue
        if r.get("expired"):
            expired.append(pk)
        elif r.get("missing", 0) > 0:
            todo.append((pk, r["missing"]))
            log(f"  pk={pk}: {r['missing']} of {r['non_gap']} non-gap missing")
        time.sleep(0.2)

    log(f"scan done: {len(todo)} games need healing, "
        f"{len(expired)} expired (unhealable): {expired}")
    if args.limit:
        todo = todo[:args.limit]
    if args.dry_run or not todo:
        return

    healed = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(heal_one, pk): pk for pk, _ in todo}
        for fut in as_completed(futs):
            pk = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"pk": pk, "ok": False, "error": str(e)}
            if r.get("ok") and r.get("failed", 0) == 0:
                healed += 1
                log(f"HEALED pk={pk}: +{r.get('fetched')} segments "
                    f"[{healed + failed}/{len(todo)}]")
            else:
                failed += 1
                log(f"FAILED pk={pk}: {r.get('error') or r} "
                    f"[{healed + failed}/{len(todo)}]")
    log(f"done: {healed} healed, {failed} failed, {len(expired)} expired")


if __name__ == "__main__":
    main()
