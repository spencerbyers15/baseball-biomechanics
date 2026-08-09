"""Parallel backfill: capture every available MLB game in a date range.

Game-level parallelism over the proven scrape_one_game() path — each worker
process handles one game start-to-finish (manifest → segments → Parquet),
so per-game resume/ingest semantics are untouched and N games proceed at
once. MLB's per-connection politeness (FV_SLEEP between segment fetches)
still applies inside each worker.

Ordering is oldest-first so games nearest MLB's retention horizon are
rescued before they expire (observed horizon 2026-08-09: a game from
2026-04-14 still served — ~117 days — but don't count on it staying that
generous).

Games already complete on disk are skipped: a game is complete when its
Parquet segment count matches the manifest's record count (--audit mode),
or cheaply when a finalized actor_frames.parquet exists (default).

Token: re-read from FV_TOKEN_FILE before each game. On 401/403 the worker
touches state/token_expired.flag (the Mac watchdog refreshes within ~60s)
and waits for a fresh token instead of dying.

Usage (on Nellie):
  FV_TOKEN_FILE=/home/spencer/.fv_token.txt \
  FV_DATA_DIR=/media/datasets/spencer/fieldvision/data \
  FV_SAMPLES_DIR=/media/scratch/spencer/fieldvision/samples \
  FV_STATE_DIR=/media/scratch/spencer/fieldvision/state \
  python3 scripts/fv_backfill.py --start 2026-05-20 --end 2026-08-08 \
      --workers 8 --delete-bins
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fieldvision.mlb_api import resolve_manifest  # noqa: E402
from scrape_team_history import http_get, scrape_one_game  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FV_DATA_DIR", REPO_ROOT / "data"))
STATE_DIR = Path(os.environ.get("FV_STATE_DIR", REPO_ROOT / "state"))
TOKEN_FILE = Path(os.environ.get("FV_TOKEN_FILE", REPO_ROOT / ".fv_token.txt"))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_token_blocking(max_wait_s: int = 900) -> str | None:
    """Read the token; if it's missing/stale-flagged, wait for the watchdog
    to deliver a fresh one (poll every 20s up to max_wait_s)."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        if TOKEN_FILE.exists():
            tok = TOKEN_FILE.read_text().strip()
            if tok.startswith("eyJ") and tok.count(".") == 2:
                return tok
        time.sleep(20)
    return None


def fetch_final_games(start: str, end: str) -> list[dict]:
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&startDate={start}&endDate={end}")
    s, body = http_get(url)
    if s != 200:
        raise SystemExit(f"statsapi schedule fetch failed: HTTP {s}")
    games = []
    for d in json.loads(body).get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            games.append({
                "pk": g["gamePk"],
                "date": g.get("officialDate", g.get("gameDate", "")[:10]),
                "away": g.get("teams", {}).get("away", {}).get("team", {}).get("name", "?"),
                "home": g.get("teams", {}).get("home", {}).get("team", {}).get("name", "?"),
            })
    # Doubleheaders share a date; pk breaks the tie deterministically.
    games.sort(key=lambda g: (g["date"], g["pk"]))
    return games


def _is_finalized_parquet(p: Path) -> bool:
    try:
        if p.stat().st_size < 8:
            return False
        with p.open("rb") as f:
            f.seek(-4, 2)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def have_game_on_disk(pk: int) -> bool:
    """Cheap completeness check: any finalized actor_frames*.parquet."""
    gdir = DATA_DIR / str(pk)
    if not gdir.is_dir():
        return False
    return any(_is_finalized_parquet(f)
               for f in gdir.glob("actor_frames*.parquet"))


def capture_game(game: dict, delete_bins: bool) -> dict:
    """Worker: capture one game, waiting out token failures."""
    pk = game["pk"]
    for attempt in range(3):
        token = read_token_blocking()
        if token is None:
            return {**game, "ok": False, "error": "no valid token after wait"}
        result = scrape_one_game(pk, token, delete_bins)
        err = str(result.get("error", ""))
        if result.get("ok") or not ("401" in err or "403" in err):
            return {**game, **result}
        # Auth failure: flag is already touched by scrape_one_game; give the
        # watchdog time to push a fresh token, then retry this game.
        time.sleep(90)
    return {**game, "ok": False, "error": "auth failures persisted"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--delete-bins", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only the first N games (oldest first; debug)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only: show what would be captured")
    args = ap.parse_args()

    games = fetch_final_games(args.start, args.end)
    log(f"{len(games)} Final games {args.start} → {args.end}")

    todo = [g for g in games if not have_game_on_disk(g["pk"])]
    log(f"{len(games) - len(todo)} already on disk, {len(todo)} to capture "
        f"(oldest first)")
    if args.limit:
        todo = todo[:args.limit]
        log(f"limited to first {len(todo)} per --limit")
    if args.dry_run:
        for g in todo[:40]:
            log(f"  would capture {g['date']} pk={g['pk']} {g['away']} @ {g['home']}")
        if len(todo) > 40:
            log(f"  ... and {len(todo) - 40} more")
        return
    if not todo:
        log("nothing to do")
        return

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results_path = STATE_DIR / f"backfill_{args.start}_{args.end}.jsonl"
    done = failed = 0
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
            results_path.open("a") as out:
        futures = {ex.submit(capture_game, g, args.delete_bins): g for g in todo}
        for fut in as_completed(futures):
            g = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {**g, "ok": False, "error": f"worker crashed: {e}"}
            out.write(json.dumps(r) + "\n")
            out.flush()
            if r.get("ok"):
                done += 1
            else:
                failed += 1
            elapsed = (time.monotonic() - t0) / 60
            rate = (done + failed) / max(elapsed, 0.01)
            eta_min = (len(todo) - done - failed) / max(rate, 0.01)
            log(f"[{done + failed}/{len(todo)}] pk={r['pk']} {r['date']} "
                f"ok={r.get('ok')} fetched={r.get('fetched', '-')} "
                f"({rate:.2f} games/min, eta {eta_min:.0f} min, {failed} failed)")

    log(f"done: {done} captured, {failed} failed, results → {results_path}")


if __name__ == "__main__":
    main()
