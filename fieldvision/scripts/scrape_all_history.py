"""Parallel bulk scrape of every available MLB game (all teams) in a date range.

Wrapper around scrape_team_history.scrape_one_game() that:
  1. Fetches the MLB schedule for the date range (sportId=1, no team filter)
  2. Probes each game's FieldVision manifest for availability
  3. Spawns N parallel workers to download + ingest each available game
  4. Each game's data goes to its own SQLite (no collisions). Registry
     updates are serialized via a file lock to avoid corruption under
     nolock SQLite on CIFS.

Usage:
  python scripts/scrape_all_history.py --workers 4
  python scripts/scrape_all_history.py --since 2026-04-15 --workers 6
  python scripts/scrape_all_history.py --workers 4 --delete-bins
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import existing pipeline machinery
import scrape_team_history as sth  # type: ignore

USER_AGENT = sth.USER_AGENT
TOKEN_FILE = sth.TOKEN_FILE


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_all_games(start: str, end: str) -> list[dict]:
    """All MLB games in [start, end], any team, any state."""
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&startDate={start}&endDate={end}"
    )
    s, body = sth.http_get(url)
    if s != 200:
        raise SystemExit(f"statsapi schedule fetch failed: HTTP {s}")
    data = json.loads(body)
    games = []
    seen = set()
    for d in data.get("dates", []):
        for g in d.get("games", []):
            pk = g.get("gamePk")
            if not pk or pk in seen: continue
            seen.add(pk)
            status = g.get("status", {})
            teams = g.get("teams", {})
            games.append({
                "pk": pk,
                "date": g.get("officialDate", g.get("gameDate", "")[:10]),
                "away": teams.get("away", {}).get("team", {}).get("name", "?"),
                "home": teams.get("home", {}).get("team", {}).get("name", "?"),
                "state": status.get("abstractGameState"),
                "detailed": status.get("detailedState"),
            })
    return games


def _scrape_worker(pk: int, token: str, delete_bins: bool, away: str, home: str, date: str) -> dict:
    """Process-pool worker. Imports its own copy of scrape_team_history and
    calls scrape_one_game. Each worker has its own SQLite handle (per-game
    file), so concurrent runs don't collide on a single DB."""
    # Reload module locally so each worker has its own state
    import scrape_team_history as sth_local
    try:
        result = sth_local.scrape_one_game(pk, token, delete_bins)
        return {"pk": pk, "date": date, "away": away, "home": home,
                "ok": result.get("ok", False), "fetched": result.get("fetched", 0),
                "failed": result.get("failed", 0), "total": result.get("total", 0),
                "error": result.get("error")}
    except Exception as e:
        return {"pk": pk, "date": date, "away": away, "home": home,
                "ok": False, "error": f"exception: {type(e).__name__}: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=None,
                        help="Start date YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--until", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--workers", type=int, default=4,
                        help="How many games to download in parallel (default 4)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-bins", action="store_true",
                        help="Delete raw .bin files after ingestion (saves ~200MB/game)")
    parser.add_argument("--skip-pk", type=int, action="append", default=[],
                        help="Skip specific gamePks (can repeat)")
    args = parser.parse_args()

    if not TOKEN_FILE.exists():
        raise SystemExit(f"No token at {TOKEN_FILE}. Refresh first.")
    token = TOKEN_FILE.read_text().strip()

    if args.since is None:
        args.since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if args.until is None:
        args.until = datetime.now().strftime("%Y-%m-%d")

    log(f"Fetching ALL MLB games  {args.since} → {args.until}")
    games = fetch_all_games(args.since, args.until)
    final_games = [g for g in games if g["state"] == "Final"
                   and g["pk"] not in args.skip_pk]
    log(f"  {len(games)} games total, {len(final_games)} Final (excl. skipped)")

    # Probe availability — serial because it's lightweight and avoids
    # hammering the manifest endpoint
    log(f"Probing fieldvision-hls availability for {len(final_games)} games...")
    available = []
    expired = []
    for g in final_games:
        n = sth.probe_availability(g["pk"], token)
        if n is not None:
            available.append({**g, "n_segments": n})
        else:
            expired.append(g)

    total_seg = sum(g["n_segments"] for g in available)
    log("")
    log("=== summary ===")
    log(f"  available: {len(available)} games, {total_seg:,} segments")
    log(f"  expired:   {len(expired)} games (>3-4 weeks old)")
    seconds_serial = total_seg * 0.4
    seconds_parallel = seconds_serial / max(args.workers, 1)
    log(f"  serial ETA: {seconds_serial/3600:.1f} h | "
        f"parallel ({args.workers} workers) ETA: {seconds_parallel/3600:.1f} h")
    log(f"  estimated disk: ~{len(available) * 3.4:.0f} GB SQLite"
        + (f"" if args.delete_bins else f" + ~{len(available) * 0.2:.0f} GB raw bins"))

    if args.dry_run:
        log("dry-run: exiting")
        return

    log("")
    log(f"=== starting bulk download with {args.workers} workers ===")
    # Sort by date ascending — oldest first. MLB's fieldvision-hls retains
    # games for ~3-4 weeks, so games near the start of the window are the
    # ones at risk of falling off the cliff. Newer games are safe even if
    # the run is interrupted.
    available.sort(key=lambda g: g["date"])

    completed = 0
    failed = 0
    started_at = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_scrape_worker, g["pk"], token, args.delete_bins,
                      g["away"], g["home"], g["date"]): g
            for g in available
        }
        for fut in as_completed(futures):
            g = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                log(f"  ✗ {g['date']}  pk={g['pk']}  worker exception: {e}")
                failed += 1
                continue
            completed += 1
            elapsed = time.monotonic() - started_at
            done_pct = completed / len(available) * 100
            eta = (elapsed / completed) * (len(available) - completed) if completed else 0
            if result.get("ok"):
                log(f"  [{completed}/{len(available)} {done_pct:.0f}%]  "
                    f"✓ {g['date']}  pk={g['pk']}  {g['away']} @ {g['home']}  "
                    f"{result.get('fetched', 0)}/{result.get('total', 0)} seg "
                    f"(eta {eta/60:.0f}m)")
            else:
                failed += 1
                log(f"  [{completed}/{len(available)} {done_pct:.0f}%]  "
                    f"✗ {g['date']}  pk={g['pk']}  {result.get('error', 'unknown')}")

    log("")
    log(f"=== done in {(time.monotonic()-started_at)/3600:.1f}h ===")
    log(f"  succeeded: {completed - failed}")
    log(f"  failed:    {failed}")


if __name__ == "__main__":
    main()
