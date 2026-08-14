"""Autopilot: keep FieldVision capture caught up, forever.

One process, one worker pool, two feeds:

  LIVE   — every tick, each currently-live MLB game gets a scrape pass
           (scrape_one_game resumes from the last ingested segment, so a
           pass just picks up the new segments). Live games always get
           pool slots first.
  BACKLOG— idle slots stream the historical backlog oldest-first
           (games nearest MLB's retention horizon are rescued first),
           from FV_BACKLOG_START (default 2026-05-20, the day capture
           originally died) through yesterday.

Completed games get a marker in state/ so they're never re-examined.
Games already on disk without a marker get one cheap resume pass (a few
hundred bytes of manifest traffic) which verifies nothing new is needed
and writes the marker.

Token: workers read FV_TOKEN_FILE per pass; on 401/403 they touch
state/token_expired.flag (the Mac watchdog refreshes + scp's within ~60s)
and the pass is retried next tick.

Run on Nellie under tmux (see launchd/nellie_autopilot_cron.txt):
  FV_TOKEN_FILE=/home/spencer/.fv_token.txt \
  FV_DATA_DIR=/media/datasets/spencer/fieldvision/data \
  FV_SAMPLES_DIR=/media/scratch/spencer/fieldvision/samples \
  FV_STATE_DIR=/media/scratch/spencer/fieldvision/state \
  python3 scripts/fv_autopilot.py --workers 8 --delete-bins
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scrape_team_history import http_get, scrape_one_game  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FV_DATA_DIR", REPO_ROOT / "data"))
STATE_DIR = Path(os.environ.get("FV_STATE_DIR", REPO_ROOT / "state"))
TOKEN_FILE = Path(os.environ.get("FV_TOKEN_FILE", REPO_ROOT / ".fv_token.txt"))
BACKLOG_START = os.environ.get("FV_BACKLOG_START", "2026-05-20")

TICK_S = 60
LIVE_PASS_S = 300  # per-game gap between live passes; each resume pass with
                   # new data writes a suffixed parquet, so don't over-tick
SCHEDULE_REFRESH_S = 600
BACKLOG_REFRESH_S = 6 * 3600
LIVE_404_GIVE_UP = 3  # consecutive manifest-404 passes on a Final game before
                      # we accept it has no Hawk-Eye feed and stop retrying


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_token() -> str | None:
    try:
        tok = TOKEN_FILE.read_text().strip()
    except OSError:
        return None
    return tok if tok.startswith("eyJ") and tok.count(".") == 2 else None


def complete_marker(pk: int) -> Path:
    return STATE_DIR / f"complete_{pk}.marker"


def live_404_streak(game_state: str | None, result: dict, prev: int) -> int:
    """Consecutive manifest-404 passes for a Final game; 0 breaks the streak.

    A Final game that keeps 404ing has no manifest and never will — some games
    are simply never rigged for Hawk-Eye (special venues: Field of Dreams,
    Little League Classic). The backlog path already marks 404s complete, but
    the live/closing path did not, so such a game was re-submitted every cycle
    forever (observed pk=823669, the 2026-08-13 Field of Dreams game: 107
    futile passes in one day, each burning a worker slot).

    Only Final games count, and only *consecutive* 404s, so a transient blip
    on a real game can never skip it permanently.
    """
    if (game_state == "Final" and not result.get("ok")
            and "manifest HTTP 404" in str(result.get("error", ""))):
        return prev + 1
    return 0


def fetch_schedule_window() -> list[dict]:
    """Yesterday → today, so late-night finishers aren't orphaned."""
    start = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&startDate={start}&endDate={end}")
    s, body = http_get(url)
    if s != 200:
        raise RuntimeError(f"schedule HTTP {s}")
    out = []
    for d in json.loads(body).get("dates", []):
        for g in d.get("games", []):
            out.append({
                "pk": g["gamePk"],
                "date": g.get("officialDate", ""),
                "state": g.get("status", {}).get("abstractGameState"),
            })
    return out


def fetch_backlog() -> list[int]:
    """Final games from BACKLOG_START through yesterday without a
    complete-marker, oldest first."""
    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&startDate={BACKLOG_START}&endDate={end}")
    s, body = http_get(url)
    if s != 200:
        raise RuntimeError(f"backlog schedule HTTP {s}")
    games = []
    for d in json.loads(body).get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            games.append((g.get("officialDate", ""), g["gamePk"]))
    games.sort()
    return [pk for _, pk in games if not complete_marker(pk).exists()]


def capture_pass(pk: int, delete_bins: bool) -> dict:
    """Worker: one scrape pass over one game (resumes where it left off)."""
    token = read_token()
    if token is None:
        return {"pk": pk, "ok": False, "error": "no token"}
    try:
        r = scrape_one_game(pk, token, delete_bins)
    except Exception as e:
        return {"pk": pk, "ok": False, "error": f"exception: {e}"}
    return {"pk": pk, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backlog-workers", type=int, default=None,
                    help="Max slots the backlog may occupy (default workers-2)")
    ap.add_argument("--delete-bins", action="store_true")
    args = ap.parse_args()
    backlog_cap = (args.backlog_workers if args.backlog_workers is not None
                   else max(1, args.workers - 2))

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"autopilot up: {args.workers} workers ({backlog_cap} max for backlog), "
        f"backlog start {BACKLOG_START}, data {DATA_DIR}")

    schedule: list[dict] = []
    backlog: list[int] = []
    sched_at = backlog_at = 0.0
    inflight: dict[int, tuple[Future, str]] = {}  # pk -> (future, kind)
    live_quiet: dict[int, int] = {}  # pk -> consecutive no-new passes while Final
    live_404: dict[int, int] = {}  # pk -> consecutive manifest-404 passes while Final
    last_live_pass: dict[int, float] = {}  # pk -> monotonic time of last submit

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            now = time.time()
            try:
                if now - sched_at > SCHEDULE_REFRESH_S:
                    schedule = fetch_schedule_window()
                    sched_at = now
                if now - backlog_at > BACKLOG_REFRESH_S:
                    backlog = fetch_backlog()
                    backlog_at = now
                    log(f"backlog refreshed: {len(backlog)} games pending")
            except Exception as e:
                log(f"ERROR refreshing schedule/backlog: {e}")

            # Reap finished passes
            for pk in list(inflight):
                fut, kind = inflight[pk]
                if not fut.done():
                    continue
                del inflight[pk]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"pk": pk, "ok": False, "error": f"crash: {e}"}
                game_state = next((g["state"] for g in schedule
                                   if g["pk"] == pk), None)
                if kind == "backlog":
                    # ok can be True even when the per-segment loop aborted
                    # after >50 failures — only a zero-failure pass counts
                    # as complete.
                    if r.get("ok") and r.get("failed", 0) == 0:
                        complete_marker(pk).write_text(json.dumps(r) + "\n")
                        log(f"backlog pk={pk} DONE fetched={r.get('fetched')} "
                            f"total={r.get('total')}")
                    elif "manifest HTTP 404" in str(r.get("error", "")):
                        # Expired from MLB retention — permanently gone, do
                        # not burn requests re-probing it every refresh.
                        complete_marker(pk).write_text(json.dumps(r) + "\n")
                        log(f"backlog pk={pk} EXPIRED from retention — marked")
                    else:
                        log(f"backlog pk={pk} FAILED: {r.get('error') or r} "
                            f"(will retry on next backlog refresh)")
                else:  # live
                    if r.get("ok") and game_state == "Final" and r.get("new", r.get("fetched", 0)) == 0:
                        live_quiet[pk] = live_quiet.get(pk, 0) + 1
                        if live_quiet[pk] >= 2:
                            complete_marker(pk).write_text(json.dumps(r) + "\n")
                            live_quiet.pop(pk, None)
                            log(f"live pk={pk} finalized + marked complete")
                    else:
                        live_quiet.pop(pk, None)
                    if not r.get("ok"):
                        log(f"live pk={pk} pass failed: {r.get('error')}")
                    live_404[pk] = live_404_streak(game_state, r,
                                                   live_404.get(pk, 0))
                    if live_404[pk] >= LIVE_404_GIVE_UP:
                        complete_marker(pk).write_text(json.dumps(r) + "\n")
                        live_404.pop(pk, None)
                        log(f"live pk={pk} no manifest after "
                            f"{LIVE_404_GIVE_UP} passes (venue likely has no "
                            f"Hawk-Eye) — marked, will not retry")

            # Submit live passes (priority)
            live_pks = [g["pk"] for g in schedule if g["state"] == "Live"]
            # Recently-Final games that aren't marked complete still need
            # closing passes.
            closing = [g["pk"] for g in schedule
                       if g["state"] == "Final"
                       and not complete_marker(g["pk"]).exists()]
            for pk in live_pks + closing:
                if pk in inflight or len(inflight) >= args.workers:
                    continue
                if time.monotonic() - last_live_pass.get(pk, 0) < LIVE_PASS_S:
                    continue
                last_live_pass[pk] = time.monotonic()
                inflight[pk] = (pool.submit(capture_pass, pk, args.delete_bins),
                                "live")

            # Fill idle slots with backlog, oldest first
            backlog_inflight = sum(1 for _, k in inflight.values() if k == "backlog")
            for pk in backlog:
                if (len(inflight) >= args.workers
                        or backlog_inflight >= backlog_cap):
                    break
                if pk in inflight or complete_marker(pk).exists():
                    continue
                inflight[pk] = (pool.submit(capture_pass, pk, args.delete_bins),
                                "backlog")
                backlog_inflight += 1
            backlog = [pk for pk in backlog
                       if not complete_marker(pk).exists()]

            time.sleep(TICK_S)


if __name__ == "__main__":
    main()
