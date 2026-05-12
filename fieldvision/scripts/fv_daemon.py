"""Long-running capture daemon for FieldVision skeletal data.

Reads the JWT from .fv_token.txt (in repo root) — refreshed by you via
the DevTools-paste snippet, no Playwright needed.

Behavior:
  - Polls statsapi.mlb.com every SCHEDULE_POLL_S for today's MLB schedule
  - For each game whose abstractGameState == 'Live':
      - Polls fieldvision-hls.mlbinfra.com/mannequin/{pk}/1.6.2/manifest.json
      - Downloads any new segments since last poll (samples/binary_capture_<pk>/)
      - Ingests each segment immediately into data/fv_<pk>.sqlite
  - Survives MLB rate limits (429) with exponential backoff
  - On HTTP 401: logs a clear "TOKEN EXPIRED" message and waits for refresh

Designed to run under launchd (KeepAlive=true, RunAtLoad=true).

State files (in state/):
  last_segment_<pk>.txt  — highest segment_idx successfully ingested per game
  token_expired.flag     — touched when daemon detects 401, removed when fresh
                           token works (so external tools can react)

Token file: .fv_token.txt at the repo root, refreshed via the DevTools
snippet at scripts/snippets/refresh_token.js (also reproduced in the
FVCAPTURE_SETUP.md runbook).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldvision.storage import (_actor_frame_insert_sql, _pitch_event_insert_sql,
                                 ingest_segment, load_lookup_tables, open_game_db,
                                 open_registry, transaction, update_registry)


REPO_ROOT = Path(__file__).resolve().parents[1]
# data/samples/state can be redirected via env vars when running on a host
# where the repo lives in one place but bulk storage lives elsewhere
# (e.g., Nellie: repo on NAS read-mostly, data on NAS at a separate path).
SAMPLES_DIR = Path(os.environ.get("FV_SAMPLES_DIR", REPO_ROOT / "samples"))
DATA_DIR = Path(os.environ.get("FV_DATA_DIR", REPO_ROOT / "data"))
STATE_DIR = Path(os.environ.get("FV_STATE_DIR", REPO_ROOT / "state"))
TOKEN_FILE = REPO_ROOT / ".fv_token.txt"

SCHEDULE_POLL_S = 600         # 10 min between schedule fetches
SEGMENT_POLL_S = 30           # 30s between manifest polls per live game
TOKEN_RECHECK_S = 300         # 5 min between token reads (cheap; reads file)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {msg}",
          file=sys.stderr, flush=True)


def state_path(name: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / name


def get_last_segment(game_pk: int) -> int:
    p = state_path(f"last_segment_{game_pk}.txt")
    if p.exists():
        try:
            return int(p.read_text().strip())
        except Exception:
            return -1
    return -1


def set_last_segment(game_pk: int, idx: int) -> None:
    state_path(f"last_segment_{game_pk}.txt").write_text(str(idx))


# ── Token loading + validation ─────────────────────────────────────────────


def read_token() -> tuple[str | None, str]:
    """Returns (token, status). Status is one of:
    'ok', 'missing', 'malformed', 'expired'."""
    if not TOKEN_FILE.exists():
        return None, "missing"
    try:
        raw = TOKEN_FILE.read_text().strip()
    except Exception:
        return None, "missing"
    if not raw or not raw.startswith("eyJ") or raw.count(".") != 2:
        return None, "malformed"
    try:
        payload = raw.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None, "malformed"
    if claims.get("exp", 0) < time.time() + 60:
        return None, "expired"
    if claims.get("aud") != "api://mlb_default":
        return None, "malformed"
    return raw, "ok"


def mark_token_expired(reason: str) -> None:
    flag = state_path("token_expired.flag")
    flag.write_text(f"{datetime.now().isoformat()}\n{reason}\n")
    err(f"TOKEN UNAVAILABLE ({reason}). Refresh by pasting the DevTools snippet "
        f"and writing ~/Downloads/fv_token.txt → mv to {TOKEN_FILE}")


def clear_token_flag() -> None:
    flag = state_path("token_expired.flag")
    if flag.exists():
        flag.unlink()


# ── Schedule + scrape ──────────────────────────────────────────────────────


def fetch_schedule(date_str: str | None = None) -> list[dict]:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            pk = g.get("gamePk")
            if not pk:
                continue
            status = g.get("status", {})
            teams = g.get("teams", {})
            games.append({
                "gamePk": pk,
                "abstract": status.get("abstractGameState"),
                "detailed": status.get("detailedState"),
                "gameDate": g.get("gameDate"),
                "away": teams.get("away", {}).get("team", {}).get("name", "?"),
                "home": teams.get("home", {}).get("team", {}).get("name", "?"),
            })
    return games


def http_get(url: str, token: str, timeout: int = 30,
             max_retries: int = 5) -> tuple[int, bytes]:
    headers = {
        "Authorization": f"Bearer {token}",
        "x-mannequin-client": "gameday",
        "Origin": "https://www.mlb.com",
        "Referer": "https://www.mlb.com/",
        "User-Agent": USER_AGENT,
    }
    delay = 1.0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = float(e.headers.get("Retry-After") or delay)
                time.sleep(min(wait + 0.1 * attempt, 30))
                delay = min(delay * 2, 30)
                continue
            try:
                body = e.read()
            except Exception:
                body = b""
            return e.code, body
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    return 0, b""


def scrape_game_once(game_pk: int, token: str) -> dict:
    """Pull manifest + any new segments + ingest. Returns summary."""
    base = f"https://fieldvision-hls.mlbinfra.com/mannequin/{game_pk}/1.6.2"
    out_dir = SAMPLES_DIR / f"binary_capture_{game_pk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / f"mlb_{game_pk}_metadata.json"
    labels_path = out_dir / f"mlb_{game_pk}_labels.json"
    manifest_path = out_dir / f"mlb_{game_pk}_manifest.json"

    status, body = http_get(f"{base}/manifest.json", token)
    if status == 401:
        return {"ok": False, "auth_failed": True, "error": "manifest 401"}
    if status != 200:
        return {"ok": False, "error": f"manifest HTTP {status}"}
    manifest_path.write_bytes(body)
    manifest = json.loads(body)

    if not metadata_path.exists():
        s, b = http_get(f"{base}/metadata.json", token)
        if s == 401:
            return {"ok": False, "auth_failed": True, "error": "metadata 401"}
        if s != 200:
            return {"ok": False, "error": f"metadata HTTP {s}"}
        metadata_path.write_bytes(b)
    if not labels_path.exists() or manifest.get("status") == "running":
        s, b = http_get(f"{base}/labels.json", token)
        if s == 200:
            labels_path.write_bytes(b)

    from fieldvision.storage_parquet import (
        ParquetGameStore, ingest_segment_parquet, max_segment_idx_for_game,
    )

    n_segments = len(manifest.get("records", []))
    # Resume: use the higher of (state file's last_segment, max segment in
    # Parquet). The state file is updated even on 404 segments; the Parquet
    # max reflects only successful ingests.
    last_seg = max(get_last_segment(game_pk),
                   max_segment_idx_for_game(DATA_DIR, game_pk))
    new_indices = [i for i in range(last_seg + 1, n_segments)]
    if not new_indices:
        return {"ok": True, "new": 0, "manifest_status": manifest.get("status"),
                "total": n_segments}

    # Open Parquet store and write lookups (idempotent — overwrites)
    store = ParquetGameStore(game_pk, DATA_DIR)
    metadata = json.loads(metadata_path.read_text())
    labels = json.loads(labels_path.read_text())
    labels_dict = store.write_lookups_from_metadata(metadata, labels)

    fetched = 0
    auth_failed = False
    try:
        for i in new_indices:
            s, b = http_get(f"{base}/{i}.bin", token)
            if s == 401:
                auth_failed = True
                break
            if s == 404:
                set_last_segment(game_pk, i)
                continue
            if s != 200:
                err(f"  game {game_pk} segment {i}: HTTP {s}")
                break
            bin_path = out_dir / f"mlb_{game_pk}_segment_{i}.bin"
            bin_path.write_bytes(b)
            try:
                ingest_segment_parquet(store, game_pk, i, bin_path, labels_dict)
                set_last_segment(game_pk, i)
                fetched += 1
            except Exception as e:
                err(f"  ingest segment {i} for game {game_pk}: {e}")
            time.sleep(0.3)
    finally:
        store.finalize()

    return {"ok": True, "auth_failed": auth_failed, "new": fetched,
            "manifest_status": manifest.get("status"), "total": n_segments}


# ── Main loop ───────────────────────────────────────────────────────────────


async def run_forever(args):
    log(f"FieldVision capture daemon starting")
    log(f"  token file: {TOKEN_FILE}")
    log(f"  samples:    {SAMPLES_DIR}")
    log(f"  data:       {DATA_DIR}")

    while True:
        try:
            token, status = read_token()
            if status != "ok":
                mark_token_expired(status)
                if args.once:
                    return
                await asyncio.sleep(TOKEN_RECHECK_S)
                continue
            clear_token_flag()

            try:
                games = fetch_schedule()
            except Exception as e:
                err(f"Schedule fetch failed: {e}")
                await asyncio.sleep(60)
                continue

            live = [g for g in games if g["abstract"] == "Live"]
            log(f"Schedule: {len(games)} games, {len(live)} live")

            if args.teams:
                filt = set(t.lower() for t in args.teams.split(","))
                live = [g for g in live
                        if any(t in g["away"].lower() or t in g["home"].lower() for t in filt)]
                log(f"  after team filter: {len(live)}")

            if not live:
                log("  no live games. Sleeping.")
                if args.once:
                    return
                await asyncio.sleep(SCHEDULE_POLL_S)
                continue

            for game in live:
                pk = game["gamePk"]
                log(f"  → game {pk}: {game['away']} @ {game['home']}")
                try:
                    summary = scrape_game_once(pk, token)
                except Exception as e:
                    err(f"  game {pk} scrape failed: {e}")
                    summary = {"ok": False, "error": str(e)}
                if summary.get("auth_failed"):
                    log("  401 — token expired or revoked. Marking flag.")
                    mark_token_expired("401-from-fieldvision-hls")
                    break
                log(f"     {summary}")

            if args.once:
                return
            await asyncio.sleep(SEGMENT_POLL_S)

        except KeyboardInterrupt:
            log("KeyboardInterrupt — exiting.")
            return
        except Exception as e:
            err(f"main loop: {e}")
            if args.once:
                return
            await asyncio.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", help="Comma-separated team substrings (e.g. 'Mariners,Dodgers')")
    parser.add_argument("--once", action="store_true",
                        help="Run one schedule poll then exit (debug)")
    args = parser.parse_args()
    asyncio.run(run_forever(args))


if __name__ == "__main__":
    main()
