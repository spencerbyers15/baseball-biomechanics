"""Long-running capture daemon for FieldVision skeletal data.

Behavior:
  - Polls statsapi.mlb.com every POLL_INTERVAL_S for today's MLB schedule
  - For each game whose abstractGameState == 'Live' (or about to start):
      - Reads the current api://mlb_default JWT from the persistent profile
      - Polls fieldvision-hls.mlbinfra.com/mannequin/{pk}/1.6.2/manifest.json
      - Downloads any new segments since last poll, saves them to
        samples/binary_capture_{pk}/, ingests into data/fv_{pk}.sqlite
  - Refreshes the JWT periodically (every JWT_REFRESH_S or on 401)
  - Survives MLB rate limits (429) with exponential backoff

Designed to run under launchd (KeepAlive=true, RunAtLoad=true).

State files:
  state/launched.json — gamePks for which lookups have been initialized
  state/last_segment_<pk>.txt — highest segment_idx successfully ingested

Logging:
  scheduler.log     stdout
  scheduler.err     stderr (errors)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.async_api import async_playwright

from fieldvision.storage import (_actor_frame_insert_sql, ingest_segment,
                                 load_lookup_tables, open_game_db,
                                 open_registry, transaction, update_registry)


# ── Config ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / ".mlb_profile"
SAMPLES_DIR = REPO_ROOT / "samples"
DATA_DIR = REPO_ROOT / "data"
STATE_DIR = REPO_ROOT / "state"

SCHEDULE_POLL_S = 600         # 10 min between schedule fetches
SEGMENT_POLL_S = 30           # 30s between manifest polls per live game
JWT_REFRESH_S = 6 * 3600      # refresh JWT every 6h preemptively
JWT_RX = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {msg}",
          file=sys.stderr, flush=True)


# ── JWT extraction ──────────────────────────────────────────────────────────


async def read_token_from_profile() -> str | None:
    """Open the persistent profile briefly (headless), read the api://mlb_default
    access token from localStorage, return it (or None if absent/expired)."""
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=True,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled",
                  "--disable-dev-shm-usage"],
            user_agent=USER_AGENT,
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        # Visiting an MLB page triggers Okta refresh if the session needs it
        await page.goto("https://www.mlb.com/", wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)
        try:
            storage = await page.evaluate(
                """() => {
                    const items = {};
                    for (let i = 0; i < localStorage.length; i++) {
                        const k = localStorage.key(i);
                        items[k] = localStorage.getItem(k);
                    }
                    return JSON.stringify(items);
                }"""
            )
        finally:
            await ctx.close()

    for jwt in set(JWT_RX.findall(storage)):
        try:
            payload = jwt.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            if claims.get("aud") == "api://mlb_default":
                if claims.get("exp", 0) > time.time() + 60:
                    return jwt
        except Exception:
            continue
    return None


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


async def scrape_game_once(game_pk: int, token: str) -> dict:
    """Pull manifest + any new segments + ingest. Returns summary."""
    base = f"https://fieldvision-hls.mlbinfra.com/mannequin/{game_pk}/1.6.2"
    out_dir = SAMPLES_DIR / f"binary_capture_{game_pk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Schemas: fetch on first contact only
    metadata_path = out_dir / f"mlb_{game_pk}_metadata.json"
    labels_path = out_dir / f"mlb_{game_pk}_labels.json"
    manifest_path = out_dir / f"mlb_{game_pk}_manifest.json"

    # Always re-fetch manifest — it grows during a live game
    status, body = http_get(f"{base}/manifest.json", token)
    if status != 200:
        return {"ok": False, "error": f"manifest HTTP {status}"}
    manifest_path.write_bytes(body)
    manifest = json.loads(body)

    if not metadata_path.exists():
        s, b = http_get(f"{base}/metadata.json", token)
        if s != 200:
            return {"ok": False, "error": f"metadata HTTP {s}"}
        metadata_path.write_bytes(b)
    if not labels_path.exists() or manifest.get("status") == "running":
        s, b = http_get(f"{base}/labels.json", token)
        if s == 200:
            labels_path.write_bytes(b)

    n_segments = len(manifest.get("records", []))
    last_seg = get_last_segment(game_pk)
    new_indices = [i for i in range(last_seg + 1, n_segments)]
    if not new_indices:
        return {"ok": True, "new": 0, "manifest_status": manifest.get("status"),
                "total": n_segments}

    # Open DB connections + lookups (only first time per pk)
    conn = open_game_db(game_pk, DATA_DIR)
    cur = conn.execute("SELECT COUNT(*) FROM labels")
    if cur.fetchone()[0] == 0:
        labels_dict = load_lookup_tables(conn, metadata_path, labels_path)
    else:
        labels_dict = {row[0]: {"actor": row[1], "type": row[2]}
                       for row in conn.execute("SELECT actor_uid, actor, actor_type FROM labels")}
    insert_sql = _actor_frame_insert_sql()

    # Download + ingest, throttled
    fetched = 0
    for i in new_indices:
        s, b = http_get(f"{base}/{i}.bin", token)
        if s == 404:
            # Gap segment — skip
            set_last_segment(game_pk, i)
            continue
        if s != 200:
            err(f"  game {game_pk} segment {i}: HTTP {s}")
            break
        bin_path = out_dir / f"mlb_{game_pk}_segment_{i}.bin"
        bin_path.write_bytes(b)
        try:
            with transaction(conn):
                ingest_segment(conn, game_pk, i, bin_path, labels_dict, insert_sql)
            set_last_segment(game_pk, i)
            fetched += 1
        except Exception as e:
            err(f"  ingest segment {i} for game {game_pk}: {e}")
        # Be polite to MLB's CDN
        time.sleep(0.3)

    # Update registry
    reg = open_registry(DATA_DIR)
    update_registry(reg, conn, game_pk, DATA_DIR / f"fv_{game_pk}.sqlite")
    reg.close()
    conn.close()

    return {"ok": True, "new": fetched, "manifest_status": manifest.get("status"),
            "total": n_segments}


# ── Main loop ───────────────────────────────────────────────────────────────


async def run_forever(args):
    log(f"FieldVision capture daemon starting")
    log(f"  profile: {PROFILE_DIR}")
    log(f"  samples: {SAMPLES_DIR}")
    log(f"  data:    {DATA_DIR}")
    log(f"  state:   {STATE_DIR}")

    if not PROFILE_DIR.exists():
        err("Persistent profile not found. Run: python scripts/setup_login.py")
        sys.exit(1)

    token: str | None = None
    token_acquired_at = 0.0

    while True:
        try:
            # Refresh token if expired or stale
            if token is None or (time.time() - token_acquired_at) > JWT_REFRESH_S:
                log("Refreshing JWT from persistent profile...")
                token = await read_token_from_profile()
                if token is None:
                    err("Failed to read JWT. Re-run setup_login.py.")
                    await asyncio.sleep(SCHEDULE_POLL_S)
                    continue
                token_acquired_at = time.time()
                log(f"  got token (length {len(token)})")

            # Fetch today's schedule
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
                await asyncio.sleep(SCHEDULE_POLL_S)
                continue

            # Scrape each live game until none have new segments, then sleep
            for game in live:
                pk = game["gamePk"]
                log(f"  → game {pk}: {game['away']} @ {game['home']}")
                try:
                    summary = await scrape_game_once(pk, token)
                except Exception as e:
                    err(f"  game {pk} scrape failed: {e}")
                    summary = {"ok": False, "error": str(e)}
                if summary.get("ok") and summary.get("new", 0) == 0 and "401" in str(summary.get("error", "")):
                    log("  401 detected — forcing token refresh")
                    token = None
                    continue
                log(f"     {summary}")

            await asyncio.sleep(SEGMENT_POLL_S)

        except KeyboardInterrupt:
            log("KeyboardInterrupt — exiting.")
            return
        except Exception as e:
            err(f"main loop: {e}")
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
