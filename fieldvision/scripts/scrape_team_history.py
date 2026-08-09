"""Scrape every available historical game for a team.

The fieldvision-hls.mlbinfra.com binary endpoint retains data for ~3-4
weeks. This script enumerates a team's Final games via statsapi, probes
each for data availability, and downloads + ingests every game that's
still alive.

Usage:
  python scripts/scrape_team_history.py --team-id 114                     # Guardians
  python scripts/scrape_team_history.py --team-id 114 --since 2026-04-01  # specific window
  python scripts/scrape_team_history.py --team-id 114 --delete-bins       # save 7 GB by deleting raw .bin after ingestion

Team IDs:
  114 Cleveland Guardians     147 New York Yankees       108 LA Angels
  136 Seattle Mariners        119 LA Dodgers             117 Houston Astros
  144 Atlanta Braves          146 Miami Marlins          ...
  Full list: https://statsapi.mlb.com/api/v1/teams?sportId=1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldvision.mlb_api import game_base, resolve_manifest
from fieldvision.storage import (_actor_frame_insert_sql, ingest_segment,
                                 load_lookup_tables, open_game_db,
                                 open_registry, transaction, update_registry)


REPO_ROOT = Path(__file__).resolve().parents[1]
# Honor the same env var overrides as fv_daemon, so bulk data can go to
# a path outside the repo (e.g., NAS-mounted bulk store).
SAMPLES_DIR = Path(os.environ.get("FV_SAMPLES_DIR", REPO_ROOT / "samples"))
DATA_DIR = Path(os.environ.get("FV_DATA_DIR", REPO_ROOT / "data"))
STATE_DIR = Path(os.environ.get("FV_STATE_DIR", REPO_ROOT / "state"))
TOKEN_FILE = Path(os.environ.get("FV_TOKEN_FILE", REPO_ROOT / ".fv_token.txt"))


def mark_token_expired(reason: str) -> None:
    """Touch state/token_expired.flag so the Mac watchdog triggers a refresh."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        flag = STATE_DIR / "token_expired.flag"
        flag.write_text(f"{datetime.now().isoformat()} {reason}\n")
    except Exception:
        pass
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# One requests.Session per process, created lazily AFTER fork (keyed by
# pid) so ProcessPool workers never share a TCP connection. Keep-alive
# matters: a fresh TLS handshake per segment costs ~0.2s (measured mean
# 0.488s/GET via urllib vs 0.292s with a reused Session, Nellie 2026-08-09).
# MLB's limiter is a per-connection burst bucket (~85 rapid requests), so
# one paced connection per worker is exactly the safe shape.
_sessions: dict[int, requests.Session] = {}


def _session() -> requests.Session:
    pid = os.getpid()
    s = _sessions.get(pid)
    if s is None:
        s = requests.Session()
        _sessions[pid] = s
    return s


def http_get(url: str, token: str | None = None,
             timeout: int = 30, max_retries: int = 5) -> tuple[int, bytes]:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers.update({
            "Authorization": f"Bearer {token}",
            "x-mannequin-client": "gameday",
            "Origin": "https://www.mlb.com",
            "Referer": "https://www.mlb.com/",
        })
    delay = 1.0
    attempt = 0
    while True:
        try:
            r = _session().get(url, headers=headers, timeout=timeout)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                attempt += 1
                # A broken connection can poison the pool — start fresh.
                _sessions.pop(os.getpid(), None)
                continue
            raise
        # 429 = rate limit: never give up, just wait it out. Retry-After
        # header is authoritative when present; otherwise exponential
        # backoff capped at 30s. Caller wants every segment.
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After") or delay)
            time.sleep(min(wait + 0.1 * min(attempt, 50), 30))
            delay = min(delay * 2, 30)
            attempt += 1
            continue
        # 5xx errors are usually transient too, but might indicate a
        # genuinely broken segment. Keep the bounded retry.
        if r.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
            wait = float(r.headers.get("Retry-After") or delay)
            time.sleep(min(wait + 0.1 * attempt, 30))
            delay = min(delay * 2, 30)
            attempt += 1
            continue
        return r.status_code, r.content


def fetch_team_games(team_id: int, start: str, end: str) -> list[dict]:
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&teamId={team_id}&startDate={start}&endDate={end}"
    )
    s, body = http_get(url)
    if s != 200:
        raise SystemExit(f"statsapi schedule fetch failed: HTTP {s}")
    data = json.loads(body)
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {})
            teams = g.get("teams", {})
            games.append({
                "pk": g.get("gamePk"),
                "date": g.get("officialDate", g.get("gameDate", "")[:10]),
                "away": teams.get("away", {}).get("team", {}).get("name", "?"),
                "home": teams.get("home", {}).get("team", {}).get("name", "?"),
                "state": status.get("abstractGameState"),
                "detailed": status.get("detailedState"),
            })
    return games


def probe_availability(pk: int, token: str) -> dict | None:
    """Returns {"n_segments", "version"} if the game is available, None if
    expired (404 at every known pipeline version)."""
    version, s, body = resolve_manifest(
        pk, lambda u: http_get(u, token, timeout=10, max_retries=2))
    if s == 200 and version:
        try:
            return {"n_segments": len(json.loads(body).get("records", [])),
                    "version": version}
        except Exception:
            return None
    return None


def scrape_one_game(pk: int, token: str, delete_bins: bool,
                    version: str | None = None) -> dict:
    """Download all segments for a game and ingest into per-game Parquet."""
    from fieldvision.storage_parquet import (
        ParquetGameStore, ingest_segment_parquet, ingested_segment_set,
    )

    out_dir = SAMPLES_DIR / f"binary_capture_{pk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest (resolving the pipeline version unless the caller already
    # knows it from probe_availability) + small JSON metadata
    if version is None:
        version, s, manifest_body = resolve_manifest(
            pk, lambda u: http_get(u, token))
    else:
        s, manifest_body = http_get(f"{game_base(pk, version)}/manifest.json", token)
    if s != 200 or version is None:
        # 401/403 => token rejected. Touch the watchdog flag so the
        # Mac refreshes our credentials within ~60s.
        if s in (401, 403):
            mark_token_expired(f"manifest HTTP {s} for pk={pk}")
        return {"ok": False, "error": f"manifest HTTP {s}"}
    base = game_base(pk, version)
    (out_dir / f"mlb_{pk}_manifest.json").write_bytes(manifest_body)
    for name in ("metadata.json", "labels.json"):
        target = out_dir / f"mlb_{pk}_{name}"
        s, body = http_get(f"{base}/{name}", token)
        if s != 200:
            if s in (401, 403):
                mark_token_expired(f"{name} HTTP {s} for pk={pk}")
            return {"ok": False, "error": f"{name} HTTP {s}"}
        target.write_bytes(body)

    manifest = json.loads(manifest_body)
    n_segments = len(manifest.get("records", []))
    log(f"  pk={pk} manifest: {n_segments} segments (v{version}), status={manifest.get('status')}")

    # Gap-aware resume: fetch exactly the non-gap manifest indices not yet
    # ingested. A high-water-mark resume (max+1) permanently loses interior
    # segments that failed mid-run (observed pk=824566: 6 segments dropped
    # below the resume point). Manifest records carry isGap, so known gaps
    # cost nothing to skip. If there's existing data, write to a suffixed
    # file so pyarrow's ParquetWriter doesn't truncate the canonical
    # actor_frames.parquet.
    records = manifest.get("records", [])
    gap_idx = {r.get("index", i) for i, r in enumerate(records) if r.get("isGap")}
    ingested = ingested_segment_set(DATA_DIR, pk)
    new_indices = [i for i in range(n_segments)
                   if i not in ingested and i not in gap_idx]
    suffix = f"resume-{int(time.time())}" if ingested else None
    store = ParquetGameStore(pk, DATA_DIR, append_suffix=suffix)
    metadata = json.loads((out_dir / f"mlb_{pk}_metadata.json").read_text())
    labels = json.loads((out_dir / f"mlb_{pk}_labels.json").read_text())
    labels_dict = store.write_lookups_from_metadata(metadata, labels)
    log(f"  pk={pk} to fetch: {len(new_indices)} of {n_segments} "
        f"({len(gap_idx)} known gaps, {len(ingested)} already ingested)")

    fetched = 0
    failed = 0
    t0 = time.monotonic()
    try:
        for i in new_indices:
            s, body = http_get(f"{base}/{i}.bin", token)
            if s == 404:
                continue
            if s != 200:
                log(f"    ✗ segment {i}: HTTP {s}")
                failed += 1
                if failed > 50:
                    log(f"    too many failures — aborting this game")
                    break
                continue
            bin_path = out_dir / f"mlb_{pk}_segment_{i}.bin"
            bin_path.write_bytes(body)
            try:
                ingest_segment_parquet(store, pk, i, bin_path, labels_dict)
                fetched += 1
                if delete_bins:
                    bin_path.unlink()
            except Exception as e:
                log(f"    ingest segment {i}: {e}")
                failed += 1
            if fetched % 200 == 0 and fetched > 0:
                elapsed = time.monotonic() - t0
                rate = fetched / max(elapsed, 0.1)
                eta = (len(new_indices) - fetched) / max(rate, 0.1)
                log(f"    pk={pk} {fetched}/{len(new_indices)}  ({rate:.1f} seg/s, eta {eta:.0f}s)")
            time.sleep(float(os.environ.get("FV_SLEEP", "0.1")))
    finally:
        store.finalize()

    if delete_bins:
        # Also remove the schema JSONs since the Parquets have everything
        for name in ("manifest.json", "metadata.json", "labels.json"):
            p = out_dir / f"mlb_{pk}_{name}"
            if p.exists():
                p.unlink()
        # Remove the directory if it's empty
        try:
            out_dir.rmdir()
        except OSError:
            pass

    elapsed = time.monotonic() - t0
    log(f"  pk={pk} game done: {fetched} fetched, {failed} failed, "
        f"{n_segments} total in {elapsed / 60:.1f} min "
        f"({fetched / max(elapsed, 0.1):.2f} seg/s)")
    return {"ok": True, "fetched": fetched, "failed": failed,
            "total": n_segments, "elapsed_s": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-id", type=int, required=True,
                        help="MLB teamId (e.g., 114=Guardians, 136=Mariners, 119=Dodgers)")
    parser.add_argument("--since", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 60 days ago)")
    parser.add_argument("--until", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Probe availability and print, don't download")
    parser.add_argument("--delete-bins", action="store_true",
                        help="Delete raw .bin files after ingestion (saves ~200MB/game)")
    parser.add_argument("--skip-game", type=int, action="append", default=[],
                        help="Skip a specific gamePk (can repeat)")
    args = parser.parse_args()

    if not TOKEN_FILE.exists():
        raise SystemExit(f"No token at {TOKEN_FILE}. Run the DevTools snippet.")
    token = TOKEN_FILE.read_text().strip()

    if args.since is None:
        args.since = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    if args.until is None:
        args.until = datetime.now().strftime("%Y-%m-%d")

    log(f"Fetching schedule for teamId={args.team_id}  {args.since} → {args.until}")
    games = fetch_team_games(args.team_id, args.since, args.until)
    final_games = [g for g in games if g["state"] == "Final"
                   and g["pk"] not in args.skip_game]
    log(f"  {len(games)} games total, {len(final_games)} Final (excl. skipped)")

    # Probe availability
    log(f"Probing fieldvision-hls availability...")
    available = []
    unavailable = []
    for g in final_games:
        info = probe_availability(g["pk"], token)
        if info is not None:
            log(f"  ✓ {g['date']}  pk={g['pk']}  {g['away']} @ {g['home']}  "
                f"{info['n_segments']} seg (v{info['version']})")
            available.append({**g, **info})
        else:
            log(f"  ✗ {g['date']}  pk={g['pk']}  expired (404)")
            unavailable.append(g)

    total_seg = sum(g["n_segments"] for g in available)
    log("")
    log(f"=== summary ===")
    log(f"  available: {len(available)} games, {total_seg:,} segments")
    log(f"  expired:   {len(unavailable)} games (>3-4 weeks old)")
    log(f"  estimated time: ~{total_seg * 0.4 / 60:.0f} min ({total_seg * 0.4 / 3600:.1f} h) at 0.3s/seg + ingest")
    log(f"  estimated disk: ~{len(available) * 3.4:.0f} GB SQLite"
        + (f"" if args.delete_bins else f" + ~{len(available) * 0.2:.1f} GB raw bins"))

    if args.dry_run:
        log("dry-run: exiting without downloading")
        return

    log("")
    log("=== starting bulk download ===")
    for i, g in enumerate(available, 1):
        log(f"\n[{i}/{len(available)}] {g['date']}  pk={g['pk']}  {g['away']} @ {g['home']}")
        try:
            result = scrape_one_game(g["pk"], token, args.delete_bins,
                                     version=g.get("version"))
            log(f"  → {result}")
        except KeyboardInterrupt:
            log("KeyboardInterrupt — exiting.")
            return
        except Exception as e:
            log(f"  → FAILED: {e}")

    log("\n=== done ===")


if __name__ == "__main__":
    main()
