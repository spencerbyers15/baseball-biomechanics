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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldvision.storage import (_actor_frame_insert_sql, ingest_segment,
                                 load_lookup_tables, open_game_db,
                                 open_registry, transaction, update_registry)


REPO_ROOT = Path(__file__).resolve().parents[1]
# Honor the same env var overrides as fv_daemon, so bulk data can go to
# a path outside the repo (e.g., NAS-mounted bulk store).
SAMPLES_DIR = Path(os.environ.get("FV_SAMPLES_DIR", REPO_ROOT / "samples"))
DATA_DIR = Path(os.environ.get("FV_DATA_DIR", REPO_ROOT / "data"))
TOKEN_FILE = REPO_ROOT / ".fv_token.txt"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def probe_availability(pk: int, token: str) -> int | None:
    """Returns segment count if game is available, None if 404."""
    url = f"https://fieldvision-hls.mlbinfra.com/mannequin/{pk}/1.6.2/manifest.json"
    s, body = http_get(url, token, timeout=10, max_retries=2)
    if s == 200:
        try:
            return len(json.loads(body).get("records", []))
        except Exception:
            return None
    return None


def scrape_one_game(pk: int, token: str, delete_bins: bool) -> dict:
    """Download all segments for a game and ingest into SQLite."""
    base = f"https://fieldvision-hls.mlbinfra.com/mannequin/{pk}/1.6.2"
    out_dir = SAMPLES_DIR / f"binary_capture_{pk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Schemas
    for name in ("manifest.json", "metadata.json", "labels.json"):
        target = out_dir / f"mlb_{pk}_{name}"
        s, body = http_get(f"{base}/{name}", token)
        if s != 200:
            return {"ok": False, "error": f"{name} HTTP {s}"}
        target.write_bytes(body)

    manifest = json.loads((out_dir / f"mlb_{pk}_manifest.json").read_text())
    n_segments = len(manifest.get("records", []))
    log(f"  manifest: {n_segments} segments, status={manifest.get('status')}")

    # Open DB + load lookups
    conn = open_game_db(pk, DATA_DIR)
    cur = conn.execute("SELECT COUNT(*) FROM labels")
    if cur.fetchone()[0] == 0:
        labels_dict = load_lookup_tables(
            conn, out_dir / f"mlb_{pk}_metadata.json", out_dir / f"mlb_{pk}_labels.json"
        )
    else:
        labels_dict = {row[0]: {"actor": row[1], "type": row[2]}
                       for row in conn.execute("SELECT actor_uid, actor, actor_type FROM labels")}
    insert_sql = _actor_frame_insert_sql()

    # Determine which segments need downloading
    cur = conn.execute("SELECT MAX(segment_idx) FROM actor_frame")
    row = cur.fetchone()
    last_in_db = row[0] if row[0] is not None else -1
    new_indices = [i for i in range(last_in_db + 1, n_segments)]
    log(f"  ingest range: segments {last_in_db + 1}..{n_segments - 1}  ({len(new_indices)} to fetch)")

    fetched = 0
    failed = 0
    t0 = time.monotonic()
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
            with transaction(conn):
                ingest_segment(conn, pk, i, bin_path, labels_dict, insert_sql)
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
            log(f"    {fetched}/{len(new_indices)}  ({rate:.1f} seg/s, eta {eta:.0f}s)")
        time.sleep(0.3)

    reg = open_registry(DATA_DIR)
    update_registry(reg, conn, pk, DATA_DIR / f"fv_{pk}.sqlite")
    reg.close()
    conn.close()

    if delete_bins:
        # Also remove the schema JSONs since the SQLite has it all
        for name in ("manifest.json", "metadata.json", "labels.json"):
            p = out_dir / f"mlb_{pk}_{name}"
            if p.exists():
                p.unlink()
        # Remove the directory if it's empty
        try:
            out_dir.rmdir()
        except OSError:
            pass

    return {"ok": True, "fetched": fetched, "failed": failed, "total": n_segments}


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
        n = probe_availability(g["pk"], token)
        if n is not None:
            log(f"  ✓ {g['date']}  pk={g['pk']}  {g['away']} @ {g['home']}  {n} seg")
            available.append({**g, "n_segments": n})
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
            result = scrape_one_game(g["pk"], token, args.delete_bins)
            log(f"  → {result}")
        except KeyboardInterrupt:
            log("KeyboardInterrupt — exiting.")
            return
        except Exception as e:
            log(f"  → FAILED: {e}")

    log("\n=== done ===")


if __name__ == "__main__":
    main()
