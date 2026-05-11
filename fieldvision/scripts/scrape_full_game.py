"""Download every available .bin segment for a live MLB game.

Reads the bearer token from a local file (default ~/Downloads/.fv_token.txt
or .fv_token.txt in the repo root) so the token never crosses the chat
boundary. The token must have audience 'api://mlb_default' — Spencer's
Okta access token. See the DevTools snippet that produces the file.

Concurrency is 8 by default, throttled politely. Output goes to
samples/binary_capture_<gamePk>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error


def find_token(explicit: str | None) -> str:
    """Locate the bearer token. Priority: --token-file → ~/Downloads/.fv_token.txt → repo .fv_token.txt"""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        Path.home() / "Downloads" / ".fv_token.txt",
        Path(__file__).resolve().parents[1] / ".fv_token.txt",
    ]
    for p in candidates:
        if p.exists():
            tok = p.read_text().strip()
            if tok.startswith("eyJ") and tok.count(".") == 2:
                print(f"  using token from {p}  (length={len(tok)})")
                return tok
    raise SystemExit(
        "No token file found. Paste the DevTools snippet to write\n"
        "  ~/Downloads/.fv_token.txt\n"
        "and try again."
    )


def http_get(url: str, headers: dict, timeout: int = 30,
             max_retries: int = 5) -> tuple[int, bytes]:
    """GET with retry-on-429-or-5xx using exponential backoff."""
    delay = 1.0
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                # Backoff and retry; honor Retry-After if MLB sets it
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else delay
                time.sleep(wait + 0.1 * attempt)
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


async def download_one(idx: int, url: str, headers: dict, dest: Path,
                       sem: asyncio.Semaphore, results: dict) -> None:
    async with sem:
        loop = asyncio.get_running_loop()
        status, body = await loop.run_in_executor(None, http_get, url, headers)
        if status == 200:
            dest.write_bytes(body)
            results["done"] += 1
        else:
            results["failed"].append((idx, status))


async def run(args) -> None:
    token = find_token(args.token_file)
    headers = {
        "Authorization": f"Bearer {token}",
        "x-mannequin-client": "gameday",
        "Origin": "https://www.mlb.com",
        "Referer": "https://www.mlb.com/",
        # MLB's edge blocks Python-urllib's default UA with HTTP 403, so we
        # advertise as a desktop Chrome.
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    base = f"https://fieldvision-hls.mlbinfra.com/mannequin/{args.game}/1.6.2"
    out_dir = Path(args.out) / f"binary_capture_{args.game}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Schemas first
    print("Fetching schema files...")
    for name in ("manifest.json", "metadata.json", "labels.json"):
        url = f"{base}/{name}"
        status, body = http_get(url, headers)
        if status != 200:
            raise SystemExit(f"  {name}: HTTP {status} — token may be expired")
        out = out_dir / f"mlb_{args.game}_{name}"
        out.write_bytes(body)
        print(f"  ✓ {name}  ({len(body):,} bytes)")

    manifest = json.loads((out_dir / f"mlb_{args.game}_manifest.json").read_text())
    records = manifest.get("records", [])
    n_segments = len(records)
    print(f"\nManifest: {n_segments} segments, status={manifest.get('status')}")
    print(f"  earliest: {records[0] if records else 'n/a'}")
    print(f"  latest:   {records[-1] if records else 'n/a'}")
    print(f"  duration: {n_segments * 5 / 60:.1f} min")

    if args.limit:
        n_segments = min(n_segments, args.limit)
        print(f"  limiting to first {n_segments} segments per --limit")

    # Skip already-downloaded
    todo = []
    for i in range(n_segments):
        target = out_dir / f"mlb_{args.game}_segment_{i}.bin"
        if target.exists() and target.stat().st_size > 1000:
            continue
        todo.append((i, target))
    print(f"\n{len(todo)} segments to fetch ({n_segments - len(todo)} already present)")
    if not todo:
        return

    sem = asyncio.Semaphore(args.parallel)
    results: dict = {"done": 0, "failed": []}
    t0 = time.monotonic()

    last_progress = t0
    async def progress_logger():
        nonlocal last_progress
        while results["done"] + len(results["failed"]) < len(todo):
            await asyncio.sleep(2)
            now = time.monotonic()
            elapsed = now - t0
            done = results["done"]
            failed = len(results["failed"])
            rate = (done + failed) / max(elapsed, 0.1)
            eta = (len(todo) - done - failed) / max(rate, 0.1)
            print(f"  {done}/{len(todo)}  ({rate:.1f}/s, eta {eta:.0f}s, {failed} failures)", flush=True)

    pl_task = asyncio.create_task(progress_logger())
    tasks = [
        download_one(i, f"{base}/{i}.bin", headers, target, sem, results)
        for i, target in todo
    ]
    await asyncio.gather(*tasks)
    pl_task.cancel()

    elapsed = time.monotonic() - t0
    print(f"\nDone: {results['done']}/{len(todo)} segments in {elapsed:.1f}s")
    if results["failed"]:
        print(f"  Failures ({len(results['failed'])}):")
        for i, status in results["failed"][:10]:
            print(f"    segment {i}: HTTP {status}")
        if len(results["failed"]) > 10:
            print(f"    ...and {len(results['failed']) - 10} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=823141)
    parser.add_argument("--out", default="samples")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, fetch only the first N segments (debug)")
    parser.add_argument("--token-file", help="Path to file containing the bearer JWT")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
