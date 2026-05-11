"""Probe whether Spencer's authenticated MLB.com session mints any JWT
that the legacy fieldvision-hls.mlbinfra.com binary endpoint accepts.

Flow (with --headed, the default for this script):
  1. Open a Playwright Chromium with a persistent profile dir at
     ~/fieldvision/.mlb_profile so the login survives
     across runs.
  2. Navigate to https://www.mlb.com/login.
  3. Wait up to LOGIN_TIMEOUT seconds for the user to log in and for
     mlb.com to redirect / show a logged-in indicator.
  4. Once logged in, navigate to the live gameday URL.
  5. Sniff cookies / localStorage / sessionStorage / request headers /
     request bodies / response bodies for JWTs.
  6. For each unique JWT found, replay it against
     https://fieldvision-hls.mlbinfra.com/mannequin/{pk}/1.6.2/manifest.json
     and report which (if any) succeed.

Privacy / token handling:
  - Full JWTs are NEVER written to disk by this script.
  - The output JSON contains only:
      * a SHA-256 hash of each JWT (so we can dedup across runs)
      * the first 30 + last 20 characters (for human triage)
      * the decoded NON-secret claims (iss/aud/sub/exp/scope/etc.)
      * the HTTP status the endpoint returned for that token
  - If a JWT returns 200, the script holds the working JWT in memory
    only long enough to also fetch metadata.json, labels.json, and the
    first segment as a proof-of-life capture, then forgets it. The
    files are saved; the token is not.

Usage:
  python scripts/auth_probe.py                # default: game 823141, headed
  python scripts/auth_probe.py --game 824198
  python scripts/auth_probe.py --no-headed    # only useful if profile already logged-in
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright


JWT_RX = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
LOGIN_TIMEOUT_S = 180  # 3 minutes for the user to complete login + 2FA
SNIFF_DURATION_S = 30


def find_jwts(blob: str) -> list[str]:
    if not blob:
        return []
    return list(set(JWT_RX.findall(blob)))


def jwt_fingerprint(jwt: str) -> dict:
    return {
        "hash": hashlib.sha256(jwt.encode()).hexdigest()[:16],
        "head": jwt[:30],
        "tail": "..." + jwt[-20:],
        "length": len(jwt),
    }


def decode_jwt_claims(jwt: str) -> dict | None:
    try:
        parts = jwt.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None


def safe_claims(claims: dict | None) -> dict:
    """Pull only the non-secret structural claims worth inspecting."""
    if not claims:
        return {}
    keep = (
        "iss", "aud", "sub", "exp", "iat", "nbf", "scope", "scopes",
        "client_id", "azp", "kid", "typ", "permissions",
        "userId", "userid", "mlbuserId", "mlbUserId", "mlb_user_id",
        "uid", "user_id", "tier", "subscriptions",
    )
    out: dict = {}
    for k, v in claims.items():
        if k in keep:
            if isinstance(v, str) and len(v) > 200:
                v = v[:200] + "..."
            out[k] = v
    return out


def test_token_against_fv(jwt: str, game_pk: int) -> dict:
    url = f"https://fieldvision-hls.mlbinfra.com/mannequin/{game_pk}/1.6.2/manifest.json"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {jwt}",
                "x-mannequin-client": "gameday",
                "Origin": "https://www.mlb.com",
                "Referer": "https://www.mlb.com/",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status, "body": resp.read()[:500].decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as e:
        try:
            body = e.read()[:500].decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return {"status": e.code, "body": body}
    except Exception as e:
        return {"status": -1, "body": f"<{e}>"}


async def wait_for_login(page, timeout_s: int) -> bool:
    """Poll the page for indicators the user is logged in."""
    print(f"\n  Browser is open. Please log in to MLB.com in the window.")
    print(f"  Waiting up to {timeout_s}s for login to complete...")
    print(f"  (This script never sees your password — Playwright runs the browser; we read cookies AFTER you log in.)")
    print()

    indicators = [
        # mlb.com puts your username/profile menu in the header when logged in
        '[data-testid*="user-menu"]',
        '[data-testid*="profile-menu"]',
        '[aria-label*="Account"]',
        '[aria-label*="profile"i]',
        '[class*="UserMenu"]',
        '[class*="ProfileMenu"]',
        # Or look for a "Sign Out" link
        'a:has-text("Sign Out")',
        'button:has-text("Sign Out")',
    ]

    elapsed = 0
    while elapsed < timeout_s:
        for sel in indicators:
            try:
                if await page.locator(sel).count() > 0:
                    print(f"  Detected login (matched: {sel!r})")
                    return True
            except Exception:
                continue
        # Also check cookies for a logged-in marker
        try:
            cookies = await page.context.cookies()
            cookie_names = {c["name"] for c in cookies}
            # Common MLB auth cookie names
            if any(name in cookie_names for name in ("Identity", "okta-token-storage", "mlb_user", "ipid")):
                # ipid is set even for anon users; but Identity / okta tokens are post-login
                if any(name in cookie_names for name in ("Identity", "okta-token-storage")):
                    print(f"  Detected login (auth cookie present)")
                    return True
        except Exception:
            pass

        await asyncio.sleep(2)
        elapsed += 2

    print(f"  Timed out waiting for login indicators after {timeout_s}s.")
    print(f"  Will proceed anyway — if you ARE logged in but the indicator wasn't detected, the sniff will still work.")
    return False


async def probe(game_pk: int, headed: bool, profile_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    found_jwts: dict[str, list[str]] = defaultdict(list)  # jwt -> [source descriptions]

    def add_jwts(source: str, blob: str) -> None:
        for j in find_jwts(blob):
            found_jwts[j].append(source)

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=not headed,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"],
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        async def on_request(req):
            try:
                headers = await req.all_headers()
                add_jwts(f"req-hdr  {req.method} {req.url[:60]}",
                         "\n".join(f"{k}: {v}" for k, v in headers.items()))
                if req.post_data:
                    add_jwts(f"req-body {req.method} {req.url[:60]}", req.post_data)
            except Exception:
                pass

        async def on_response(resp):
            try:
                ct = (resp.headers.get("content-type") or "").lower()
                if any(t in ct for t in ("json", "text", "javascript", "html", "xml")):
                    body = await resp.text()
                    add_jwts(f"resp-body {resp.status} {resp.url[:60]}", body)
                add_jwts(f"resp-hdr  {resp.status} {resp.url[:60]}",
                         "\n".join(f"{k}: {v}" for k, v in resp.headers.items()))
            except Exception:
                pass

        page.on("request", lambda r: asyncio.create_task(on_request(r)))
        page.on("response", lambda r: asyncio.create_task(on_response(r)))

        # 1. Drive the login
        print("Step 1/4: navigating to mlb.com login")
        await page.goto("https://www.mlb.com/login", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        await wait_for_login(page, LOGIN_TIMEOUT_S)

        # 2. Navigate to the gameday URL so the page exercises whatever
        # auth flow it does for live games
        print(f"\nStep 2/4: navigating to gameday for game {game_pk}")
        await page.goto(f"https://www.mlb.com/gameday/{game_pk}/live",
                        wait_until="domcontentloaded", timeout=60000)
        print(f"  sniffing for {SNIFF_DURATION_S}s...")
        await page.wait_for_timeout(SNIFF_DURATION_S * 1000)

        # 3. Sweep storage
        print("\nStep 3/4: sweeping cookies / localStorage / sessionStorage")
        cookies = await ctx.cookies()
        add_jwts("cookie", "\n".join(f"{c['name']}={c['value']}" for c in cookies))
        try:
            storage = await page.evaluate(
                """() => {
                    const ls = {}, ss = {};
                    try { for (let i = 0; i < localStorage.length; i++) {
                        const k = localStorage.key(i); ls[k] = localStorage.getItem(k);
                    } } catch(e) {}
                    try { for (let i = 0; i < sessionStorage.length; i++) {
                        const k = sessionStorage.key(i); ss[k] = sessionStorage.getItem(k);
                    } } catch(e) {}
                    return { ls, ss };
                }"""
            )
            for k, v in (storage.get("ls") or {}).items():
                add_jwts(f"localStorage[{k}]", v or "")
            for k, v in (storage.get("ss") or {}).items():
                add_jwts(f"sessionStorage[{k}]", v or "")
        except Exception as e:
            print(f"  storage sweep failed: {e}")

        await ctx.close()

    # 4. Test each unique JWT
    print(f"\nStep 4/4: testing {len(found_jwts)} unique JWT(s) against fieldvision-hls")
    results = []
    successful_jwt: str | None = None
    for jwt, sources in sorted(found_jwts.items()):
        fp = jwt_fingerprint(jwt)
        claims = decode_jwt_claims(jwt)
        test = test_token_against_fv(jwt, game_pk)
        results.append({
            **fp,
            "claims": safe_claims(claims),
            "claimKeys": list(claims.keys()) if claims else [],
            "sources": sources[:3],
            "fieldvisionHls": test,
        })
        marker = "  " if test["status"] != 200 else "★ "
        scope = (claims or {}).get("scope") or (claims or {}).get("scopes") or "-"
        if isinstance(scope, list):
            scope = ",".join(scope)
        print(f"  {marker}status={test['status']:4}  hash={fp['hash']}  "
              f"sub={(claims or {}).get('sub', '-')[:20]:20}  "
              f"scope={str(scope)[:40]:40}")
        if test["status"] == 200:
            successful_jwt = jwt

    # Save anonymized results
    out_path = out_dir / f"auth_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "gamePk": game_pk,
        "jwtCount": len(found_jwts),
        "results": results,
    }, indent=2))
    print(f"\nResults saved (no full tokens): {out_path}")

    # If we got a 200, try fetching the manifest + metadata + labels + first segment
    if successful_jwt:
        capture_dir = Path("samples") / f"binary_capture_{game_pk}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        capture_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n★ Live token confirmed. Capturing schema files to {capture_dir}/")

        base = f"https://fieldvision-hls.mlbinfra.com/mannequin/{game_pk}/1.6.2"
        for fname in ("manifest.json", "metadata.json", "labels.json", "0.bin"):
            try:
                req = urllib.request.Request(
                    f"{base}/{fname}",
                    headers={
                        "Authorization": f"Bearer {successful_jwt}",
                        "x-mannequin-client": "gameday",
                        "Origin": "https://www.mlb.com",
                        "Referer": "https://www.mlb.com/",
                    },
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                    (capture_dir / fname).write_bytes(data)
                    print(f"  saved {fname}  ({len(data):,} bytes, status {resp.status})")
            except urllib.error.HTTPError as e:
                print(f"  {fname}: HTTP {e.code}")
            except Exception as e:
                print(f"  {fname}: {e}")
        del successful_jwt
        print("\n  Token discarded from memory.")
        print(f"  Inspect captured files at {capture_dir}/")
    else:
        print(f"\nNo JWT returned 200. The legacy binary endpoint may be closed to web users post-migration.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=823141)
    parser.add_argument("--no-headed", action="store_true",
                        help="Run headless (only useful if profile already has logged-in cookies)")
    parser.add_argument(
        "--profile",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / ".mlb_profile"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "diagnostics" / "raw"),
    )
    args = parser.parse_args()
    asyncio.run(probe(args.game, headed=not args.no_headed,
                      profile_dir=Path(args.profile), out_dir=Path(args.out)))


if __name__ == "__main__":
    main()
