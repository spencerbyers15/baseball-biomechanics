"""One-time login flow that establishes a persistent Playwright profile so
the daemon can autonomously read fresh JWTs from your MLB.com session.

Usage (run once):
  python scripts/setup_login.py

Flow:
  1. Opens a headed Chromium with user_data_dir=~/fieldvision/.mlb_profile/
  2. Forces the window to the foreground via AppleScript (works around the
     macOS focus issue we hit earlier).
  3. Navigates to mlb.com/login.
  4. Waits up to 5 minutes for you to log in.
  5. Verifies the resulting JWT works against fieldvision-hls.mlbinfra.com.
  6. Closes. Cookies + Okta refresh tokens persist on disk.

After this, fv_daemon.py can launch the same profile headlessly and grab
fresh access tokens without your involvement until the cookies expire
(typically weeks).
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

PROFILE_DIR = Path.home() / "Documents" / "GitHub" / "fieldvision" / ".mlb_profile"
LOGIN_TIMEOUT_S = 300


def force_focus_chromium():
    """macOS only — bring Chromium frontmost so the user can type."""
    try:
        subprocess.run(
            ["osascript", "-e",
             'tell application "Chromium" to activate'],
            check=False, capture_output=True, timeout=3,
        )
    except Exception:
        pass


def find_jwt_in_storage(storage_text: str) -> str | None:
    rx = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
    import base64
    for j in set(rx.findall(storage_text)):
        try:
            payload = j.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            if claims.get("aud") == "api://mlb_default":
                return j
        except Exception:
            continue
    return None


def verify_token(jwt: str) -> tuple[bool, str]:
    """Make a real call to fieldvision-hls. Returns (ok, info)."""
    url = "https://fieldvision-hls.mlbinfra.com/mannequin/823141/1.6.2/manifest.json"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {jwt}",
        "x-mannequin-client": "gameday",
        "Origin": "https://www.mlb.com",
        "Referer": "https://www.mlb.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


async def main():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Persistent profile: {PROFILE_DIR}")

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled",
                  "--disable-dev-shm-usage",
                  "--start-maximized"],
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://www.mlb.com/login", wait_until="domcontentloaded", timeout=60000)

        # Bring window to front so user can interact
        await asyncio.sleep(1)
        force_focus_chromium()

        print()
        print("=" * 70)
        print("  Browser is open at https://www.mlb.com/login")
        print(f"  Please log in. You have {LOGIN_TIMEOUT_S}s.")
        print("  This window will close automatically once login is detected.")
        print("=" * 70)
        print()

        # Poll for login by checking storage for a valid JWT
        token = None
        elapsed = 0
        check_interval = 3
        while elapsed < LOGIN_TIMEOUT_S:
            await asyncio.sleep(check_interval)
            elapsed += check_interval
            try:
                # Read all localStorage entries
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
                token = find_jwt_in_storage(storage)
                if token:
                    print(f"  ✓ Detected api://mlb_default token (length {len(token)})")
                    break
            except Exception as e:
                # Page might be navigating during login; keep polling
                pass
            if elapsed % 30 == 0:
                print(f"  (still waiting... {elapsed}s elapsed)")

        if not token:
            print("  ✗ Timed out without finding a valid token.")
            print("    Make sure you're logged in (top-right shows your username).")
            await ctx.close()
            sys.exit(1)

        # Verify the token actually works against fieldvision-hls
        ok, info = verify_token(token)
        if ok:
            print(f"  ✓ Token verified against fieldvision-hls.mlbinfra.com ({info})")
            print()
            print(f"Login successful. Profile saved to:")
            print(f"  {PROFILE_DIR}")
            print()
            print("You can now run the daemon: scripts/fv_daemon.py")
        else:
            print(f"  ✗ Token did NOT verify ({info}).")
            print("    The session is logged in to MLB.com but doesn't have")
            print("    fieldvision scope. Check that your account is in good standing.")

        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
