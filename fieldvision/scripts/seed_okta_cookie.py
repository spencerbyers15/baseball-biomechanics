#!/usr/bin/env python3
"""Seed the Okta session cookie from a logged-in Chrome into Nellie.

This is the ONLY step in the token pipeline that needs Spencer's Mac, and it
runs once — after that `refresh_token_headless.py` on Nellie renews the
session itself on every refresh (Okta rotates `idx` each authorize, and we
persist the rotation). Re-run this only if the headless refresher exits 2
("Okta session is dead"), which means the server-side session finally lapsed
and mlb.com needs a real sign-in again.

Why read Chrome's cookie STORE instead of the page: Okta's `idx` session
cookie is **httpOnly**, so injected JavaScript / `document.cookie` cannot see
it. `browser_cookie3` decrypts Chrome's cookie DB (via the macOS Keychain) and
returns httpOnly cookies. Same trick as the ESPN refresher in
fantasy-baseball-hub — and unlike the old AppleScript path, nothing has to be
visible, focused, or even open in the browser.

Usage:
    python3 scripts/seed_okta_cookie.py --push nellie     # read + ship (usual)
    python3 scripts/seed_okta_cookie.py --check           # is Chrome logged in?
    python3 scripts/seed_okta_cookie.py --out cookies.json

The cookie value is never printed and never written anywhere but the
destination file (0600, in Nellie's private home — never the NAS, which is
world-readable).

Exit codes:
    0  seeded (or --check found a session cookie)
    2  Chrome has no ids.mlb.com session cookie — sign in to mlb.com first
    5  transport failure pushing to the remote host
"""
from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys

# Remote home is expanded by the remote shell, so `~` keeps this
# host-agnostic. Override with FV_REMOTE_COOKIE_PATH if needed.
REMOTE_PATH = os.environ.get("FV_REMOTE_COOKIE_PATH",
                             "~/.fv_okta_cookies.json")
# `idx` is Okta's session cookie and is sufficient on its own (verified
# 2026-08-12). `DT` is the device token — harmless to carry along, and Okta
# sometimes wants it for device-binding checks.
WANTED = ("idx", "DT")


def read_chrome_cookies() -> dict:
    try:
        import browser_cookie3
    except ImportError:
        print("browser_cookie3 is required: pip install browser_cookie3",
              file=sys.stderr)
        raise SystemExit(2)

    jar = browser_cookie3.chrome(domain_name="mlb.com")
    found = {c.name: c.value for c in jar
             if "ids.mlb.com" in c.domain and c.name in WANTED and c.value}
    if "idx" not in found:
        print("No ids.mlb.com 'idx' session cookie in Chrome — sign in to "
              "mlb.com in Chrome once, then re-run.", file=sys.stderr)
        raise SystemExit(2)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--push", metavar="HOST",
                        help="ssh host to write the cookie jar on "
                             f"(at {REMOTE_PATH}, mode 0600)")
    parser.add_argument("--out", metavar="PATH",
                        help="write the cookie jar to a local file instead")
    parser.add_argument("--check", action="store_true",
                        help="only report whether Chrome has a session cookie")
    args = parser.parse_args()

    cookies = read_chrome_cookies()
    print(f"found Okta cookies in Chrome: {sorted(cookies)} "
          f"(idx is {len(cookies['idx'])} chars)")

    if args.check:
        return 0

    payload = json.dumps(cookies)

    if args.out:
        import os
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        print(f"wrote {args.out} (0600)")
        return 0

    if not args.push:
        print("nothing to do — pass --push HOST or --out PATH", file=sys.stderr)
        return 0

    # Pipe over stdin so the cookie never appears in argv / shell history /
    # the process table on either machine.
    remote = (f"umask 077; cat > {REMOTE_PATH}; chmod 600 {REMOTE_PATH}")
    proc = subprocess.run(["ssh", args.push, remote], input=payload,
                          text=True, capture_output=True)
    if proc.returncode != 0:
        print(f"ssh push failed: {proc.stderr.strip()}", file=sys.stderr)
        return 5
    print(f"seeded {args.push}:{REMOTE_PATH} (0600)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
