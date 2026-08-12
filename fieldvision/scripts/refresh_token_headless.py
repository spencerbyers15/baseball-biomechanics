#!/usr/bin/env python3
"""Headless MLB Okta JWT refresh — no browser, no Mac, no human.

Replaces the Mac-dependent `refresh_token_via_chrome.sh` (AppleScript into a
logged-in Chrome tab), which caused three near-death token outages in three
days: it only fires while the Mac is awake, and it fails silently when the
pinned mlb.com tab is gone.

## How it works

mlb.com's SPA keeps its `api://mlb_default` access token alive by running
Okta's PKCE silent-auth in a hidden iframe: an authorize call with
`prompt=none`, authenticated by the long-lived Okta **session cookie** rather
than by any credential the page holds. This script replays exactly that
exchange from the command line:

    1. mint a PKCE verifier/challenge pair
    2. GET  /v1/authorize?prompt=none&...  with the session cookie -> auth code
    3. POST /v1/token  (code + verifier)                          -> access token

Two properties make this durable, both verified 2026-08-12:

* **`idx` is the only cookie required.** Okta's session cookie alone produces
  a code; `DT`/`JSESSIONID`/`proximity_*` are not needed.
* **Okta rotates `idx` on every authorize.** We persist the rotated value, so
  each refresh renews the session. As long as this runs more often than the
  session's idle timeout, the cookie never goes stale and the Mac is never
  needed again after the one-time seed (`seed_okta_cookie.py`).

## The 451 gotcha

`ids.mlb.com/v1/authorize` sits behind edge bot-detection that blocks on TLS
fingerprint: plain `requests` and `curl` get **HTTP 451 with an empty body**
(even for a bare, parameterless request) while a real browser gets a normal
302. Header spoofing does not help. Hence `curl_cffi` with
`impersonate="chrome"`, which reproduces Chrome's TLS/HTTP2 fingerprint.
A 451 from this script means the impersonation stopped being convincing --
try a newer curl_cffi / a newer `--impersonate` target.

## Usage (Nellie, from cron)

    FV_TOKEN_FILE=/home/spencer/.fv_token.txt \
    FV_STATE_DIR=/media/scratch/spencer/fieldvision/state \
      ~/venvs/fieldvision/bin/python scripts/refresh_token_headless.py

Skips silently while the current token has more than --min-hours left, so it
is cheap to run every few minutes.

## Exit codes

    0  refreshed, or skipped because the token is still fresh
    2  Okta session is dead -> re-seed the cookie from the Mac (see runbook)
    3  got a token but it failed validation (wrong aud / already expiring)
    4  edge blocked us (451) or the network failed -> retry; if persistent,
       the TLS impersonation needs updating
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.parse as up
from datetime import datetime
from pathlib import Path

try:
    from curl_cffi import requests as cr
except ImportError:  # pragma: no cover - environment problem, not logic
    print("curl_cffi is required (pip install curl_cffi) — plain requests "
          "gets HTTP 451 from ids.mlb.com", file=sys.stderr)
    raise SystemExit(4)

# Okta authorization server + SPA client, read off mlb.com's own silent-refresh
# request (performance.getEntriesByType('resource') in a logged-in tab).
ISSUER = os.environ.get("FV_OKTA_ISSUER",
                        "https://ids.mlb.com/oauth2/aus1m088yK07noBfh356")
CLIENT_ID = os.environ.get("FV_OKTA_CLIENT_ID", "0oap7wa857jcvPlZ5355")
REDIRECT_URI = os.environ.get("FV_OKTA_REDIRECT_URI", "https://www.mlb.com/login")
SCOPE = "openid email"
EXPECTED_AUD = "api://mlb_default"
IMPERSONATE = os.environ.get("FV_IMPERSONATE", "chrome")

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_FILE = Path(os.environ.get("FV_TOKEN_FILE", REPO_ROOT / ".fv_token.txt"))
STATE_DIR = Path(os.environ.get("FV_STATE_DIR", REPO_ROOT / "state"))
COOKIE_FILE = Path(os.environ.get("FV_OKTA_COOKIE_FILE",
                                  Path.home() / ".fv_okta_cookies.json"))

EXIT_OK, EXIT_SESSION_DEAD, EXIT_BAD_TOKEN, EXIT_BLOCKED = 0, 2, 3, 4


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [token] {msg}",
          flush=True)


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def jwt_claims(token: str) -> dict:
    """Decode a JWT payload without verifying the signature (Okta signed it;
    we only need the claims to check aud/exp before trusting the token)."""
    if token.count(".") != 2:
        raise ValueError("not a JWT")
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def hours_left(token: str) -> float:
    """Hours until this token expires; negative/0 if unusable."""
    try:
        claims = jwt_claims(token)
    except Exception:
        return 0.0
    if claims.get("aud") != EXPECTED_AUD:
        return 0.0
    return (claims["exp"] - time.time()) / 3600


def write_private(path: Path, text: str) -> None:
    """Write 0600, atomically — the autopilot reads the token file
    concurrently and must never see a half-written value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def load_cookies() -> dict:
    if not COOKIE_FILE.exists():
        log(f"ERROR: no cookie jar at {COOKIE_FILE} — seed it from the Mac "
            f"(scripts/seed_okta_cookie.py --push nellie)")
        raise SystemExit(EXIT_SESSION_DEAD)
    jar = json.loads(COOKIE_FILE.read_text())
    if not jar.get("idx"):
        log(f"ERROR: cookie jar {COOKIE_FILE} has no 'idx' session cookie")
        raise SystemExit(EXIT_SESSION_DEAD)
    return jar


def save_cookies(jar: dict) -> None:
    write_private(COOKIE_FILE, json.dumps(jar))


def silent_refresh(jar: dict) -> tuple[str, dict]:
    """Run the prompt=none PKCE exchange. Returns (access_token, updated_jar).

    Raises SystemExit with the appropriate code on failure.
    """
    verifier = b64url(secrets.token_bytes(32))
    challenge = b64url(hashlib.sha256(verifier.encode()).digest())
    params = {
        "client_id": CLIENT_ID,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "nonce": b64url(secrets.token_bytes(32)),
        "prompt": "none",
        "redirect_uri": REDIRECT_URI,
        "response_mode": "query",  # 302 with ?code=, simpler than okta_post_message
        "response_type": "code",
        "state": b64url(secrets.token_bytes(32)),
        "scope": SCOPE,
    }

    session = cr.Session(impersonate=IMPERSONATE)
    for name, value in jar.items():
        session.cookies.set(name, value, domain="ids.mlb.com")

    try:
        resp = session.get(f"{ISSUER}/v1/authorize?" + up.urlencode(params),
                           allow_redirects=False, timeout=30,
                           headers={"Referer": "https://www.mlb.com/"})
    except Exception as exc:
        log(f"ERROR: authorize request failed: {exc}")
        raise SystemExit(EXIT_BLOCKED)

    if resp.status_code == 451:
        log("ERROR: HTTP 451 from authorize — edge bot-detection rejected our "
            "TLS fingerprint. Update curl_cffi or FV_IMPERSONATE.")
        raise SystemExit(EXIT_BLOCKED)

    location = resp.headers.get("Location", "")
    query = up.parse_qs(up.urlparse(location).query)

    if "code" not in query:
        error = (query.get("error") or ["?"])[0]
        detail = (query.get("error_description") or [""])[0]
        if error in ("login_required", "interaction_required",
                     "consent_required", "?"):
            reason = detail or f"no code in a {resp.status_code} response"
            log(f"ERROR: Okta session is dead ({error}: {reason}). Re-seed "
                f"the cookie from the Mac — see docs/headless_token_refresh.md")
            raise SystemExit(EXIT_SESSION_DEAD)
        log(f"ERROR: authorize returned {resp.status_code} {error}: {detail}")
        raise SystemExit(EXIT_BLOCKED)

    try:
        token_resp = session.post(
            f"{ISSUER}/v1/token",
            data={"grant_type": "authorization_code", "client_id": CLIENT_ID,
                  "redirect_uri": REDIRECT_URI, "code": query["code"][0],
                  "code_verifier": verifier},
            timeout=30)
    except Exception as exc:
        log(f"ERROR: token exchange failed: {exc}")
        raise SystemExit(EXIT_BLOCKED)

    if token_resp.status_code != 200:
        log(f"ERROR: token endpoint returned {token_resp.status_code}: "
            f"{token_resp.text[:200]}")
        raise SystemExit(EXIT_BAD_TOKEN)

    access_token = token_resp.json().get("access_token", "")
    if not access_token.startswith("eyJ"):
        log("ERROR: token response contained no access_token")
        raise SystemExit(EXIT_BAD_TOKEN)

    # Okta re-issues `idx` on every authorize; persisting it is what keeps the
    # session alive indefinitely without the Mac.
    updated = dict(jar)
    for cookie in session.cookies.jar:
        if cookie.name in ("idx", "DT") and cookie.value:
            updated[cookie.name] = cookie.value
    return access_token, updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="refresh even if the current token is still fresh")
    parser.add_argument("--min-hours", type=float,
                        default=float(os.environ.get("FV_TOKEN_MIN_HOURS", 6)),
                        help="refresh when fewer than this many hours remain "
                             "(default 6)")
    parser.add_argument("--validate-only", action="store_true",
                        help="report the current token's remaining life, "
                             "change nothing")
    args = parser.parse_args()

    current = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else ""
    remaining = hours_left(current) if current else 0.0

    if args.validate_only:
        log(f"current token: {remaining:.2f}h remaining ({TOKEN_FILE})")
        return EXIT_OK if remaining > 0 else EXIT_BAD_TOKEN

    expired_flag = STATE_DIR / "token_expired.flag"
    # A worker that hit 401/403 raises this flag; honor it even when the JWT's
    # own exp still looks healthy (MLB has revoked tokens early before).
    flagged = expired_flag.exists()

    if not args.force and not flagged and remaining > args.min_hours:
        log(f"skipping — {remaining:.2f}h left (> {args.min_hours}h threshold)")
        return EXIT_OK
    if flagged:
        log("token_expired.flag present — a worker saw 401/403, refreshing")

    jar = load_cookies()
    token, updated_jar = silent_refresh(jar)

    fresh_hours = hours_left(token)
    if fresh_hours < 0.5:
        log(f"ERROR: minted token is unusable ({fresh_hours:.2f}h, "
            f"aud={jwt_claims(token).get('aud')})")
        return EXIT_BAD_TOKEN

    write_private(TOKEN_FILE, token + "\n")
    if updated_jar != jar:
        save_cookies(updated_jar)
        log("Okta session cookie rotated and persisted")

    try:
        expired_flag.unlink()
        log("cleared token_expired.flag")
    except OSError:
        pass

    log(f"fresh token written ({len(token)} bytes, {fresh_hours:.2f}h remaining)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
