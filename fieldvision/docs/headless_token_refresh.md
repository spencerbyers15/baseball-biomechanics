# Headless MLB token refresh

The FieldVision capture pipeline needs a valid `api://mlb_default` Okta JWT at
all times; it expires every 24h. Until 2026-08-12 that token could only be
minted on Spencer's Mac, by AppleScript-ing a logged-in Chrome tab
(`refresh_token_via_chrome.sh`). That path caused **three near-death outages in
three days**: it fires only while the Mac is awake, and it fails silently when
the pinned mlb.com tab disappears — one failure even happened *with* a tab
present and was never diagnosed.

Nellie now refreshes the token entirely on its own.

## How it works

mlb.com keeps its access token alive with Okta's PKCE **silent auth**: a
`prompt=none` authorize call from a hidden iframe, authenticated by the
long-lived Okta *session cookie* rather than by any stored credential.
`scripts/refresh_token_headless.py` replays exactly that exchange:

```
mint PKCE verifier ─► GET /v1/authorize?prompt=none  (+ session cookie) ─► auth code
                   ─► POST /v1/token (code + verifier)                  ─► access token
```

Two measured properties (verified 2026-08-12) make this self-sustaining:

| Property | Consequence |
|---|---|
| `idx` alone authenticates the authorize call | nothing else has to be captured or kept in sync |
| Okta **rotates `idx` on every authorize**, and we persist the rotation | each refresh renews the session, so the cookie never goes stale and the Mac is never needed again |

The Okta session had already been alive ~103 days at build time (`auth_time`
in the JWT), so the underlying session lifetime is generous; rotating it
every few hours keeps it indefinitely warm.

## The HTTP 451 gotcha

`ids.mlb.com/v1/authorize` sits behind edge bot-detection that keys on **TLS
fingerprint**. Plain `requests` and `curl` get `HTTP 451` with an empty body —
even for a bare parameterless request — while a real browser gets a normal
302. Spoofing headers (UA, `Sec-Fetch-*`, `sec-ch-ua`) does *not* help; only
the TLS/HTTP2 fingerprint matters. Hence `curl_cffi` with
`impersonate="chrome"`.

**If the refresher starts exiting 4 with a 451**, the impersonation has aged
out: upgrade `curl_cffi`, or set `FV_IMPERSONATE` to a newer target
(`chrome124`, `chrome131`, …). This is the most likely long-term breakage.

## Deployment on Nellie

| Piece | Location |
|---|---|
| Canonical script | this repo, `fieldvision/scripts/refresh_token_headless.py` |
| Deployed copy | `${NELLIE_HOME}/refresh_token_headless.py` (**local disk, not the NAS** — token refresh must survive an unmounted NAS) |
| Session cookie jar | `${NELLIE_HOME}/.fv_okta_cookies.json`, mode 0600 |
| Token output | `${NELLIE_HOME}/.fv_token.txt`, mode 0600, written atomically |
| Log | `${NELLIE_HOME}/logs/token_refresh.log` |
| Schedule | cron `*/5 * * * *` (skip-if-fresh makes this nearly free) |

Deployed copy and repo copy must be kept in sync — redeploy with:

```bash
scp fieldvision/scripts/refresh_token_headless.py nellie:${NELLIE_HOME}/refresh_token_headless.py
```

Cron entry:

```cron
*/5 * * * * FV_TOKEN_FILE=${NELLIE_HOME}/.fv_token.txt FV_STATE_DIR=/media/scratch/${NELLIE_USER}/fieldvision/state ${NELLIE_HOME}/venvs/fieldvision/bin/python ${NELLIE_HOME}/refresh_token_headless.py >> ${NELLIE_HOME}/logs/token_refresh.log 2>&1
```

Running every 5 minutes is deliberate: the script no-ops while the token has
more than 6h left, but it also watches for `state/token_expired.flag` (raised
by any worker that sees a 401/403), so real recovery time is ≤5 minutes —
matching the old 60s Mac watchdog closely enough without a browser.

## One-time seed (the only step needing the Mac)

```bash
python3 fieldvision/scripts/seed_okta_cookie.py --push nellie
```

Reads the httpOnly `idx` cookie out of Chrome's encrypted cookie store with
`browser_cookie3` (injected JS *cannot* read httpOnly cookies — same reason
the ESPN refresher in `fantasy-baseball-hub` reads the store, not the DOM),
and pipes it over ssh via stdin so the value never lands in argv, shell
history, or the process table. Requires being signed in to mlb.com in Chrome;
the browser does not need to be open or focused when it runs.

Re-run this **only** when the refresher exits 2.

## Exit codes and what to do

| Code | Meaning | Action |
|---|---|---|
| 0 | refreshed, or skipped (still fresh) | none |
| 2 | Okta session dead | sign in to mlb.com in Chrome on the Mac, re-run the seeder |
| 3 | minted token failed validation | check the Okta client config didn't change |
| 4 | 451 / network failure | usually transient; if persistent, update the TLS impersonation (see above) |

## Monitoring

The dashboard (`http://${NELLIE_HOST}:8377`) Token tile shows hours remaining —
it should sawtooth between 24h and 18h and never approach zero. To check by
hand:

```bash
ssh nellie 'FV_TOKEN_FILE=${NELLIE_HOME}/.fv_token.txt ~/venvs/fieldvision/bin/python ${NELLIE_HOME}/refresh_token_headless.py --validate-only'
```

## Status of the old Mac path

`refresh_token_via_chrome.sh` and `refresh_token_watchdog.sh` are retired; the
launchd job `com.spencerbyers.fvtoken-watchdog` is unloaded. They are kept in
the repo as a manual fallback — if the headless path ever breaks completely,
the Chrome route still mints a token that can be scp'd to Nellie. Do not run
both on a schedule: they write the same file, and the Mac path can clobber a
newer token with an older one.
