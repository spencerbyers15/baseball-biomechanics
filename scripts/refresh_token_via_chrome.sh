#!/bin/bash
# Auto-refresh the FieldVision Okta JWT by driving Chrome via AppleScript.
#
# Requires (one-time setup):
#   1. Chrome menu bar → View → Developer → Allow JavaScript from Apple Events
#   2. You're signed in to mlb.com in Chrome (cookies persisted; usually lasts weeks)
#
# Behavior:
#   - Opens or reuses an mlb.com tab in Chrome
#   - Waits for Okta's silent-refresh to populate localStorage
#   - Extracts the api://mlb_default JWT
#   - Validates aud + exp via Python
#   - Writes to .fv_token.txt
#   - Removes state/token_expired.flag if present
#   - Closes the tab if we opened it
#
# Exit codes:
#   0 = success
#   1 = chrome not running and couldn't launch
#   2 = JS returned no token (probably logged out of mlb.com)
#   3 = JWT validation failed

set -e

REPO_ROOT="${REPO_ROOT:-$HOME/fieldvision}"
TOKEN_FILE="$REPO_ROOT/.fv_token.txt"
EXPIRED_FLAG="$REPO_ROOT/state/token_expired.flag"
LOG="$REPO_ROOT/scheduler.log"
PY="${PY:-/Users/spencerbyers/anaconda3/bin/python3}"

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
log() { echo "$(ts) [refresh-token] $*" | tee -a "$LOG"; }

# Step 0: skip-if-fresh. If the existing token still has > MIN_HOURS_LEFT,
# do nothing — avoids Chrome tab reloads when token doesn't need refresh.
# Override with --force to refresh anyway.
MIN_HOURS_LEFT="${MIN_HOURS_LEFT:-6}"
if [ -f "$TOKEN_FILE" ] && [ "${1:-}" != "--force" ]; then
    REMAINING=$("$PY" -c "
import base64, json, time, sys
try:
    tok = open('$TOKEN_FILE').read().strip()
    if tok.count('.') != 2: print('0'); sys.exit(0)
    p = tok.split('.')[1] + '=' * (-len(tok.split('.')[1]) % 4)
    c = json.loads(base64.urlsafe_b64decode(p))
    if c.get('aud') != 'api://mlb_default': print('0'); sys.exit(0)
    print(f'{(c[\"exp\"] - time.time()) / 3600:.2f}')
except Exception:
    print('0')
" 2>/dev/null)
    # bash float compare via awk
    if awk "BEGIN { exit !($REMAINING > $MIN_HOURS_LEFT) }"; then
        log "skipping — current token still has ${REMAINING}h (> ${MIN_HOURS_LEFT}h threshold)"
        exit 0
    fi
fi

# Step 1: ensure Chrome is running
if ! pgrep -x "Google Chrome" >/dev/null; then
    log "Chrome not running, launching it"
    open -a "Google Chrome" || { log "ERROR: failed to open Chrome"; exit 1; }
    sleep 5
fi

# Step 2: find or create an mlb.com tab. Use a temp file for the AppleScript
# to avoid bash's heredoc-in-$() parenthesis-matching gotcha.
TAB_SCRIPT=$(mktemp /tmp/fv-tab.XXXXXX.scpt)
cat > "$TAB_SCRIPT" <<'OSA'
tell application "Google Chrome"
    set foundTab to missing value
    repeat with w in windows
        repeat with t in tabs of w
            if URL of t starts with "https://www.mlb.com" then
                set foundTab to t
                exit repeat
            end if
        end repeat
        if foundTab is not missing value then exit repeat
    end repeat
    if foundTab is missing value then
        if (count of windows) is 0 then
            make new window
        end if
        make new tab at end of tabs of front window with properties {URL:"https://www.mlb.com"}
        return "CREATED"
    else
        set URL of foundTab to "https://www.mlb.com"
        return "EXISTING"
    end if
end tell
OSA
TAB_STATE=$(osascript "$TAB_SCRIPT" 2>&1)
rm -f "$TAB_SCRIPT"
log "tab state: $TAB_STATE"

# Step 3: wait for Okta silent-refresh to land in localStorage
sleep 8

# Step 4: extract the JWT from the mlb.com tab.
EXTRACT_SCRIPT=$(mktemp /tmp/fv-extract.XXXXXX.scpt)
cat > "$EXTRACT_SCRIPT" <<'OSA'
tell application "Google Chrome"
    repeat with w in windows
        repeat with t in tabs of w
            if URL of t starts with "https://www.mlb.com" then
                tell t
                    return execute javascript "(() => { const raw = localStorage.getItem('okta-token-storage'); if (!raw) return ''; const RX = /eyJ[A-Za-z0-9_-]{8,}\\.eyJ[A-Za-z0-9_-]{8,}\\.[A-Za-z0-9_-]{8,}/g; const matches = raw.match(RX) || []; for (const j of new Set(matches)) { try { const c = JSON.parse(atob(j.split('.')[1].replace(/-/g,'+').replace(/_/g,'/'))); if (c.aud === 'api://mlb_default') return j; } catch(e) {} } return ''; })()"
                end tell
            end if
        end repeat
    end repeat
    return ""
end tell
OSA
TOKEN=$(osascript "$EXTRACT_SCRIPT" 2>&1)
rm -f "$EXTRACT_SCRIPT"

if [ -z "$TOKEN" ] || [ "${TOKEN:0:3}" != "eyJ" ]; then
    log "ERROR: no api://mlb_default JWT found. You may be logged out of mlb.com — sign in manually once."
    exit 2
fi

# Step 5: validate the JWT
HOURS_LEFT=$("$PY" - "$TOKEN" <<'PY'
import sys, base64, json, time
tok = sys.argv[1].strip()
if tok.count('.') != 2:
    print("not a JWT", file=sys.stderr); sys.exit(3)
p = tok.split('.')[1] + '=' * (-len(tok.split('.')[1]) % 4)
try:
    c = json.loads(base64.urlsafe_b64decode(p))
except Exception as e:
    print(f"payload decode failed: {e}", file=sys.stderr); sys.exit(3)
if c.get('aud') != 'api://mlb_default':
    print(f"wrong aud: {c.get('aud')}", file=sys.stderr); sys.exit(3)
hours_left = (c['exp'] - time.time()) / 3600
if hours_left < 0.5:
    print(f"already expired or about to ({hours_left:.2f}h)", file=sys.stderr); sys.exit(3)
print(f"{hours_left:.2f}")
PY
) || { log "ERROR: JWT validation failed"; exit 3; }

# Step 6: write the token + clear expired flag
echo "$TOKEN" > "$TOKEN_FILE"
rm -f "$EXPIRED_FLAG"
log "fresh token written ($(wc -c < "$TOKEN_FILE") bytes, ${HOURS_LEFT}h remaining)"

# Step 7: if we created the tab, close it
if [ "$TAB_STATE" = "CREATED" ]; then
    CLOSE_SCRIPT=$(mktemp /tmp/fv-close.XXXXXX.scpt)
    cat > "$CLOSE_SCRIPT" <<'OSA'
tell application "Google Chrome"
    repeat with w in windows
        set tabsToClose to {}
        repeat with t in tabs of w
            if URL of t starts with "https://www.mlb.com" then
                set tabsToClose to tabsToClose & {t}
            end if
        end repeat
        repeat with t in tabsToClose
            close t
        end repeat
    end repeat
end tell
OSA
    osascript "$CLOSE_SCRIPT" >/dev/null 2>&1 || true
    rm -f "$CLOSE_SCRIPT"
    log "closed the mlb.com tab we created"
fi

exit 0
