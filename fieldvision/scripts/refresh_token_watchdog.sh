#!/bin/bash
# Token watchdog — runs every 60s via launchd.
#
# Decides whether to refresh the MLB JWT and push it to Nellie.
# Triggers refresh if EITHER:
#   - Nellie's state/token_expired.flag is present (scraper hit 401/403)
#   - Local .fv_token.txt is missing or > 4h old (proactive safety net)
#
# After a successful refresh, the Nellie flag is cleared.
#
# Exit codes: 0 = no action taken (token still fresh, no flag set),
#             0 = refreshed successfully, 1 = refresh attempted but failed.

set -uo pipefail

# REPO_ROOT must be a TCC-safe path (outside ~/Documents, ~/Desktop, ~/Downloads)
# because launchd-spawned processes can't write into Documents on modern macOS.
# The refresh_token_via_chrome.sh and our own log writes use this dir.
REPO_ROOT="${REPO_ROOT:-/Users/spencerbyers/fieldvision}"
REPO_GIT="${REPO_GIT:-/Users/spencerbyers/Documents/GitHub/baseball-biomechanics/fieldvision}"
NELLIE_HOST="${NELLIE_HOST:-nellie}"
NELLIE_STATE_DIR="${NELLIE_STATE_DIR:-/media/scratch/spencer/data/fieldvision/state}"
NELLIE_TOKEN_PATH="${NELLIE_TOKEN_PATH:-/media/scratch/spencer/github/baseball-biomechanics/fieldvision/.fv_token.txt}"
NELLIE_FLAG="$NELLIE_STATE_DIR/token_expired.flag"
LOCAL_TOKEN="$REPO_ROOT/.fv_token.txt"
LOG="$REPO_ROOT/state/watchdog.log"
PROACTIVE_MAX_AGE_HOURS=4

mkdir -p "$REPO_ROOT/state"
# Mirror the fresh token back to the repo path as well so the repo-resident
# scripts also see it. This is a best-effort write — if TCC blocks the
# Documents write (which it can do for launchd-spawned processes) we just
# skip it; Nellie still gets the fresh token via scp.

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# Trim log file if it gets huge (>5MB)
if [ -f "$LOG" ] && [ "$(stat -f %z "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ]; then
  tail -n 1000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

SSH_OPTS=(-o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

nellie_flag_set() {
  ssh "${SSH_OPTS[@]}" "$NELLIE_HOST" "test -f '$NELLIE_FLAG'" 2>/dev/null
}

local_token_age_hours() {
  if [ ! -f "$LOCAL_TOKEN" ]; then echo 999; return; fi
  local mtime now age
  mtime=$(stat -f '%m' "$LOCAL_TOKEN" 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$(( (now - mtime) / 3600 ))
  echo "$age"
}

REASON=""
if nellie_flag_set; then
  REASON="nellie flag set"
fi
age=$(local_token_age_hours)
if [ -z "$REASON" ] && [ "$age" -ge "$PROACTIVE_MAX_AGE_HOURS" ]; then
  REASON="local token age ${age}h >= ${PROACTIVE_MAX_AGE_HOURS}h"
fi

if [ -z "$REASON" ]; then
  # Nothing to do — token is fresh and Nellie isn't asking
  exit 0
fi

log "trigger: $REASON"

# Run the