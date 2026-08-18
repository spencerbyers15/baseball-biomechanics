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
REPO_ROOT="${REPO_ROOT:-$HOME/fieldvision}"
REPO_GIT="${REPO_GIT:-$HOME/Documents/GitHub/baseball-biomechanics/fieldvision}"
NELLIE_HOST="${NELLIE_HOST:-nellie}"   # ssh alias; real host in .env
NELLIE_USER="${NELLIE_USER:-$(id -un)}"
NELLIE_STATE_DIR="${NELLIE_STATE_DIR:-/media/scratch/$NELLIE_USER/fieldvision/state}"
NELLIE_TOKEN_PATH="${NELLIE_TOKEN_PATH:-/home/$NELLIE_USER/.fv_token.txt}"  # private home, NOT the world-readable NAS
NELLIE_FLAG="$NELLIE_STATE_DIR/token_expired.flag"
LOCAL_TOKEN="$REPO_ROOT/.fv_token.txt"
PROACTIVE_MAX_AGE_HOURS=18

# Log to stdout — launchd captures via StandardOutPath, which is the only
# write-permission we reliably have under launchd's sandbox. Trying to
# `>> $REPO_ROOT/state/watchdog.log` directly raises "Operation not
# permitted" even though the path is outside ~/Documents.
log() { echo "[$(date '+%F %T')] $*"; }

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

# Run the refresh (writes the fresh token to $LOCAL_TOKEN at $REPO_ROOT)
if REPO_ROOT="$REPO_ROOT" bash "$REPO_ROOT/scripts/refresh_token_via_chrome.sh" --force 2>&1; then
  # Best-effort mirror to the repo path (may fail under TCC; that's fine)
  cp "$LOCAL_TOKEN" "$REPO_GIT/.fv_token.txt" 2>/dev/null || true
  # Push to Nellie + clear the flag
  if scp "${SSH_OPTS[@]}" "$LOCAL_TOKEN" "$NELLIE_HOST:$NELLIE_TOKEN_PATH" 2>&1; then
    ssh "${SSH_OPTS[@]}" "$NELLIE_HOST" "rm -f '$NELLIE_FLAG'" 2>&1
    log "refresh OK + synced to Nellie + flag cleared"
  else
    log "refresh OK but scp to Nellie FAILED — flag will be retried next tick"
    exit 1
  fi
else
  log "refresh FAILED — flag remains, will retry next tick"
  exit 1
fi
