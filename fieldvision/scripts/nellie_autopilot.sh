#!/bin/bash
# Launch (or re-launch) the FieldVision autopilot inside tmux on Nellie.
# Idempotent: exits quietly if the session is already up. Invoked from
# cron every 10 minutes as a keepalive (see launchd/nellie_autopilot_cron.txt).
#
# The NAS is mounted noexec, so cron must invoke this as `bash <path>`.

SESSION=fv-autopilot

# Host-specific values live in fieldvision/.env (gitignored — this repo is
# public and must carry no real hostnames or usernames). See .env.example.
# Everything below falls back to $HOME / the invoking account, which is what
# the capture host has always resolved to. cron runs with a minimal env, so
# derive the account with `id -un` rather than trusting $USER.
FV_USER="${FV_USER:-$(id -un)}"
FV_HOME="${HOME:-/home/$FV_USER}"

FV_ENV_FILE="${FV_ENV_FILE:-$(dirname "$0")/../.env}"
if [ -f "$FV_ENV_FILE" ]; then
  set -a; . "$FV_ENV_FILE"; set +a
fi

REPO="${FV_REPO:-/media/scratch/$FV_USER/github/baseball-biomechanics/fieldvision}"
# Raw data lives on the datasets share (ALL raw-data datasets under
# /media/datasets/<user>/); transient samples + operational state on scratch.
FV_DATASET_ROOT="${FV_DATASET_ROOT:-/media/datasets/$FV_USER/fieldvision}"
FV_SCRATCH_ROOT="${FV_SCRATCH_ROOT:-/media/scratch/$FV_USER/fieldvision}"
FV_DATA="${FV_DATA_DIR:-$FV_DATASET_ROOT/data}"
FV_ROOT="$FV_SCRATCH_ROOT"
FV_LOG_DIR="${FV_LOG_DIR:-$FV_HOME/logs}"
FV_TOKEN_FILE="${FV_TOKEN_FILE:-$FV_HOME/.fv_token.txt}"

# Without the NAS there is no repo, no data dir, nothing to do. (The
# fstab entries that make mounts survive a reboot are still pending —
# until then a reboot needs a manual /scratch/mount_nas_40gb.sh.)
mountpoint -q /media/scratch || exit 1
mountpoint -q /media/datasets || exit 1

tmux has-session -t "$SESSION" 2>/dev/null && exit 0

# Log to LOCAL disk, not the NAS: a tee/redirect writing to CIFS can die
# under heavy NAS I/O (soft mount, EIO) and take the daemon down with a
# silent SIGPIPE — observed 2026-08-09.
mkdir -p "$FV_ROOT/state" "$FV_DATA" "$FV_LOG_DIR"
tmux new-session -d -s "$SESSION" \
  "cd $REPO && \
   FV_TOKEN_FILE=$FV_TOKEN_FILE \
   FV_DATA_DIR=$FV_DATA \
   FV_SAMPLES_DIR=$FV_ROOT/samples \
   FV_STATE_DIR=$FV_ROOT/state \
   FV_BACKLOG_START=2026-04-14 \
   ${FV_PYTHON:-$FV_HOME/venvs/fieldvision/bin/python3} scripts/fv_autopilot.py \
     --workers 8 --delete-bins ${FV_AUTOPILOT_ARGS:---backlog-workers 6} \
     >> $FV_LOG_DIR/autopilot.log 2>&1"
# Backlog green-lit by Spencer 2026-08-09: drains Apr 14 (dataset birthday)
# -> today oldest-first in idle slots (live games always get worker slots
# first). FV_AUTOPILOT_ARGS="--backlog-workers 0" for live-only if ever needed.
