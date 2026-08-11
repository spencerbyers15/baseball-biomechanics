#!/bin/bash
# Launch (or re-launch) the FieldVision autopilot inside tmux on Nellie.
# Idempotent: exits quietly if the session is already up. Invoked from
# cron every 10 minutes as a keepalive (see launchd/nellie_autopilot_cron.txt).
#
# The NAS is mounted noexec, so cron must invoke this as `bash <path>`.

SESSION=fv-autopilot
REPO=/media/scratch/spencer/github/baseball-biomechanics/fieldvision
# Raw data lives on the datasets share (Spencer's convention: ALL raw-data
# datasets under /media/datasets/spencer/); transient samples + operational
# state stay on scratch.
FV_DATA=/media/datasets/spencer/fieldvision/data
FV_ROOT=/media/scratch/spencer/fieldvision

# Without the NAS there is no repo, no data dir, nothing to do. (The
# fstab entries that make mounts survive a reboot are still pending —
# until then a reboot needs a manual /scratch/mount_nas_40gb.sh.)
mountpoint -q /media/scratch || exit 1
mountpoint -q /media/datasets || exit 1

tmux has-session -t "$SESSION" 2>/dev/null && exit 0

# Log to LOCAL disk, not the NAS: a tee/redirect writing to CIFS can die
# under heavy NAS I/O (soft mount, EIO) and take the daemon down with a
# silent SIGPIPE — observed 2026-08-09.
mkdir -p "$FV_ROOT/state" "$FV_DATA" /home/spencer/logs
tmux new-session -d -s "$SESSION" \
  "cd $REPO && \
   FV_TOKEN_FILE=/home/spencer/.fv_token.txt \
   FV_DATA_DIR=$FV_DATA \
   FV_SAMPLES_DIR=$FV_ROOT/samples \
   FV_STATE_DIR=$FV_ROOT/state \
   FV_BACKLOG_START=2026-04-14 \
   /home/spencer/venvs/fieldvision/bin/python3 scripts/fv_autopilot.py \
     --workers 8 --delete-bins ${FV_AUTOPILOT_ARGS:---backlog-workers 6} \
     >> /home/spencer/logs/autopilot.log 2>&1"
# Backlog green-lit by Spencer 2026-08-09: drains Apr 14 (dataset birthday)
# -> today oldest-first in idle slots (live games always get worker slots
# first). FV_AUTOPILOT_ARGS="--backlog-workers 0" for live-only if ever needed.
