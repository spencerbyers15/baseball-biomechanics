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

mkdir -p "$FV_ROOT/state" "$FV_DATA"
tmux new-session -d -s "$SESSION" \
  "cd $REPO && \
   FV_TOKEN_FILE=/home/spencer/.fv_token.txt \
   FV_DATA_DIR=$FV_DATA \
   FV_SAMPLES_DIR=$FV_ROOT/samples \
   FV_STATE_DIR=$FV_ROOT/state \
   /home/spencer/venvs/fieldvision/bin/python3 scripts/fv_autopilot.py \
     --workers 8 --delete-bins ${FV_AUTOPILOT_ARGS:-} \
     >> $FV_ROOT/state/autopilot.log 2>&1"
