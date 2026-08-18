#!/bin/bash
# Nightly NAS integrity sweep on Nellie (cron: 30 9 * * *).
#
# Runs from LOCAL disk, not the NAS clone: the sweep's whole job is to report
# on NAS health, so it must still run when the NAS is misbehaving (and the
# NAS is mounted noexec anyway).
#
# Deploy:  scp fieldvision/scripts/nas_integrity_sweep.py \
#              "$NELLIE:~/nas_integrity_sweep.py"
#
# Exit 1 (damage found) also appends to state/INTEGRITY_ALERT.txt, which is
# what the dashboard and the next session look at.
FV_LOG_DIR="${FV_LOG_DIR:-$HOME/logs}"
mkdir -p "$FV_LOG_DIR"
exec ~/venvs/fieldvision/bin/python3 ~/nas_integrity_sweep.py \
    >> "$FV_LOG_DIR/nightly_sweep.log" 2>&1
