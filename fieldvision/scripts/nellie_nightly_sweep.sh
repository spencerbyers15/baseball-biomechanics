#!/bin/bash
# Nightly NAS integrity sweep on Nellie (cron: 30 9 * * *).
#
# Runs from LOCAL disk, not the NAS clone: the sweep's whole job is to report
# on NAS health, so it must still run when the NAS is misbehaving (and the
# NAS is mounted noexec anyway).
#
# Deploy:  scp fieldvision/scripts/nas_integrity_sweep.py \
#              nellie:/home/spencer/nas_integrity_sweep.py
#
# Exit 1 (damage found) also appends to state/INTEGRITY_ALERT.txt, which is
# what the dashboard and the next session look at.
exec ~/venvs/fieldvision/bin/python3 /home/spencer/nas_integrity_sweep.py \
    >> /home/spencer/logs/nightly_sweep.log 2>&1
