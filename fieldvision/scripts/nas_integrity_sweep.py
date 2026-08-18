#!/usr/bin/env python3
"""Readability sweep over the FieldVision corpus — does the NAS still have
files it told us it wrote?

Background: on 2026-08-11 a readability check found ~20 canonical parquet
files broken **after** successful writes (EINVAL-on-open, lost footers,
vanished files), correlated with ENOSPC-jam-era writes and Unraid mover
cycles. Scratch-corpus deletion is on hold until we know whether it recurs,
so this sweep is the experiment's instrument — and an instrument that cries
wolf is worse than none.

## Why the naive version cried wolf

The first version flagged any `actor_frames*.parquet` whose last 4 bytes
weren't `PAR1`. But a parquet file gets its footer only when the writer
closes it, so **every game currently being captured looks corrupt**. The
autopilot keeps ~4 games in flight at all times, so a 09:30 UTC sweep
reliably reported ~4 "bad" files that were simply mid-write — verified
2026-08-12, when all four flagged files were held open by the four live
capture worker PIDs.

This version excludes in-flight files two ways (belt and braces, because a
false negative here is much cheaper than a false alarm that trains us to
ignore the alert):

  1. **open by a live process** — scan `/proc/*/fd` for the path;
  2. **too recently modified** — mtime within `--min-age-min` (default 15),
     which also covers a writer that died and left a partial file behind
     before we could see its fd.

## What counts as damage

Beyond the cheap `PAR1` footer probe, this reads each file's parquet
**metadata** (a few KB from the tail, not the data). That is what actually
catches the failure mode we saw: a file whose footer is truncated or
structurally unreadable can still end in the right magic bytes.

Usage:
    nas_integrity_sweep.py                 # sweep both trees, alert on damage
    nas_integrity_sweep.py --quick         # footer probe only, skip metadata
    nas_integrity_sweep.py --json          # machine-readable summary

Exit codes: 0 = clean, 1 = damage found, 2 = nothing to sweep (NAS unmounted).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Host-specific roots come from the environment (fieldvision/.env — see
# .env.example) so this public repo carries no real usernames. Defaults
# derive from $USER and match what the capture host has always used.
_USER = os.environ.get("USER") or Path.home().name
_DATASET_ROOT = os.environ.get("FV_DATASET_ROOT", f"/media/datasets/{_USER}/fieldvision")
_SCRATCH_ROOT = os.environ.get("FV_SCRATCH_ROOT", f"/media/scratch/{_USER}/fieldvision")

DEFAULT_TREES = [
    f"{_DATASET_ROOT}/data",
    f"{_SCRATCH_ROOT}/data",
]
ALERT_FILE = Path(os.environ.get(
    "FV_INTEGRITY_ALERT", f"{_SCRATCH_ROOT}/state/INTEGRITY_ALERT.txt"))


def open_paths() -> set[str]:
    """Every file currently held open by a process we can see.

    Reading /proc is cheap and exact, and beats shelling out to lsof (which
    stats every fd on a 256-core box and is slow enough to matter here).
    """
    live: set[str] = set()
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        try:
            entries = os.listdir(fd_dir)
        except OSError:
            continue  # process exited, or not ours — either way, move on
        for entry in entries:
            try:
                live.add(os.readlink(os.path.join(fd_dir, entry)))
            except OSError:
                continue
    return live


def check_file(path: str, quick: bool) -> str | None:
    """Return a damage description, or None if the file reads fine."""
    try:
        with open(path, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            if fh.read(4) != b"PAR1":
                return "no footer magic"
    except OSError as exc:
        return f"unreadable: {exc}"

    if quick:
        return None

    try:
        import pyarrow.parquet as pq
        pq.read_metadata(path)
    except Exception as exc:
        # Truncated/structurally broken footers still end in PAR1, so this is
        # the check that actually catches the 2026-08-11 failure mode.
        return f"bad parquet metadata: {type(exc).__name__}: {exc}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trees", nargs="*", default=DEFAULT_TREES)
    parser.add_argument("--pattern", default="*/actor_frames*.parquet")
    parser.add_argument("--min-age-min", type=float, default=15.0,
                        help="skip files modified within this many minutes "
                             "(they are probably still being written)")
    parser.add_argument("--quick", action="store_true",
                        help="footer probe only; skip the parquet metadata read")
    parser.add_argument("--json", action="store_true",
                        help="emit a JSON summary instead of text")
    args = parser.parse_args()

    files: list[str] = []
    for tree in args.trees:
        if not os.path.isdir(tree):
            continue
        files.extend(glob.glob(os.path.join(tree, args.pattern)))

    if not files:
        print(f"[{time.strftime('%F %T')}] sweep: no files found — is the NAS "
              f"mounted?")
        return 2

    live = open_paths()
    now = time.time()
    cutoff = args.min_age_min * 60

    bad: list[dict] = []
    skipped_open = skipped_fresh = checked = 0

    for path in sorted(files):
        if path in live:
            skipped_open += 1
            continue
        try:
            age = now - os.stat(path).st_mtime
        except OSError as exc:
            bad.append({"path": path, "reason": f"vanished: {exc}"})
            continue
        if age < cutoff:
            skipped_fresh += 1
            continue

        checked += 1
        reason = check_file(path, args.quick)
        if reason:
            bad.append({"path": path, "reason": reason})

    stamp = time.strftime("%F %T")
    summary = {
        "stamp": stamp, "total": len(files), "checked": checked,
        "skipped_open": skipped_open, "skipped_fresh": skipped_fresh,
        "bad": bad,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"[{stamp}] sweep: {len(bad)} bad / {checked} checked "
              f"({skipped_open} in-flight, {skipped_fresh} too fresh, "
              f"{len(files)} total)")
        for entry in bad:
            print(f"   {entry['path']} -> {entry['reason']}")

    if bad:
        try:
            ALERT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with ALERT_FILE.open("a") as fh:
                fh.write(f"{stamp}: {len(bad)} bad files\n")
                for entry in bad:
                    fh.write(f"    {entry['path']} -> {entry['reason']}\n")
        except OSError as exc:
            print(f"WARNING: could not write alert file: {exc}",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
