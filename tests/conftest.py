"""Shared pytest fixtures for fieldvision tests."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Sample segments confirmed (via Task 4 recon) to contain events.
# Format: (game_pk, segment_idx)
# Segment 23: well into the game, gameEvents=4 (count updates / atbat boundaries)
# Segment 131: has BOTH gameEvents=1 AND trackedEvents=2 (pitch event segment)
# Segment 167: late segment with trackedEvents=4 (deep into game pitches)
FIXTURE_SEG_WITH_GAME_EVENTS = (823141, 23)       # gameEvents=4, trackedEvents=0
FIXTURE_SEG_WITH_TRACKED_EVENTS = (823141, 131)   # gameEvents=1, trackedEvents=2 (both present)
FIXTURE_SEG_LATE = (823141, 167)                  # gameEvents=0, trackedEvents=4


def fixture_bin_path(game_pk: int, seg_idx: int):
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "samples" / \
        f"binary_capture_{game_pk}" / f"mlb_{game_pk}_segment_{seg_idx}.bin"
