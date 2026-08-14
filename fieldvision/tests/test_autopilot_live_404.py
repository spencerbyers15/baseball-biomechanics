"""Give-up logic for Final games that have no Hawk-Eye manifest.

Regression cover for pk=823669 (2026-08-13 Field of Dreams, PHI @ MIN): the
venue has no Hawk-Eye rig, fieldvision-hls served no manifest, and the
live/closing path retried it 107 times in one day because only the *backlog*
path knew how to mark a 404 complete.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fv_autopilot.py"
spec = importlib.util.spec_from_file_location("fv_autopilot", SCRIPT)
fv_autopilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv_autopilot)

live_404_streak = fv_autopilot.live_404_streak
GIVE_UP = fv_autopilot.LIVE_404_GIVE_UP

NOT_FOUND = {"ok": False, "error": "manifest HTTP 404"}
OK = {"ok": True, "fetched": 12}


def test_consecutive_404s_on_final_game_reach_give_up():
    """The Field of Dreams case: repeated 404s eventually stop the retries."""
    streak = 0
    for _ in range(GIVE_UP):
        streak = live_404_streak("Final", NOT_FOUND, streak)
    assert streak >= GIVE_UP


def test_single_404_does_not_give_up():
    """One blip must not permanently skip a real game."""
    assert live_404_streak("Final", NOT_FOUND, 0) < GIVE_UP


def test_success_breaks_the_streak():
    assert live_404_streak("Final", OK, GIVE_UP - 1) == 0


def test_other_error_breaks_the_streak():
    """Only *consecutive* 404s count — a 403 in between resets it."""
    streak = live_404_streak("Final", NOT_FOUND, GIVE_UP - 1)
    assert streak >= GIVE_UP  # would have given up...
    reset = live_404_streak("Final", {"ok": False, "error": "manifest HTTP 403"},
                            GIVE_UP - 1)
    assert reset == 0  # ...but a different error starts over


@pytest.mark.parametrize("state", ["Live", "Preview", None])
def test_non_final_games_never_accumulate(state):
    """A game still in progress may 404 before its stream starts — never mark it."""
    streak = 0
    for _ in range(GIVE_UP * 3):
        streak = live_404_streak(state, NOT_FOUND, streak)
    assert streak == 0
