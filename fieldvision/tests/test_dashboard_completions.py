"""The dashboard's completions/hour must count REAL completions, not passes.

Regression cover for the inflated "Games completed per hour" chart: it counted
`pk=N game done` log lines, which fire on every 5-min resume/verify pass over
an already-finished game. In one sampled hour that was 107 log lines for 3 real
completions — a ~25x overstatement that also poisoned the ETA. The true signal
is the marker-writing lines, which carry a full YYYY-MM-DD HH:MM:SS stamp.
"""
import importlib.util
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fv_dashboard.py"
spec = importlib.util.spec_from_file_location("fv_dashboard", SCRIPT)
fv_dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv_dashboard)

completions_by_hour = fv_dashboard.completions_by_hour

NOW = datetime(2026, 8, 26, 12, 0, 0)


def _log(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_counts_each_kind_of_completion_once():
    log = _log(
        "[2026-08-26 09:15:01] backlog pk=823203 DONE fetched=2108 total=2116",
        "[2026-08-26 09:40:22] live pk=823181 finalized + marked complete",
        "[2026-08-26 10:05:00] backlog pk=999001 EXPIRED from retention — marked",
        "[2026-08-26 10:30:00] live pk=823669 no manifest after 3 passes "
        "(venue likely has no Hawk-Eye) — marked, will not retry",
    )
    hours = completions_by_hour(log, NOW)
    assert hours == Counter({9: 2, 10: 2})
    assert sum(hours.values()) == 4


def test_ignores_generic_game_done_passes():
    """The whole bug: `game done` lines must NOT count."""
    log = _log(
        "[10:05:13]   pk=823427 game done: 10 fetched, 0 failed, 4634 total",
        "[10:06:01]   pk=824641 game done: 2 fetched, 0 failed, 2917 total",
        "[10:07:44]   pk=824077 game done: 0 fetched, 0 failed, 2339 total",
    )
    assert sum(completions_by_hour(log, NOW).values()) == 0


def test_real_completion_not_shadowed_by_a_same_game_pass():
    """A game logs many `game done` passes then one real DONE — count 1."""
    log = _log(
        "[09:50:13]   pk=823203 game done: 1 fetched, 0 failed, 2116 total",
        "[09:55:13]   pk=823203 game done: 1 fetched, 0 failed, 2116 total",
        "[2026-08-26 09:58:20] backlog pk=823203 DONE fetched=2108 total=2116",
    )
    assert completions_by_hour(log, NOW) == Counter({9: 1})


def test_drops_events_older_than_24h():
    log = _log(
        "[2026-08-25 08:00:00] backlog pk=1 DONE fetched=1 total=1",   # 28h ago
        "[2026-08-26 08:00:00] backlog pk=2 DONE fetched=1 total=1",   # 4h ago
    )
    hours = completions_by_hour(log, NOW)
    assert hours == Counter({8: 1})  # only the recent one


def test_empty_log_is_zero():
    assert completions_by_hour("", NOW) == Counter()
