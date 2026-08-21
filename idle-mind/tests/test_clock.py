import time

import pytest

from idle_mind.clock import Clock, human_duration


def test_now_tracks_wall_clock():
    c = Clock()
    assert abs(c.now() - time.time()) < 0.5
    assert not c.simulated


def test_advance_offsets_time():
    c = Clock()
    before = c.now()
    c.advance(3600)
    assert c.now() - before >= 3600
    assert c.simulated
    assert c.offset == 3600


def test_advance_rejects_negative():
    c = Clock()
    with pytest.raises(ValueError):
        c.advance(-5)


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "a few seconds"),
        (3, "a few seconds"),
        (12, "12 seconds"),
        (59, "59 seconds"),
        (70, "about a minute"),
        (40 * 60, "about 40 minutes"),
        (60 * 60, "about an hour"),
        (95 * 60, "about an hour and a half"),
        (5 * 3600, "about 5 hours"),
        (26 * 3600, "about a day"),
        (3 * 86400, "about 3 days"),
    ],
)
def test_human_duration(seconds, expected):
    assert human_duration(seconds) == expected
