"""Wall-clock time with an optional simulated offset.

Every timestamp in the system is unix epoch seconds (a float) as returned by
``Clock.now()``. The offset exists only for the ``/sleep`` simulator; real
operation never advances it. Anything recorded while the offset is nonzero is
flagged ``simulated`` so real time stays distinguishable in the logs.
"""

from __future__ import annotations

import time
from datetime import datetime


class Clock:
    def __init__(self) -> None:
        self._offset = 0.0

    def now(self) -> float:
        return time.time() + self._offset

    @property
    def offset(self) -> float:
        return self._offset

    @property
    def simulated(self) -> bool:
        return self._offset > 0

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot advance the clock backwards")
        self._offset += seconds


def human_duration(seconds: float) -> str:
    """A natural-language duration: 'a few seconds', 'about 40 minutes'."""
    s = max(0.0, seconds)
    if s < 5:
        return "a few seconds"
    if s < 60:
        return f"{int(s)} seconds"
    minutes = s / 60
    if minutes < 1.5:
        return "about a minute"
    if minutes < 55:
        return f"about {round(minutes)} minutes"
    hours = s / 3600
    if hours < 1.25:
        return "about an hour"
    if hours < 1.75:
        return "about an hour and a half"
    if hours < 23:
        return f"about {round(hours)} hours"
    days = s / 86400
    if days < 1.5:
        return "about a day"
    return f"about {round(days)} days"


def local_stamp(ts: float) -> str:
    """Local human-readable timestamp for display and logs."""
    return datetime.fromtimestamp(ts).strftime("%a %Y-%m-%d %H:%M:%S")
