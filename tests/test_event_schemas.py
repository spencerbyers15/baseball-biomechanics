"""Tests for minimal v2 wire schema decoders (GameEvent.playId + TrackedEvent flat)."""

from tests.conftest import (
    FIXTURE_SEG_WITH_GAME_EVENTS,
    FIXTURE_SEG_WITH_TRACKED_EVENTS,
    fixture_bin_path,
)
from fieldvision.wire_schemas import read_tracking_data


def test_game_events_decode_with_dataType():
    """GameEvent populates with a non-negative dataType integer."""
    game_pk, seg = FIXTURE_SEG_WITH_GAME_EVENTS
    td = read_tracking_data(fixture_bin_path(game_pk, seg).read_bytes())
    found = False
    for f in td.frames:
        for ge in f.gameEvents:
            assert isinstance(ge.dataType, int) and ge.dataType >= 0
            found = True
    assert found, "expected at least one gameEvent in the fixture"


def test_tracked_events_have_event_types():
    """TrackedEvent in our fixture should yield real eventType strings (e.g., BALL_WAS_RELEASED)."""
    game_pk, seg = FIXTURE_SEG_WITH_TRACKED_EVENTS
    td = read_tracking_data(fixture_bin_path(game_pk, seg).read_bytes())
    etypes = set()
    for f in td.frames:
        for te in f.trackedEvents:
            if te.eventType:
                etypes.add(te.eventType)
    assert etypes, "expected at least one tracked event with eventType"
    assert any("BALL_WAS_" in et for et in etypes), f"unexpected event types: {etypes}"


def test_play_event_play_id_extraction():
    """At least one PlayEvent (dataType=7) somewhere in the first 500 segments should yield a UUID-shaped playId.

    PlayEvents only appear deep in the segment stream (recon found first one around seg 352).
    This test scans the file system to find a PlayEvent rather than relying on the standard fixtures."""
    from pathlib import Path
    samples = sorted(
        (Path(__file__).resolve().parents[1] / "samples" / "binary_capture_823141").glob("mlb_823141_segment_*.bin"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )[:500]
    found_play_id = None
    for p in samples:
        td = read_tracking_data(p.read_bytes())
        for f in td.frames:
            for ge in f.gameEvents:
                if ge.dataType == 7 and ge.playId:
                    found_play_id = ge.playId
                    break
            if found_play_id:
                break
        if found_play_id:
            break
    assert found_play_id is not None, "expected at least one PlayEvent.playId in first 500 segments"
    # UUID format: 8-4-4-4-12 hex with dashes = 36 chars total
    assert len(found_play_id) == 36, f"playId not UUID-shaped: {found_play_id!r}"
    assert found_play_id.count("-") == 4, f"playId not UUID-shaped: {found_play_id!r}"
