"""Test that ingest_segment populates pitch_event rows from wire events."""
from pathlib import Path

from fieldvision.storage import (
    _actor_frame_insert_sql, ingest_segment, open_game_db, transaction,
)
from tests.conftest import FIXTURE_SEG_WITH_TRACKED_EVENTS, fixture_bin_path


def test_ingest_writes_pitch_event_markers(tmp_path: Path):
    """Segment 131 has both gameEvents (1) and trackedEvents (2 — BALL_WAS_RELEASED + BALL_WAS_CAUGHT)."""
    game_pk, seg_idx = FIXTURE_SEG_WITH_TRACKED_EVENTS
    bin_path = fixture_bin_path(game_pk, seg_idx)
    conn = open_game_db(game_pk, tmp_path)
    with transaction(conn):
        ingest_segment(conn, game_pk, seg_idx, bin_path, {}, _actor_frame_insert_sql())

    n = conn.execute("SELECT COUNT(*) FROM pitch_event").fetchone()[0]
    assert n >= 2, f"expected at least 2 pitch_event rows from seg 131, got {n}"

    # Both BALL_WAS_RELEASED and BALL_WAS_CAUGHT should appear
    etypes = {row[0] for row in conn.execute(
        "SELECT DISTINCT event_type FROM pitch_event"
    )}
    assert "BALL_WAS_RELEASED" in etypes, f"missing BALL_WAS_RELEASED, got {etypes}"
    assert "BALL_WAS_CAUGHT" in etypes, f"missing BALL_WAS_CAUGHT, got {etypes}"

    # Tracked-event rows should have non-null pos_x/y/z
    pos_rows = conn.execute(
        "SELECT pos_x, pos_y, pos_z FROM pitch_event "
        "WHERE event_type='BALL_WAS_RELEASED'"
    ).fetchall()
    assert pos_rows, "expected at least one BALL_WAS_RELEASED row"
    px, py, pz = pos_rows[0]
    assert px is not None and py is not None and pz is not None
    # Released ball is on the mound, ~50-65 ft from home plate. Y should be 90-110 ft (recon saw 97.95).
    assert 80 < py < 130, f"BALL_WAS_RELEASED y={py} not in plausible mound-distance range"

    # Any PLAY_EVENT row in seg 131 should have a UUID-shaped play_id
    for (pid,) in conn.execute(
        "SELECT play_id FROM pitch_event WHERE event_type='PLAY_EVENT'"
    ):
        assert pid and len(pid) == 36 and pid.count("-") == 4, f"bad play_id: {pid}"


def test_ingest_segment_with_explicit_pe_sql(tmp_path: Path):
    """When the caller pre-builds pitch_event_insert_sql, ingest_segment uses it instead of rebuilding."""
    from fieldvision.storage import _pitch_event_insert_sql
    game_pk, seg_idx = FIXTURE_SEG_WITH_TRACKED_EVENTS
    bin_path = fixture_bin_path(game_pk, seg_idx)
    conn = open_game_db(game_pk, tmp_path)
    pe_sql = _pitch_event_insert_sql()
    with transaction(conn):
        ingest_segment(
            conn, game_pk, seg_idx, bin_path, {},
            _actor_frame_insert_sql(),
            pitch_event_insert_sql=pe_sql,
        )
    n = conn.execute("SELECT COUNT(*) FROM pitch_event").fetchone()[0]
    assert n >= 2
