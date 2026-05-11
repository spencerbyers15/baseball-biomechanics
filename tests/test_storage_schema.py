"""Smoke tests for the per-game SQLite schema (v2: pitch_event + pitch_label)."""
from pathlib import Path
from fieldvision.storage import open_game_db


def test_pitch_event_columns(tmp_path: Path):
    conn = open_game_db(999998, tmp_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(pitch_event)")}
    expected = {
        "game_pk", "segment_idx", "frame_num", "time_unix",
        "event_type", "play_id", "pos_x", "pos_y", "pos_z",
    }
    missing = expected - cols
    assert not missing, f"missing columns: {missing}"


def test_pitch_label_columns(tmp_path: Path):
    conn = open_game_db(999997, tmp_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(pitch_label)")}
    expected = {
        "game_pk", "play_id", "ab_index", "pitch_number",
        "inning", "top_inning",
        "batter_id", "pitcher_id", "batter_side", "pitcher_throws",
        "balls_before", "strikes_before", "outs_before",
        "pitch_type", "pitch_type_desc",
        "start_speed", "end_speed", "spin_rate", "spin_direction",
        "release_x", "release_y", "release_z", "release_extension",
        "plate_x", "plate_z", "sz_top", "sz_bot",
        "result_call", "result_desc",
        "is_in_play", "is_strike", "is_ball",
        "start_time", "end_time", "start_time_unix", "end_time_unix",
    }
    missing = expected - cols
    assert not missing, f"missing columns: {missing}"


def test_indexes_exist(tmp_path: Path):
    conn = open_game_db(999996, tmp_path)
    indexes = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )}
    assert "idx_pe_play" in indexes
    assert "idx_pe_type" in indexes
    assert "idx_pl_pitcher" in indexes
    assert "idx_pl_type" in indexes


def test_no_game_event_table(tmp_path: Path):
    """Old v1 game_event table should not exist in fresh DBs."""
    conn = open_game_db(999995, tmp_path)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert "game_event" not in tables, "game_event was dropped in v2; should not be in fresh schema"
