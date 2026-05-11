"""Smoke tests for the per-game SQLite schema."""

import sqlite3
from pathlib import Path

from fieldvision.storage import open_game_db


def test_game_event_columns(tmp_path: Path):
    conn = open_game_db(999999, tmp_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(game_event)")}
    expected = {
        "game_pk", "segment_idx", "frame_num", "time_unix",
        "event_type", "data_type", "is_key_framed", "play_id",
        "balls", "strikes", "outs",
        "inning", "top_inning",
        "batter_id", "pitcher_id",
        "batter_handedness", "pitcher_handedness",
        "data_json",
    }
    missing = expected - cols
    assert not missing, f"missing columns: {missing}"


def test_pitch_event_columns(tmp_path: Path):
    conn = open_game_db(999998, tmp_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(pitch_event)")}
    expected = {
        "game_pk", "segment_idx", "frame_num", "time_unix",
        "play_id",
        "release_x", "release_y", "release_z",
        "plate_x", "plate_y", "plate_z",
        "velocity_release", "velocity_plate",
        "spin_rate", "spin_axis_x", "spin_axis_y", "spin_axis_z",
        "pitch_type", "pitch_type_id", "extension",
        "sz_top", "sz_bottom", "sz_left", "sz_right",
    }
    missing = expected - cols
    assert not missing, f"missing columns: {missing}"


def test_indexes_exist(tmp_path: Path):
    conn = open_game_db(999997, tmp_path)
    indexes = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )}
    assert "idx_event_type" in indexes
    assert "idx_event_play" in indexes
    assert "idx_pitch_play" in indexes
