"""Test that ingest_pitch_labels populates pitch_label rows from a statsapi feed."""
import sys
from pathlib import Path

# Make scripts/ importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fieldvision.storage import open_game_db, transaction
from ingest_pitch_labels import ingest_feed_dict


def test_ingest_pitch_label_from_minimal_feed(tmp_path: Path):
    game_pk = 999900
    conn = open_game_db(game_pk, tmp_path)
    feed = {
        "liveData": {"plays": {"allPlays": [{
            "atBatIndex": 5,
            "about": {"inning": 3, "halfInning": "top"},
            "matchup": {
                "batter": {"id": 660271}, "pitcher": {"id": 543037},
                "batSide": {"code": "L"}, "pitcherHand": {"code": "R"},
            },
            "playEvents": [{
                "isPitch": True,
                "playId": "abc12345-aaaa-bbbb-cccc-111122223333",
                "pitchNumber": 2,
                "count": {"balls": 1, "strikes": 0, "outs": 1},
                "details": {
                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                    "call": {"code": "S"}, "description": "Called Strike",
                    "isInPlay": False, "isStrike": True, "isBall": False,
                },
                "pitchData": {
                    "startSpeed": 95.4, "endSpeed": 87.1,
                    "breaks": {"spinRate": 2300, "spinDirection": 215},
                    "extension": 6.4,
                    "strikeZoneTop": 3.5, "strikeZoneBottom": 1.6,
                    "coordinates": {
                        "x0": -1.5, "y0": 50.0, "z0": 5.9,
                        "px": 0.2, "pz": 2.4,
                    },
                },
                "startTime": "2026-05-06T19:53:43.223Z",
                "endTime":   "2026-05-06T19:53:43.678Z",
            }],
        }]}}
    }
    with transaction(conn):
        n = ingest_feed_dict(conn, game_pk, feed)
    assert n == 1
    row = conn.execute(
        "SELECT pitch_type, start_speed, batter_id, pitcher_id, "
        "balls_before, strikes_before, start_time_unix, "
        "release_x, release_y, release_z, plate_x, plate_z, "
        "sz_top, sz_bot, result_call, is_strike "
        "FROM pitch_label WHERE play_id='abc12345-aaaa-bbbb-cccc-111122223333'"
    ).fetchone()
    assert row is not None, "row not inserted"
    pitch_type, start_speed, batter_id, pitcher_id, balls_before, strikes_before, start_time_unix, \
        release_x, release_y, release_z, plate_x, plate_z, \
        sz_top, sz_bot, result_call, is_strike = row
    assert pitch_type == "FF"
    assert abs(start_speed - 95.4) < 0.01
    assert batter_id == 660271
    assert pitcher_id == 543037
    assert balls_before == 1 and strikes_before == 0
    assert start_time_unix is not None and start_time_unix > 1.7e9  # post-2024
    assert abs(release_x - (-1.5)) < 0.01
    assert abs(release_y - 50.0) < 0.01
    assert abs(release_z - 5.9) < 0.01
    assert abs(plate_x - 0.2) < 0.01
    assert abs(plate_z - 2.4) < 0.01
    assert abs(sz_top - 3.5) < 0.01
    assert abs(sz_bot - 1.6) < 0.01
    assert result_call == "S"
    assert is_strike == 1


def test_ingest_skips_non_pitch_events(tmp_path: Path):
    game_pk = 999901
    conn = open_game_db(game_pk, tmp_path)
    feed = {
        "liveData": {"plays": {"allPlays": [{
            "atBatIndex": 0,
            "about": {"inning": 1, "halfInning": "bottom"},
            "matchup": {"batter": {"id": 1}, "pitcher": {"id": 2}},
            "playEvents": [
                {"isPitch": False, "playId": "should-not-appear", "type": "pickoff"},
                {"isPitch": True, "playId": "real-pitch-uuid",
                 "details": {"type": {"code": "SL"}}},
            ],
        }]}}
    }
    with transaction(conn):
        n = ingest_feed_dict(conn, game_pk, feed)
    assert n == 1
    rows = conn.execute("SELECT play_id FROM pitch_label").fetchall()
    assert rows == [("real-pitch-uuid",)]
