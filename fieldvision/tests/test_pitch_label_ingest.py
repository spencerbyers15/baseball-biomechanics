"""Test that ingest_pitch_labels writes pitch_labels.parquet rows from a statsapi feed."""
import sys
from pathlib import Path

import pyarrow.parquet as pq

# Make scripts/ importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ingest_pitch_labels import feed_to_rows, write_pitch_labels


def _ingest(game_pk: int, data_dir: Path, feed: dict) -> tuple[int, list[dict]]:
    n = write_pitch_labels(game_pk, data_dir, feed_to_rows(game_pk, feed))
    rows = pq.read_table(data_dir / str(game_pk) / "pitch_labels.parquet").to_pylist()
    return n, rows


def test_ingest_pitch_label_from_minimal_feed(tmp_path: Path):
    game_pk = 999900
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
    n, rows = _ingest(game_pk, tmp_path, feed)
    assert n == 1
    matches = [r for r in rows if r["play_id"] == "abc12345-aaaa-bbbb-cccc-111122223333"]
    assert len(matches) == 1, "row not inserted"
    row = matches[0]
    assert row["pitch_type"] == "FF"
    assert abs(row["start_speed"] - 95.4) < 0.01
    assert row["batter_id"] == 660271
    assert row["pitcher_id"] == 543037
    assert row["balls_before"] == 1 and row["strikes_before"] == 0
    assert row["start_time_unix"] is not None and row["start_time_unix"] > 1.7e9  # post-2024
    assert abs(row["release_x"] - (-1.5)) < 0.01
    assert abs(row["release_y"] - 50.0) < 0.01
    assert abs(row["release_z"] - 5.9) < 0.01
    assert abs(row["plate_x"] - 0.2) < 0.01
    assert abs(row["plate_z"] - 2.4) < 0.01
    assert abs(row["sz_top"] - 3.5) < 0.01
    assert abs(row["sz_bot"] - 1.6) < 0.01
    assert row["result_call"] == "S"
    assert row["is_strike"] == 1


def test_ingest_skips_non_pitch_events(tmp_path: Path):
    game_pk = 999901
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
    n, rows = _ingest(game_pk, tmp_path, feed)
    assert n == 1
    assert [r["play_id"] for r in rows] == ["real-pitch-uuid"]


def _pitch_feed(*pitches: tuple[str, str]) -> dict:
    """A one-at-bat feed with one pitch per (play_id, pitch_type) pair."""
    return {
        "liveData": {"plays": {"allPlays": [{
            "atBatIndex": 0,
            "about": {"inning": 1, "halfInning": "top"},
            "matchup": {"batter": {"id": 1}, "pitcher": {"id": 2}},
            "playEvents": [
                {"isPitch": True, "playId": pid, "details": {"type": {"code": code}}}
                for pid, code in pitches
            ],
        }]}}
    }


def test_reingest_replaces_rows_by_play_id(tmp_path: Path):
    game_pk = 999902
    _ingest(game_pk, tmp_path, _pitch_feed(("p1", "FF"), ("p2", "SL")))
    # p2 is re-classified and p3 is new; p1 isn't in this pull and must survive.
    n, rows = _ingest(game_pk, tmp_path, _pitch_feed(("p2", "CH"), ("p3", "CU")))
    assert n == 2
    by_id = {r["play_id"]: r["pitch_type"] for r in rows}
    assert len(rows) == 3, "a re-ingested pitch was duplicated"
    assert by_id == {"p1": "FF", "p2": "CH", "p3": "CU"}


def test_reingest_is_idempotent(tmp_path: Path):
    game_pk = 999903
    feed = _pitch_feed(("p1", "FF"), ("p2", "SL"))
    _ingest(game_pk, tmp_path, feed)
    _, rows = _ingest(game_pk, tmp_path, feed)
    assert sorted(r["play_id"] for r in rows) == ["p1", "p2"]


def test_merges_an_old_file_with_reordered_columns_and_narrow_ints(tmp_path: Path):
    """Files migrated from SQLite may differ from the canonical schema."""
    import pyarrow as pa
    import pyarrow.compute as pc
    from ingest_pitch_labels import _SCHEMA

    game_pk = 999904
    _ingest(game_pk, tmp_path, _pitch_feed(("p1", "FF")))
    path = tmp_path / str(game_pk) / "pitch_labels.parquet"
    old = pq.read_table(path)
    old = old.set_column(0, "game_pk", pc.cast(old.column("game_pk"), pa.int32()))
    pq.write_table(old.select(list(reversed(old.schema.names))), path)

    _, rows = _ingest(game_pk, tmp_path, _pitch_feed(("p2", "SL")))
    assert sorted(r["play_id"] for r in rows) == ["p1", "p2"]
    assert pq.read_table(path).schema == _SCHEMA
