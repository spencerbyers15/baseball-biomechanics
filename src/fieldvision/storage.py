"""SQLite-per-game storage for decoded FieldVision skeletal data.

Layout:
  ~/fieldvision/data/
    games_registry.sqlite           # one row per captured game
    fv_<gamePk>.sqlite              # everything for that game

Per-game schema:
  actor_frame   one row per (actor, frame), wide joint columns (xyz triples)
  ball_frame    one row per frame with a ball position
  game_event    one row per game-event (atBat / play / pitch / inning / etc)
  bones         lookup: bone_id -> name
  players       lookup: mlb_player_id -> jersey, role, position, team
  meta          key/value bag (game info, strike zone params)

DuckDB can ATTACH any of these SQLite files for cross-game analytics.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .skeleton import (BONE_NAMES, REST_OFFSET, REST_ROTATION,
                       TRACKED_NODE_IDS, forward_kinematics)
from .wire_schemas import read_tracking_data, unpack_smallest_three


# Column ordering for actor_frame joint x/y/z fields. Order matches the
# tracked nodeIds, name follows BONE_NAMES (lowercased + .RT/.LT compacted).
JOINT_COLS = [
    (0, "pelvis"), (1, "hipmaster"),
    (2, "hip_rt"), (3, "knee_rt"), (4, "foot_rt"),
    (10, "hip_lt"), (11, "knee_lt"), (12, "foot_lt"),
    (18, "torso_a"), (19, "torso_b"),
    (20, "neck"), (21, "head"),
    (25, "clavicle_rt"), (26, "shoulder_rt"), (27, "elbow_rt"), (28, "hand_rt"),
    (64, "clavicle_lt"), (65, "shoulder_lt"), (66, "elbow_lt"), (67, "hand_lt"),
]


def _xyz_columns_ddl() -> str:
    parts = []
    for _, name in JOINT_COLS:
        parts.append(f"{name}_x REAL")
        parts.append(f"{name}_y REAL")
        parts.append(f"{name}_z REAL")
    parts += [
        "bat_handle_x REAL",
        "bat_handle_y REAL",
        "bat_handle_z REAL",
    ]
    return ",\n  ".join(parts)


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS bones (
    bone_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    mlb_player_id INTEGER PRIMARY KEY,
    jersey_number INTEGER,
    role_id INTEGER,
    team TEXT,                  -- 'home' / 'away' / NULL
    position_abbr TEXT,
    parent_team_id INTEGER
);

CREATE TABLE IF NOT EXISTS labels (
    actor_uid INTEGER PRIMARY KEY,
    actor INTEGER,              -- mlb_player_id, or negative for umpires
    actor_type TEXT             -- fielder / plate-umpire / etc
);

CREATE TABLE IF NOT EXISTS actor_frame (
    game_pk INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,
    frame_num INTEGER NOT NULL,         -- frame-in-segment (0..149)
    actor_uid INTEGER NOT NULL,         -- tracking-segment id (key into labels)
    mlb_player_id INTEGER,              -- copy from labels.actor for fast queries
    actor_type TEXT,
    time_unix REAL NOT NULL,
    timestamp TEXT,
    is_gap INTEGER,
    scale REAL,
    ground REAL,
    apex REAL,
    {_xyz_columns_ddl()},
    PRIMARY KEY (game_pk, segment_idx, frame_num, actor_uid)
);
CREATE INDEX IF NOT EXISTS idx_af_player_time ON actor_frame(mlb_player_id, time_unix);
CREATE INDEX IF NOT EXISTS idx_af_time ON actor_frame(time_unix);
CREATE INDEX IF NOT EXISTS idx_af_actor ON actor_frame(actor_uid);

CREATE TABLE IF NOT EXISTS ball_frame (
    game_pk INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,
    frame_num INTEGER NOT NULL,
    time_unix REAL NOT NULL,
    ball_x REAL, ball_y REAL, ball_z REAL,
    PRIMARY KEY (game_pk, segment_idx, frame_num)
);

-- Per-frame bat orientation from inferredBat (TrackingBatPositionWire).
-- Bat axis = (head - handle); length is consistently ~34 in (standard MLB).
CREATE TABLE IF NOT EXISTS bat_frame (
    game_pk INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,
    frame_num INTEGER NOT NULL,
    time_unix REAL NOT NULL,
    head_x REAL, head_y REAL, head_z REAL,
    handle_x REAL, handle_y REAL, handle_z REAL,
    PRIMARY KEY (game_pk, segment_idx, frame_num)
);

CREATE TABLE IF NOT EXISTS pitch_event (
    game_pk INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,
    frame_num INTEGER NOT NULL,
    time_unix REAL NOT NULL,
    event_type TEXT NOT NULL,           -- 'PLAY_EVENT' | 'BEGIN_OF_PLAY' | 'BALL_WAS_RELEASED' | etc.
    play_id TEXT,                       -- populated for PLAY_EVENT (from PlayEventDataWire.playId); NULL for tracked events that don't carry one
    pos_x REAL, pos_y REAL, pos_z REAL  -- populated for tracked events that have X/Y/Z (e.g., BALL_WAS_RELEASED)
);
CREATE INDEX IF NOT EXISTS idx_pe_play   ON pitch_event(play_id);
CREATE INDEX IF NOT EXISTS idx_pe_type   ON pitch_event(event_type, time_unix);
CREATE INDEX IF NOT EXISTS idx_pe_time   ON pitch_event(time_unix);

CREATE TABLE IF NOT EXISTS pitch_label (
    game_pk INTEGER NOT NULL,
    play_id TEXT NOT NULL,
    ab_index INTEGER,
    pitch_number INTEGER,
    inning INTEGER,
    top_inning INTEGER,                 -- 1=top, 0=bottom
    batter_id INTEGER,
    pitcher_id INTEGER,
    batter_side TEXT,                   -- 'L' | 'R'
    pitcher_throws TEXT,                -- 'L' | 'R'
    balls_before INTEGER,
    strikes_before INTEGER,
    outs_before INTEGER,
    pitch_type TEXT,                    -- 'FF', 'SL', etc. (statsapi details.type.code)
    pitch_type_desc TEXT,
    start_speed REAL,                   -- mph at release
    end_speed REAL,                     -- mph at plate
    spin_rate REAL,
    spin_direction REAL,
    release_x REAL, release_y REAL, release_z REAL,
    release_extension REAL,
    plate_x REAL, plate_z REAL,
    sz_top REAL, sz_bot REAL,
    result_call TEXT,                   -- 'B' | 'S' | 'X'
    result_desc TEXT,
    is_in_play INTEGER,
    is_strike INTEGER,
    is_ball INTEGER,
    start_time TEXT,                    -- ISO from statsapi
    end_time TEXT,
    start_time_unix REAL,
    end_time_unix REAL,
    PRIMARY KEY (game_pk, play_id)
);
CREATE INDEX IF NOT EXISTS idx_pl_pitcher ON pitch_label(pitcher_id);
CREATE INDEX IF NOT EXISTS idx_pl_batter  ON pitch_label(batter_id);
CREATE INDEX IF NOT EXISTS idx_pl_type    ON pitch_label(pitch_type);
CREATE INDEX IF NOT EXISTS idx_pl_time    ON pitch_label(start_time_unix);
"""


REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_pk INTEGER PRIMARY KEY,
    venue_id INTEGER,
    away_team TEXT,
    home_team TEXT,
    earliest_ts TEXT,
    latest_ts TEXT,
    n_segments INTEGER,
    n_frames INTEGER,
    n_actor_frames INTEGER,
    db_path TEXT,
    captured_at TEXT,
    status TEXT
);
"""


def open_game_db(game_pk: int, data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / f"fv_{game_pk}.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def open_registry(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(data_dir / "games_registry.sqlite"))
    conn.executescript(REGISTRY_SCHEMA)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _build_actor_frame_row(
    game_pk: int,
    segment_idx: int,
    frame_num: int,
    time_unix: float,
    timestamp: str | None,
    is_gap: bool,
    actor_uid: int,
    actor_type: str | None,
    mlb_player_id: int | None,
    scale: float,
    ground: float,
    apex: float,
    world_pos: dict[int, "list[float]"],
    bat_pos: tuple[float, float, float] | None,
) -> tuple:
    row = [
        game_pk, segment_idx, frame_num, actor_uid,
        mlb_player_id, actor_type, time_unix, timestamp,
        1 if is_gap else 0, scale, ground, apex,
    ]
    for bone_id, _ in JOINT_COLS:
        p = world_pos.get(bone_id)
        if p is None:
            row.extend([None, None, None])
        else:
            row.extend([float(p[0]), float(p[1]), float(p[2])])
    if bat_pos is None:
        row.extend([None, None, None])
    else:
        row.extend([bat_pos[0], bat_pos[1], bat_pos[2]])
    return tuple(row)


def _actor_frame_insert_sql() -> str:
    cols = [
        "game_pk", "segment_idx", "frame_num", "actor_uid",
        "mlb_player_id", "actor_type", "time_unix", "timestamp",
        "is_gap", "scale", "ground", "apex",
    ]
    for _, name in JOINT_COLS:
        cols += [f"{name}_x", f"{name}_y", f"{name}_z"]
    cols += ["bat_handle_x", "bat_handle_y", "bat_handle_z"]
    placeholders = ", ".join("?" * len(cols))
    return f"INSERT OR REPLACE INTO actor_frame ({', '.join(cols)}) VALUES ({placeholders})"


def load_lookup_tables(
    conn: sqlite3.Connection,
    metadata_path: Path,
    labels_path: Path,
) -> dict[int, dict]:
    """Populate bones, players, labels lookup tables. Returns the labels
    dict for fast lookup during segment ingestion."""
    metadata = json.loads(metadata_path.read_text())
    labels = json.loads(labels_path.read_text())

    with transaction(conn):
        # Bones
        bone_id_map = metadata.get("boneIdMap", {})
        bat_id_map = metadata.get("batBoneIdMap", {})
        rows = [(int(k), v) for k, v in {**bone_id_map, **bat_id_map}.items()]
        conn.executemany("INSERT OR REPLACE INTO bones (bone_id, name) VALUES (?, ?)", rows)

        # Players (from boxscore)
        boxscore = metadata.get("boxscore", {}).get("teams", {})
        rows = []
        for team_key, team_data in boxscore.items():
            for player_key, p in team_data.get("players", {}).items():
                pid = p.get("person", {}).get("id")
                if pid is None:
                    continue
                rows.append((
                    int(pid),
                    p.get("jerseyNumber"),
                    None,  # role_id – fill later from rawJoints if available
                    team_key,
                    p.get("position", {}).get("abbreviation") if p.get("position") else None,
                    p.get("parentTeamId"),
                ))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO players "
                "(mlb_player_id, jersey_number, role_id, team, position_abbr, parent_team_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

        # Labels
        label_rows = []
        labels_dict = {}
        for k, v in labels.items():
            try:
                uid = int(k)
            except ValueError:
                continue
            actor = v.get("actor")
            atype = v.get("type")
            label_rows.append((uid, actor, atype))
            labels_dict[uid] = {"actor": actor, "type": atype}
        conn.executemany(
            "INSERT OR REPLACE INTO labels (actor_uid, actor, actor_type) VALUES (?, ?, ?)",
            label_rows,
        )

        # Strike zone params + game info into meta
        for k in ("version", "gamePk", "venueId"):
            if k in metadata:
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                             (k, json.dumps(metadata[k])))
        for setting in metadata.get("ruleSettings", []):
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (f"rule.{setting.get('settingName')}", json.dumps(setting.get("settingValue"))),
            )

    return labels_dict


def _pitch_event_insert_sql() -> str:
    cols = [
        "game_pk", "segment_idx", "frame_num", "time_unix",
        "event_type", "play_id",
        "pos_x", "pos_y", "pos_z",
    ]
    placeholders = ", ".join("?" * len(cols))
    return f"INSERT INTO pitch_event ({', '.join(cols)}) VALUES ({placeholders})"


def _pitch_label_insert_sql() -> str:
    cols = [
        "game_pk", "play_id",
        "ab_index", "pitch_number",
        "inning", "top_inning",
        "batter_id", "pitcher_id",
        "batter_side", "pitcher_throws",
        "balls_before", "strikes_before", "outs_before",
        "pitch_type", "pitch_type_desc",
        "start_speed", "end_speed",
        "spin_rate", "spin_direction",
        "release_x", "release_y", "release_z", "release_extension",
        "plate_x", "plate_z",
        "sz_top", "sz_bot",
        "result_call", "result_desc",
        "is_in_play", "is_strike", "is_ball",
        "start_time", "end_time",
        "start_time_unix", "end_time_unix",
    ]
    placeholders = ", ".join("?" * len(cols))
    return f"INSERT OR REPLACE INTO pitch_label ({', '.join(cols)}) VALUES ({placeholders})"


def ingest_segment(
    conn: sqlite3.Connection,
    game_pk: int,
    segment_idx: int,
    bin_path: Path,
    labels_dict: dict[int, dict],
    insert_sql: str,
) -> tuple[int, int]:
    """Decode one .bin segment and insert all actor-frames + ball-frames + bat-frames.
    Returns (n_actor_rows_inserted, n_ball_rows_inserted)."""
    td = read_tracking_data(bin_path.read_bytes())
    actor_rows: list[tuple] = []
    ball_rows: list[tuple] = []
    bat_rows: list[tuple] = []

    for f in td.frames:
        # Per-frame metadata
        is_gap = f.isGap
        time_unix = f.time
        ts = f.timestamp

        # Ball position
        if f.ballPosition is not None:
            ball_rows.append((
                game_pk, segment_idx, f.num, time_unix,
                f.ballPosition.x, f.ballPosition.y, f.ballPosition.z,
            ))

        # Bat orientation from inferredBat
        if f.inferredBat is not None and f.inferredBat.headPosition is not None \
                and f.inferredBat.handlePosition is not None:
            head = f.inferredBat.headPosition
            handle = f.inferredBat.handlePosition
            bat_rows.append((
                game_pk, segment_idx, f.num, time_unix,
                head.x, head.y, head.z,
                handle.x, handle.y, handle.z,
            ))

        # Actor poses
        for a in f.actorPoses:
            if a.rootPos is None:
                continue
            quats = [unpack_smallest_three(p) for p in a.packedQuats]
            ws = forward_kinematics(
                root_pos=(a.rootPos.x, a.rootPos.y, a.rootPos.z),
                scale=a.scale if a.scale > 0 else 1.0,
                node_ids=a.nodeIds,
                quats_xyzw=quats,
            )
            world_pos = {bid: list(p) for bid, p in ws.bone_world_pos.items()}
            label_info = labels_dict.get(a.uid, {})
            mlb_id = label_info.get("actor")
            atype = label_info.get("type")
            bat = (a.batRootPos.x, a.batRootPos.y, a.batRootPos.z) if a.batRootPos else None
            actor_rows.append(_build_actor_frame_row(
                game_pk, segment_idx, f.num, time_unix, ts, is_gap,
                a.uid, atype, mlb_id, a.scale, a.ground, a.apex,
                world_pos, bat,
            ))

    if actor_rows:
        conn.executemany(insert_sql, actor_rows)
    if ball_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO ball_frame "
            "(game_pk, segment_idx, frame_num, time_unix, ball_x, ball_y, ball_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ball_rows,
        )
    if bat_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO bat_frame "
            "(game_pk, segment_idx, frame_num, time_unix, "
            "head_x, head_y, head_z, handle_x, handle_y, handle_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            bat_rows,
        )
    return len(actor_rows), len(ball_rows)


def update_registry(
    registry_conn: sqlite3.Connection,
    game_db_conn: sqlite3.Connection,
    game_pk: int,
    db_path: Path,
) -> None:
    """Refresh the cross-game registry row for this game."""
    # Pull summary stats from the per-game DB
    cur = game_db_conn.cursor()
    n_actor_frames = cur.execute("SELECT COUNT(*) FROM actor_frame").fetchone()[0]
    n_segments = cur.execute(
        "SELECT COUNT(DISTINCT segment_idx) FROM actor_frame"
    ).fetchone()[0]
    n_frames = cur.execute(
        "SELECT COUNT(DISTINCT segment_idx || ':' || frame_num) FROM actor_frame"
    ).fetchone()[0]
    earliest_ts, latest_ts = cur.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM actor_frame"
    ).fetchone()
    venue_id = None
    venue_row = cur.execute("SELECT value FROM meta WHERE key='venueId'").fetchone()
    if venue_row:
        try:
            venue_id = int(json.loads(venue_row[0]))
        except Exception:
            pass

    with transaction(registry_conn):
        registry_conn.execute(
            """
            INSERT OR REPLACE INTO games
              (game_pk, venue_id, away_team, home_team,
               earliest_ts, latest_ts, n_segments, n_frames,
               n_actor_frames, db_path, captured_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_pk, venue_id, None, None,
                earliest_ts, latest_ts, n_segments, n_frames,
                n_actor_frames, str(db_path),
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                "loaded",
            ),
        )
