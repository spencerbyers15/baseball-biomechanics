"""Parquet-first storage for FieldVision data.

Per-game directory layout:
    data/<gamePk>/
        actor_frames.parquet
        bat_frames.parquet
        ball_frames.parquet
        pitch_events.parquet
        pitch_labels.parquet
        players.parquet
        labels.parquet
        bones.parquet
        meta.parquet

Why Parquet instead of SQLite:
  - On CIFS, SQLite was bottlenecked at ~0.5-1 segment/sec because every
    transaction commit waited for the NAS to acknowledge fsync. Parquet
    writes the whole table in one shot (one fsync), so ~50× faster on
    the same hardware.
  - Columnar compression (zstd) shrinks the float-heavy actor_frame data
    1.4-2× vs raw SQLite.
  - DuckDB queries Parquet files directly — `SELECT ... FROM
    read_parquet('data/*/actor_frames.parquet')` — so analysis code
    needs only a near-trivial change from sqlite3.connect to
    duckdb.connect.

Usage pattern (history scrape, one game per call):

    from fieldvision.storage_parquet import ParquetGameStore

    store = ParquetGameStore(game_pk=823141, data_dir=Path("data"))
    store.write_lookups(metadata_path, labels_path)
    for seg in segments:
        rows = decode_segment(seg)
        store.add_actor_frames(rows["actors"])
        store.add_bat_frames(rows["bats"])
        # …
    store.finalize()                  # writes all buffered tables to disk

Usage pattern (live daemon, segments dribble in over time):

    store = ParquetGameStore(game_pk=..., data_dir=...)
    # each poll:
    rows = decode_segment(new_segment)
    store.add_actor_frames(rows["actors"])
    store.flush_if_full()             # writes a row group when buffer > threshold
    # at game end (or on shutdown):
    store.finalize()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from fieldvision.storage import JOINT_COLS


# Schemas
def _actor_frame_schema() -> pa.Schema:
    fields = [
        ("game_pk", pa.int64()),
        ("segment_idx", pa.int64()),
        ("frame_num", pa.int64()),
        ("actor_uid", pa.int64()),
        ("mlb_player_id", pa.int64()),
        ("actor_type", pa.string()),
        ("time_unix", pa.float64()),
        ("timestamp", pa.string()),
        ("is_gap", pa.int8()),
        ("scale", pa.float32()),
        ("ground", pa.float32()),
        ("apex", pa.float32()),
    ]
    for _, name in JOINT_COLS:
        fields += [
            (f"{name}_x",  pa.float32()),
            (f"{name}_y",  pa.float32()),
            (f"{name}_z",  pa.float32()),
            (f"{name}_qx", pa.float32()),
            (f"{name}_qy", pa.float32()),
            (f"{name}_qz", pa.float32()),
            (f"{name}_qw", pa.float32()),
        ]
    fields += [
        ("bat_handle_x", pa.float32()),
        ("bat_handle_y", pa.float32()),
        ("bat_handle_z", pa.float32()),
    ]
    return pa.schema(fields)


def _bat_frame_schema() -> pa.Schema:
    return pa.schema([
        ("game_pk", pa.int64()),
        ("segment_idx", pa.int64()),
        ("frame_num", pa.int64()),
        ("time_unix", pa.float64()),
        ("head_x", pa.float32()), ("head_y", pa.float32()), ("head_z", pa.float32()),
        ("handle_x", pa.float32()), ("handle_y", pa.float32()), ("handle_z", pa.float32()),
    ])


def _ball_frame_schema() -> pa.Schema:
    return pa.schema([
        ("game_pk", pa.int64()),
        ("segment_idx", pa.int64()),
        ("frame_num", pa.int64()),
        ("time_unix", pa.float64()),
        ("ball_x", pa.float32()), ("ball_y", pa.float32()), ("ball_z", pa.float32()),
    ])


def _pitch_event_schema() -> pa.Schema:
    return pa.schema([
        ("game_pk", pa.int64()),
        ("segment_idx", pa.int64()),
        ("frame_num", pa.int64()),
        ("time_unix", pa.float64()),
        ("event_type", pa.string()),
        ("play_id", pa.string()),
        ("pos_x", pa.float32()),
        ("pos_y", pa.float32()),
        ("pos_z", pa.float32()),
    ])


def _labels_schema() -> pa.Schema:
    return pa.schema([
        ("actor_uid", pa.int64()),
        ("actor", pa.int64()),
        ("actor_type", pa.string()),
    ])


def _bones_schema() -> pa.Schema:
    return pa.schema([
        ("bone_id", pa.int64()),
        ("name", pa.string()),
    ])


def _players_schema() -> pa.Schema:
    return pa.schema([
        ("mlb_player_id", pa.int64()),
        ("jersey_number", pa.string()),
        ("role_id", pa.int64()),
        ("team", pa.string()),
        ("position_abbr", pa.string()),
        ("parent_team_id", pa.int64()),
    ])


def _meta_schema() -> pa.Schema:
    return pa.schema([
        ("key", pa.string()),
        ("value", pa.string()),
    ])


SCHEMAS = {
    "actor_frames":  _actor_frame_schema(),
    "bat_frames":    _bat_frame_schema(),
    "ball_frames":   _ball_frame_schema(),
    "pitch_events":  _pitch_event_schema(),
    "labels":        _labels_schema(),
    "bones":         _bones_schema(),
    "players":       _players_schema(),
    "meta":          _meta_schema(),
}


def _actor_frame_col_names() -> list[str]:
    """Column names matching the order produced by _build_actor_frame_row."""
    cols = [
        "game_pk", "segment_idx", "frame_num", "actor_uid",
        "mlb_player_id", "actor_type", "time_unix", "timestamp",
        "is_gap", "scale", "ground", "apex",
    ]
    for _, name in JOINT_COLS:
        cols += [f"{name}_x", f"{name}_y", f"{name}_z",
                 f"{name}_qx", f"{name}_qy", f"{name}_qz", f"{name}_qw"]
    cols += ["bat_handle_x", "bat_handle_y", "bat_handle_z"]
    return cols


ACTOR_FRAME_COLS = _actor_frame_col_names()


class ParquetGameStore:
    """Per-game Parquet writer.

    Buffers rows in memory; flushes when buffer exceeds threshold OR on
    finalize(). Each flush appends a Parquet row group to the on-disk file
    via pyarrow's ParquetWriter (no read-modify-write).
    """

    def __init__(self, game_pk: int, data_dir: Path,
                 row_group_size: int = 100_000,
                 compression: str = "zstd",
                 append_suffix: str | None = None):
        """
        append_suffix: when set, time-series tables (actor_frames, bat_frames,
            ball_frames, pitch_events) write to `<table>-<suffix>.parquet`
            instead of `<table>.parquet`. Use this for the live daemon, where
            every poll cycle creates a fresh store — otherwise pyarrow's
            ParquetWriter truncates the single file each poll, silently
            destroying prior data. Lookup tables (players/labels/bones/meta)
            always use the canonical name since they're idempotent rewrites.
        """
        self.game_pk = int(game_pk)
        self.dir = Path(data_dir) / str(self.game_pk)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.row_group_size = row_group_size
        self.compression = compression
        self.append_suffix = append_suffix
        self._buffers: dict[str, list[dict]] = {k: [] for k in SCHEMAS}
        self._writers: dict[str, pq.ParquetWriter] = {}

    # Tables that get the per-instance suffix when append_suffix is set.
    # Lookup tables don't — they're small + idempotent.
    _APPEND_TABLES = ("actor_frames", "bat_frames", "ball_frames", "pitch_events")

    def _path_for(self, table: str) -> Path:
        if self.append_suffix and table in self._APPEND_TABLES:
            return self.dir / f"{table}-{self.append_suffix}.parquet"
        return self.dir / f"{table}.parquet"

    # ─────────────────────────────────────────────
    # row buffering
    # ─────────────────────────────────────────────

    def add_actor_frames(self, rows: list[tuple]) -> None:
        """Rows are tuples in the order produced by _build_actor_frame_row."""
        if not rows: return
        cols = ACTOR_FRAME_COLS
        for row in rows:
            self._buffers["actor_frames"].append(dict(zip(cols, row)))
        if len(self._buffers["actor_frames"]) >= self.row_group_size:
            self._flush_table("actor_frames")

    def add_dict_rows(self, table: str, rows: list[dict]) -> None:
        if not rows: return
        self._buffers[table].extend(rows)
        if len(self._buffers[table]) >= self.row_group_size:
            self._flush_table(table)

    def add_bat_frames(self, rows: list[dict]) -> None:
        self.add_dict_rows("bat_frames", rows)

    def add_ball_frames(self, rows: list[dict]) -> None:
        self.add_dict_rows("ball_frames", rows)

    def add_pitch_events(self, rows: list[dict]) -> None:
        self.add_dict_rows("pitch_events", rows)

    # ─────────────────────────────────────────────
    # lookup tables (one-shot, small)
    # ─────────────────────────────────────────────

    def write_lookups_from_metadata(self, metadata: dict, labels: dict) -> dict[int, dict]:
        """Mirrors storage.load_lookup_tables(): writes bones, players, labels,
        meta. Returns the labels dict for downstream use."""
        # Bones
        bone_id_map = metadata.get("boneIdMap", {})
        bat_id_map = metadata.get("batBoneIdMap", {})
        bones_rows = [{"bone_id": int(k), "name": v}
                      for k, v in {**bone_id_map, **bat_id_map}.items()]

        # Players
        players_rows = []
        for team_key, team_data in metadata.get("boxscore", {}).get("teams", {}).items():
            for _, p in team_data.get("players", {}).items():
                pid = p.get("person", {}).get("id")
                if pid is None: continue
                players_rows.append({
                    "mlb_player_id": int(pid),
                    "jersey_number": p.get("jerseyNumber"),
                    "role_id": None,
                    "team": team_key,
                    "position_abbr": (p.get("position", {}) or {}).get("abbreviation"),
                    "parent_team_id": p.get("parentTeamId"),
                })

        # Labels
        labels_rows = []
        labels_dict: dict[int, dict] = {}
        for k, v in labels.items():
            try: uid = int(k)
            except ValueError: continue
            actor = v.get("actor")
            atype = v.get("type")
            labels_rows.append({"actor_uid": uid, "actor": actor, "actor_type": atype})
            labels_dict[uid] = {"actor": actor, "type": atype}

        # Meta
        meta_rows = []
        for k in ("version", "gamePk", "venueId"):
            if k in metadata:
                meta_rows.append({"key": k, "value": json.dumps(metadata[k])})
        for setting in metadata.get("ruleSettings", []):
            meta_rows.append({"key": f"rule.{setting.get('settingName')}",
                              "value": json.dumps(setting.get("settingValue"))})

        self._write_table_one_shot("bones", bones_rows)
        self._write_table_one_shot("players", players_rows)
        self._write_table_one_shot("labels", labels_rows)
        self._write_table_one_shot("meta", meta_rows)
        return labels_dict

    # ─────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────

    def _flush_table(self, table: str) -> None:
        rows = self._buffers[table]
        if not rows: return
        schema = SCHEMAS[table]
        arrays = []
        for field in schema:
            arrays.append(pa.array([row.get(field.name) for row in rows], type=field.type))
        record_batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
        if table not in self._writers:
            path = self._path_for(table)
            self._writers[table] = pq.ParquetWriter(str(path), schema,
                                                    compression=self.compression)
        self._writers[table].write_table(pa.Table.from_batches([record_batch], schema=schema))
        self._buffers[table] = []

    def _write_table_one_shot(self, table: str, rows: list[dict]) -> None:
        """Overwrites the Parquet file for a small table — used for lookups."""
        # Lookup tables always use the canonical (no-suffix) path.
        path = self.dir / f"{table}.parquet"
        schema = SCHEMAS[table]
        if rows:
            arrays = [pa.array([r.get(f.name) for r in rows], type=f.type) for f in schema]
            tbl = pa.Table.from_arrays(arrays, schema=schema)
        else:
            tbl = pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)
        pq.write_table(tbl, str(path), compression=self.compression)

    def finalize(self) -> None:
        """Flush remaining buffers and close writers."""
        for table in list(self._buffers.keys()):
            self._flush_table(table)
        for w in self._writers.values():
            w.close()
        self._writers.clear()


# ────────────────────────────────────────────────────────────────────────
# Segment ingestion — Parquet equivalent of storage.ingest_segment
# ────────────────────────────────────────────────────────────────────────

# Column names for the three smaller per-frame tables, matching the order
# in which storage.ingest_segment builds tuples.
BAT_FRAME_COLS = [
    "game_pk", "segment_idx", "frame_num", "time_unix",
    "head_x", "head_y", "head_z",
    "handle_x", "handle_y", "handle_z",
]
BALL_FRAME_COLS = [
    "game_pk", "segment_idx", "frame_num", "time_unix",
    "ball_x", "ball_y", "ball_z",
]
PITCH_EVENT_COLS = [
    "game_pk", "segment_idx", "frame_num", "time_unix",
    "event_type", "play_id", "pos_x", "pos_y", "pos_z",
]


def ingest_segment_parquet(
    store: "ParquetGameStore",
    game_pk: int,
    segment_idx: int,
    bin_path: Path,
    labels_dict: dict[int, dict],
) -> tuple[int, int]:
    """Decode one .bin segment and add all rows to the Parquet store.
    Mirrors storage.ingest_segment but writes to Parquet instead of SQLite.
    Returns (n_actor_rows, n_ball_rows)."""
    # Imports kept local to avoid circular imports + cheap on every call
    from fieldvision.skeleton import forward_kinematics
    from fieldvision.storage import _build_actor_frame_row
    from fieldvision.wire_schemas import read_tracking_data, unpack_smallest_three

    td = read_tracking_data(bin_path.read_bytes())
    actor_rows: list[tuple] = []
    ball_rows: list[tuple] = []
    bat_rows: list[tuple] = []
    pitch_event_rows: list[tuple] = []

    for f in td.frames:
        is_gap = f.isGap
        time_unix = f.time
        ts = f.timestamp

        if f.ballPosition is not None:
            ball_rows.append((
                game_pk, segment_idx, f.num, time_unix,
                f.ballPosition.x, f.ballPosition.y, f.ballPosition.z,
            ))

        if (f.inferredBat is not None and f.inferredBat.headPosition is not None
                and f.inferredBat.handlePosition is not None):
            head = f.inferredBat.headPosition
            handle = f.inferredBat.handlePosition
            bat_rows.append((
                game_pk, segment_idx, f.num, time_unix,
                head.x, head.y, head.z, handle.x, handle.y, handle.z,
            ))

        for ge in f.gameEvents:
            if ge.dataType == 7 and ge.playId:
                pitch_event_rows.append((
                    game_pk, segment_idx, f.num, time_unix,
                    "PLAY_EVENT", ge.playId, None, None, None,
                ))
        for te in f.trackedEvents:
            if te.eventType:
                pitch_event_rows.append((
                    game_pk, segment_idx, f.num, time_unix,
                    te.eventType, None, te.x, te.y, te.z,
                ))

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
            world_rot = dict(ws.bone_world_rot) if ws.bone_world_rot else None
            label_info = labels_dict.get(a.uid, {})
            mlb_id = label_info.get("actor")
            atype = label_info.get("type")
            bat = ((a.batRootPos.x, a.batRootPos.y, a.batRootPos.z)
                   if a.batRootPos else None)
            actor_rows.append(_build_actor_frame_row(
                game_pk, segment_idx, f.num, time_unix, ts, is_gap,
                a.uid, atype, mlb_id, a.scale, a.ground, a.apex,
                world_pos, bat, world_rot,
            ))

    # Push to Parquet store
    if actor_rows:
        store.add_actor_frames(actor_rows)
    if ball_rows:
        store.add_dict_rows("ball_frames",
                             [dict(zip(BALL_FRAME_COLS, r)) for r in ball_rows])
    if bat_rows:
        store.add_dict_rows("bat_frames",
                             [dict(zip(BAT_FRAME_COLS, r)) for r in bat_rows])
    if pitch_event_rows:
        store.add_dict_rows("pitch_events",
                             [dict(zip(PITCH_EVENT_COLS, r)) for r in pitch_event_rows])
    return len(actor_rows), len(ball_rows)


def actor_frame_parquet_paths(data_dir: Path, game_pk: int) -> list[Path]:
    """All actor_frames*.parquet files for a game, in lexicographic order.

    Returns the canonical single file (`actor_frames.parquet`) plus any
    per-poll appended files (`actor_frames-<suffix>.parquet`) the live
    daemon writes. Filters out empty / footer-less files so callers see
    only finalized data."""
    gdir = Path(data_dir) / str(game_pk)
    if not gdir.is_dir():
        return []
    out = []
    for p in sorted(gdir.glob("actor_frames*.parquet")):
        if p.stat().st_size < 8:
            continue
        try:
            with p.open("rb") as f:
                f.seek(-4, 2)
                if f.read(4) != b"PAR1":
                    continue
        except OSError:
            continue
        out.append(p)
    return out


def max_segment_idx_for_game(data_dir: Path, game_pk: int) -> int:
    """Return the highest segment_idx already ingested for a game (for resume),
    or -1 if the game has no Parquet data yet. Reads only the actor_frames
    Parquet column metadata — no full scan."""
    paths = actor_frame_parquet_paths(data_dir, game_pk)
    if not paths:
        return -1
    try:
        # Use DuckDB so we can do a max() without loading any data
        import duckdb
        con = duckdb.connect()
        files_sql = "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"
        max_seg = con.execute(
            f"SELECT MAX(segment_idx) FROM read_parquet({files_sql})"
        ).fetchone()[0]
        con.close()
        return int(max_seg) if max_seg is not None else -1
    except Exception:
        return -1
