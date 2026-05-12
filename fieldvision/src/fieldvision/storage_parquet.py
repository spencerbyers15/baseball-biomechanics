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
                 compression: str = "zstd"):
        self.game_pk = int(game_pk)
        self.dir = Path(data_dir) / str(self.game_pk)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.row_group_size = row_group_size
        self.compression = compression
        self._buffers: dict[str, list[dict]] = {k: [] for k in SCHEMAS}
        self._writers: dict[str, pq.ParquetWriter] = {}

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
            path = self.dir / f"{table}.parquet"
            self._writers[table] = pq.ParquetWriter(str(path), schema,
                                                    compression=self.compression)
        self._writers[table].write_table(pa.Table.from_batches([record_batch], schema=schema))
        self._buffers[table] = []

    def _write_table_one_shot(self, table: str, rows: list[dict]) -> None:
        """Overwrites the Parquet file for a small table — used for lookups."""
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
