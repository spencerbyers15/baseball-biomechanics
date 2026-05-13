"""DuckDB shim that lets old SQL-against-SQLite queries run against the
per-game Parquet layout without touching the SQL.

For each game, the storage tree is:

    <data_dir>/<game_pk>/{actor_frames, bat_frames, ball_frames,
                          pitch_events, pitch_labels, players,
                          labels, bones, meta}.parquet

This module returns a DuckDB connection where each Parquet file is
exposed as a view named the same as the legacy SQLite table (singular,
no `s`):

    actor_frame, bat_frame, ball_frame, pitch_event, pitch_label,
    players, labels, bones, meta

So `conn.execute("SELECT * FROM actor_frame WHERE ...").fetchall()`
works on the new Parquet layout. DuckDB uses the same `?` placeholder
style and `cursor()`/`fetchall()`/`fetchone()` API as `sqlite3`, so
most existing scripts can swap `sqlite3.connect(...)` for
`open_game(game_pk, data_dir)` with no other changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import duckdb


# legacy SQLite table name -> Parquet file name (without .parquet)
_TABLE_TO_FILE = {
    "actor_frame":  "actor_frames",
    "bat_frame":    "bat_frames",
    "ball_frame":   "ball_frames",
    "pitch_event":  "pitch_events",
    "pitch_label":  "pitch_labels",
    "players":      "players",
    "labels":       "labels",
    "bones":        "bones",
    "meta":         "meta",
}


def _default_data_dir() -> Path:
    """Honor the env var first, fall back to a repo-relative `data/`."""
    env = os.environ.get("FV_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def game_dir(game_pk: int, data_dir: Path | None = None) -> Path:
    return (data_dir or _default_data_dir()) / str(game_pk)


def _parquet_finalized(p: Path) -> bool:
    """Cheap check that a Parquet file has its footer (trailing PAR1 magic).
    In-flight files written by ParquetWriter only get the footer at close()."""
    try:
        if p.stat().st_size < 8:
            return False
        with p.open("rb") as f:
            f.seek(-4, 2)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def _finalized_parquets_for_table(gdir: Path, fname: str) -> list[Path]:
    """Return every finalized parquet for one logical table.

    Time-series tables (actor_frames, bat_frames, ball_frames, pitch_events)
    may be split across the canonical `<fname>.parquet` plus per-poll files
    like `<fname>-<suffix>.parquet` that the live daemon writes. Lookup
    tables only ever have the single canonical file.
    """
    candidates = [gdir / f"{fname}.parquet"]
    # Glob -suffixed appended files (only relevant for time-series tables;
    # for lookup tables this glob just returns nothing extra).
    candidates += sorted(gdir.glob(f"{fname}-*.parquet"))
    return [p for p in candidates if p.exists() and _parquet_finalized(p)]


def open_game(game_pk: int, data_dir: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with views for every Parquet file the
    game has on disk. Missing tables (e.g. a game without `pitch_label`
    yet) are silently skipped — querying them will raise the same
    'table does not exist' error a fresh SQLite would.

    Files that exist but aren't finalized (no Parquet footer — the daemon
    is still appending) are also skipped silently so a backfill in
    flight doesn't poison every reader.

    Per-poll appended files (`actor_frames-<suffix>.parquet`) are unioned
    into the same view alongside `actor_frames.parquet`.
    """
    gdir = game_dir(game_pk, data_dir)
    if not gdir.exists():
        raise FileNotFoundError(f"no game dir for {game_pk}: {gdir}")
    con = duckdb.connect()
    for table, fname in _TABLE_TO_FILE.items():
        paths = _finalized_parquets_for_table(gdir, fname)
        if not paths:
            continue
        files_sql = "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"
        con.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT * FROM read_parquet({files_sql}, union_by_name=true)"
        )
    return con


def open_games(game_pks: list[int], data_dir: Path | None = None,
               ) -> duckdb.DuckDBPyConnection:
    """Multi-game connection. Each table becomes a UNION ALL across all
    games that have it. Use this for cross-game queries; per-game
    `open_game` is faster when you only need one."""
    dd = data_dir or _default_data_dir()
    con = duckdb.connect()
    for table, fname in _TABLE_TO_FILE.items():
        parts = [
            (dd / str(pk) / f"{fname}.parquet").as_posix()
            for pk in game_pks
            if (dd / str(pk) / f"{fname}.parquet").exists()
        ]
        if not parts:
            continue
        # DuckDB lets you pass a list literal to read_parquet directly.
        files_sql = "[" + ", ".join(f"'{p}'" for p in parts) + "]"
        con.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT * FROM read_parquet({files_sql}, union_by_name=true)"
        )
    return con


def list_games(data_dir: Path | None = None) -> list[int]:
    """All gamePks present on disk (i.e. all numeric subdirs of data_dir
    that contain at least one actor_frames*.parquet)."""
    dd = data_dir or _default_data_dir()
    if not dd.exists():
        return []
    out = []
    for p in dd.iterdir():
        if p.is_dir() and p.name.isdigit() and any(p.glob("actor_frames*.parquet")):
            out.append(int(p.name))
    return sorted(out)


def iter_games(data_dir: Path | None = None
               ) -> Iterator[tuple[int, duckdb.DuckDBPyConnection]]:
    """Yield (game_pk, conn) for each game on disk. Caller is responsible
    for closing the connection (or just letting it be garbage-collected)."""
    for pk in list_games(data_dir):
        yield pk, open_game(pk, data_dir)
