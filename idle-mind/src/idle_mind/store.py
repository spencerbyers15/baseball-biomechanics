"""SQLite state store. One connection, async via aiosqlite.

Timestamps are unix epoch seconds (floats) from the Clock, so simulated time
lands in the same axis as real time; rows written under a simulated offset
carry ``simulated=1`` where it matters.

The ``memories`` and ``percepts`` tables are created now so the schema is
stable, but only basic memory CRUD is implemented — consolidation, evocation
and forgetting are Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idle_thoughts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    content TEXT NOT NULL,
    tick_n INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    simulated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS digests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    covering_from_ts REAL NOT NULL,
    covering_to_ts REAL NOT NULL,
    content_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clock_marks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    simulated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS summaries(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    upto_turn_id INTEGER NOT NULL,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts REAL NOT NULL,
    last_evoked_ts REAL,
    evoke_count INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    source_type TEXT NOT NULL,
    embedding BLOB,
    salience REAL NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS percepts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    modality TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS usage_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    day TEXT NOT NULL,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS turn_metrics(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    turn_id INTEGER,
    digest_bytes INTEGER,
    n_digest_items INTEGER,
    n_surfaced INTEGER,
    surfaced_ids TEXT,
    latency_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_thoughts_ts ON idle_thoughts(ts);
CREATE INDEX IF NOT EXISTS idx_marks_event ON clock_marks(event, ts);
CREATE INDEX IF NOT EXISTS idx_usage_day ON usage_log(day);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "store not opened"
        return self._db

    async def _exec(self, sql: str, params: tuple = ()) -> int:
        cur = await self.db.execute(sql, params)
        await self.db.commit()
        return cur.lastrowid or 0

    # -- turns ------------------------------------------------------------

    async def add_turn(self, ts: float, role: str, content: str) -> int:
        return await self._exec(
            "INSERT INTO turns(ts, role, content) VALUES (?,?,?)", (ts, role, content)
        )

    async def recent_turns(self, n: int) -> list[aiosqlite.Row]:
        cur = await self.db.execute(
            "SELECT * FROM turns ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = await cur.fetchall()
        return list(reversed(rows))

    async def turn_count(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) AS c FROM turns")
        row = await cur.fetchone()
        return int(row["c"])

    async def turns_before(self, turn_id: int) -> list[aiosqlite.Row]:
        cur = await self.db.execute(
            "SELECT * FROM turns WHERE id <= ? ORDER BY id", (turn_id,)
        )
        return list(await cur.fetchall())

    # -- idle thoughts ----------------------------------------------------

    async def add_idle_thought(
        self, ts: float, content: str, tick_n: int, session_id: str, simulated: bool
    ) -> int:
        return await self._exec(
            "INSERT INTO idle_thoughts(ts, content, tick_n, session_id, simulated)"
            " VALUES (?,?,?,?,?)",
            (ts, content, tick_n, session_id, int(simulated)),
        )

    async def thoughts_since(self, ts: float) -> list[aiosqlite.Row]:
        cur = await self.db.execute(
            "SELECT * FROM idle_thoughts WHERE ts > ? ORDER BY id", (ts,)
        )
        return list(await cur.fetchall())

    async def last_thoughts(self, k: int) -> list[aiosqlite.Row]:
        cur = await self.db.execute(
            "SELECT * FROM idle_thoughts ORDER BY id DESC LIMIT ?", (k,)
        )
        rows = await cur.fetchall()
        return list(reversed(rows))

    # -- digests ----------------------------------------------------------

    async def add_digest(
        self, ts: float, covering_from_ts: float, covering_to_ts: float, content: dict
    ) -> int:
        return await self._exec(
            "INSERT INTO digests(ts, covering_from_ts, covering_to_ts, content_json)"
            " VALUES (?,?,?,?)",
            (ts, covering_from_ts, covering_to_ts, json.dumps(content)),
        )

    async def last_digest(self) -> aiosqlite.Row | None:
        cur = await self.db.execute("SELECT * FROM digests ORDER BY id DESC LIMIT 1")
        return await cur.fetchone()

    # -- clock marks ------------------------------------------------------

    async def add_clock_mark(self, ts: float, event: str, simulated: bool = False) -> int:
        return await self._exec(
            "INSERT INTO clock_marks(ts, event, simulated) VALUES (?,?,?)",
            (ts, event, int(simulated)),
        )

    async def recent_marks(self, n: int) -> list[aiosqlite.Row]:
        cur = await self.db.execute(
            "SELECT * FROM clock_marks ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = await cur.fetchall()
        return list(reversed(rows))

    async def latest_mark_ts(self, event: str) -> float | None:
        cur = await self.db.execute(
            "SELECT ts FROM clock_marks WHERE event = ? ORDER BY id DESC LIMIT 1",
            (event,),
        )
        row = await cur.fetchone()
        return float(row["ts"]) if row else None

    # -- transcript summaries --------------------------------------------

    async def add_summary(self, ts: float, upto_turn_id: int, content: str) -> int:
        return await self._exec(
            "INSERT INTO summaries(ts, upto_turn_id, content) VALUES (?,?,?)",
            (ts, upto_turn_id, content),
        )

    async def latest_summary(self) -> aiosqlite.Row | None:
        cur = await self.db.execute("SELECT * FROM summaries ORDER BY id DESC LIMIT 1")
        return await cur.fetchone()

    # -- token usage / budget --------------------------------------------

    async def add_usage(
        self, ts: float, day: str, role: str, model: str,
        input_tokens: int, output_tokens: int,
    ) -> int:
        return await self._exec(
            "INSERT INTO usage_log(ts, day, role, model, input_tokens, output_tokens)"
            " VALUES (?,?,?,?,?,?)",
            (ts, day, role, model, input_tokens, output_tokens),
        )

    async def tokens_used_on(self, day: str) -> int:
        cur = await self.db.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS total"
            " FROM usage_log WHERE day = ?",
            (day,),
        )
        row = await cur.fetchone()
        return int(row["total"])

    # -- per-turn metrics -------------------------------------------------

    async def add_turn_metrics(
        self, ts: float, turn_id: int | None, digest_bytes: int | None,
        n_digest_items: int | None, n_surfaced: int | None,
        surfaced_ids: list[str] | None, latency_ms: float,
    ) -> int:
        return await self._exec(
            "INSERT INTO turn_metrics(ts, turn_id, digest_bytes, n_digest_items,"
            " n_surfaced, surfaced_ids, latency_ms) VALUES (?,?,?,?,?,?,?)",
            (
                ts, turn_id, digest_bytes, n_digest_items, n_surfaced,
                json.dumps(surfaced_ids) if surfaced_ids is not None else None,
                latency_ms,
            ),
        )

    # -- memories (Phase 2 data layer; logic comes later) -----------------

    async def add_memory(
        self, created_ts: float, content: str, source_type: str,
        salience: float, embedding: bytes | None = None,
    ) -> int:
        return await self._exec(
            "INSERT INTO memories(created_ts, content, source_type, salience, embedding)"
            " VALUES (?,?,?,?,?)",
            (created_ts, content, source_type, salience, embedding),
        )

    async def get_memory(self, memory_id: int) -> aiosqlite.Row | None:
        cur = await self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return await cur.fetchone()

    async def memories(self, include_archived: bool = False) -> list[aiosqlite.Row]:
        sql = "SELECT * FROM memories"
        if not include_archived:
            sql += " WHERE archived = 0"
        cur = await self.db.execute(sql + " ORDER BY id")
        return list(await cur.fetchall())

    async def record_evocation(self, memory_id: int, ts: float) -> None:
        await self._exec(
            "UPDATE memories SET evoke_count = evoke_count + 1, last_evoked_ts = ?"
            " WHERE id = ?",
            (ts, memory_id),
        )

    async def set_salience(self, memory_id: int, salience: float) -> None:
        await self._exec(
            "UPDATE memories SET salience = ? WHERE id = ?", (salience, memory_id)
        )

    async def archive_memory(self, memory_id: int) -> None:
        await self._exec("UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,))


def row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}
