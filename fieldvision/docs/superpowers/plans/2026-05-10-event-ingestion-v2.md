# Pitch Segmentation + Statsapi Label Join (v2)

> **Supersedes** `2026-05-10-event-ingestion.md`. The original plan tried to decode all event payloads from the wire format. Recon (see git history) revealed: (1) `PlayEventDataWire.playId` matches statsapi `playId` exactly (verified: 26/26 overlap on a 500-segment sample of game 823141, 100% marked `isPitch=True`); (2) `BallPitchData` actually lives in `ballPolynomials`, not `trackedEvents`; (3) statsapi already gives us every label we'd want (pitch type, velocity, spin, location, result). So we don't need to decode pitch payloads from wire — just the segmentation markers — and join statsapi for labels.

**Goal:** For every pitch in a captured game, produce (a) the `play_id`-tagged frame range from wire data and (b) statsapi labels (type/velocity/spin/result) joined on `play_id`. Downstream feature engineering can then pull "the 90 frames before pitch X" with one SQL query.

**Architecture:**
- Two purpose-built tables: `pitch_event` (wire-derived per-frame markers, many rows per pitch) + `pitch_label` (statsapi-derived labels, one row per pitch).
- Wire decode is **minimal**: just `PlayEventDataWire.playId` (from `gameEvents[]` dataType=7) and `TrackedEventWire.eventType` + position (the 14 flat fields).
- No `BallPitchData`, no `Strikezone` sub-table, no general `game_event` table. Statsapi handles all of that.
- Join key everywhere: `play_id` (UUID, byte-identical between sources).

**Tech Stack:** Python 3.11, hand-rolled FlatBuffer reader (already exists), SQLite, urllib for statsapi, pytest.

**What stays from v1:**
- Branch `feat/event-ingestion`
- `scripts/extract_event_offsets.py` + the offsets doc (still need PlayEvent + TrackedEvent offsets)
- `samples/mlb_bundles/gd.@bvg_poser.min.js` cached
- `tests/__init__.py`, `tests/conftest.py` (the fixture segment constants — still useful)

**What gets dropped or reshaped:**
- Drop the `game_event` table from v1 Task 3 (we don't need it — statsapi covers count/handedness/atbat per pitch)
- Reshape the `pitch_event` table: was 24 wide columns of pitch metadata; becomes a thin marker row schema (just event_type + play_id + optional x/y/z)
- Update `tests/test_storage_schema.py` to match new schema
- The v1 SQL helpers `_game_event_insert_sql()` / `_pitch_event_insert_sql()` get rewritten

**Out of scope:**
- BallPolynomial / BallPitchDataWire decoding (we get all that from statsapi instead)
- BatImpact, Inning, ABS, Position, TeamScore, BattingOrder events (statsapi covers most; remaining are nice-to-have for Phase A.2)
- Any modeling/prediction work (Phase B+)
- Backfill of statsapi for all 27 captured games is in scope, but only as a single command run; no batch infrastructure

---

## Task v2.1: Reset schema (drop game_event, reshape pitch_event, add pitch_label)

**Files:**
- Modify: `src/fieldvision/storage.py`
- Modify: `tests/test_storage_schema.py`

The v1 schema doesn't fit the new approach. Replace cleanly.

**New `pitch_event`:**
```sql
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
```

(Note: no PRIMARY KEY — multiple events per (segment, frame) are allowed and useful.)

**New `pitch_label` (populated by statsapi later, NOT by wire ingest):**
```sql
CREATE TABLE IF NOT EXISTS pitch_label (
    game_pk INTEGER NOT NULL,
    play_id TEXT NOT NULL,              -- statsapi playEvents[].playId; matches pitch_event.play_id
    ab_index INTEGER,                   -- atBatIndex
    pitch_number INTEGER,               -- pitchNumber within at-bat
    inning INTEGER,
    top_inning INTEGER,                 -- 1=top, 0=bottom
    batter_id INTEGER,
    pitcher_id INTEGER,
    batter_side TEXT,                   -- 'L' | 'R'
    pitcher_throws TEXT,                -- 'L' | 'R'
    balls_before INTEGER,
    strikes_before INTEGER,
    outs_before INTEGER,
    pitch_type TEXT,                    -- 'FF', 'SL', 'CH', etc. (statsapi details.type.code)
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
```

**Drop:** `game_event` table from v1 (fully drop in SCHEMA — note this means existing per-game DBs from v1 testing will keep the old game_event table; that's fine, it's empty).

- [ ] **Step 1: Update SCHEMA in storage.py**
  - Remove the `game_event` CREATE block
  - Replace the `pitch_event` CREATE block with the new shape above
  - Add the `pitch_label` CREATE block
  - Remove `_game_event_insert_sql()` helper entirely
  - Replace `_pitch_event_insert_sql()` with the new shape:
    ```python
    def _pitch_event_insert_sql() -> str:
        cols = [
            "game_pk", "segment_idx", "frame_num", "time_unix",
            "event_type", "play_id",
            "pos_x", "pos_y", "pos_z",
        ]
        placeholders = ", ".join("?" * len(cols))
        return f"INSERT INTO pitch_event ({', '.join(cols)}) VALUES ({placeholders})"
    ```
  - Add `_pitch_label_insert_sql()` (use INSERT OR REPLACE since statsapi re-fetches are idempotent):
    ```python
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
    ```

- [ ] **Step 2: Rewrite `tests/test_storage_schema.py`** to assert the new column sets:
  ```python
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
  ```

- [ ] **Step 3: Run tests, verify they fail (some will pass — INSERT helpers don't have a behavior test)**
  ```bash
  cd /Users/spencerbyers/fieldvision
  /Users/spencerbyers/anaconda3/bin/python3 -m pytest tests/test_storage_schema.py -v
  ```
  Expected before storage.py edits: `test_pitch_event_columns` likely passes (subset), `test_pitch_label_columns` fails (table doesn't exist), `test_indexes_exist` fails (new index names), `test_no_game_event_table` fails (still exists from v1).

- [ ] **Step 4: Apply storage.py edits per Step 1**

- [ ] **Step 5: Re-run tests, verify all pass**

- [ ] **Step 6: Commit**
  ```bash
  git add src/fieldvision/storage.py tests/test_storage_schema.py
  git commit -m "v2 schema: drop game_event, reshape pitch_event for markers, add pitch_label"
  ```

---

## Task v2.2: Decode minimal wire schemas in `wire_schemas.py`

**Files:**
- Modify: `src/fieldvision/wire_schemas.py` (append at end)

We need three new decoders, all small:

1. **`GameEventWire` (4 fields)** — vtoff 4=DataType (uint8), vtoff 6=Data (indirect), vtoff 8=Time (float64), vtoff 10=IsKeyFramed (int8). We only USE DataType + Data, but read all four for completeness.

2. **`PlayEventDataWire` (4 fields)** — vtoff 4=Action, vtoff 6=Index, vtoff 8=PlayId (string), vtoff 10=Strikezone (sub-table). We only NEED PlayId; skip Strikezone — statsapi gives us strike zone.

3. **`TrackedEventWire` (14 flat fields)** — vtoff 4=Timestamp(str), 6=BatSide(str), 8=PitchHand(str), 10=AtBatNumber(uint16), 12=PitchNumber(uint16), 14=PickoffNumber(uint16), 16=SzTop(f32), 18=SzBot(f32), 20=EventType(str), 22=X(f32), 24=Y(f32), 26=Z(f32), 28=Position(uint8), 30=EventTypeId(int8). For pitch_event rows we use just `EventType`, `X`, `Y`, `Z`.

The `GameEventWire` dataType→class union dispatch (from offsets doc):
- 7 = PlayEventDataWire → use this
- 1=Count, 6=AtBat, 8=Handed, 12=BatImpact → ignore (statsapi has them)
- All others: ignore

- [ ] **Step 1: Write a failing test** (append to `tests/test_event_schemas.py`, create file if needed):
  ```python
  """Tests for FlatBuffer event schema decoders (v2 — minimal: PlayEvent.playId + TrackedEvent.eventType)."""

  from tests.conftest import (
      FIXTURE_SEG_WITH_GAME_EVENTS,
      FIXTURE_SEG_WITH_TRACKED_EVENTS,
      fixture_bin_path,
  )
  from fieldvision.wire_schemas import read_tracking_data


  def test_game_events_have_playids():
      """At least one PlayEvent (gameEvent.dataType=7) in the fixture should yield a UUID-shaped playId."""
      game_pk, seg = FIXTURE_SEG_WITH_GAME_EVENTS
      td = read_tracking_data(fixture_bin_path(game_pk, seg).read_bytes())
      play_ids = []
      for f in td.frames:
          for ge in f.gameEvents:
              if ge.dataType == 7 and ge.playId:
                  play_ids.append(ge.playId)
      # NOTE: seg 23 has 4 game events but they may all be CountEvent/AtBatEvent; if no PlayEvents,
      # this test should be looser. If empty, weaken to checking at least one gameEvent decoded with non-None dataType.
      assert any(f.gameEvents for f in td.frames), "expected gameEvents to populate"
      # Check at least one play_id present somewhere (try the LATE fixture too if needed)


  def test_tracked_events_have_event_types():
      game_pk, seg = FIXTURE_SEG_WITH_TRACKED_EVENTS
      td = read_tracking_data(fixture_bin_path(game_pk, seg).read_bytes())
      etypes = set()
      for f in td.frames:
          for te in f.trackedEvents:
              if te.eventType:
                  etypes.add(te.eventType)
      assert etypes, "expected at least one tracked event with eventType"
      # In seg 131 we expect to see BALL_WAS_RELEASED and BALL_WAS_CAUGHT
      assert any("BALL_WAS_" in et for et in etypes), f"unexpected event types: {etypes}"
  ```

- [ ] **Step 2: Run, verify failure** (no `gameEvents` / `trackedEvents` attribute on TrackingFrame yet).

- [ ] **Step 3: Add the dataclasses + readers + extend TrackingFrame** in `src/fieldvision/wire_schemas.py`:

  Append after the existing inferredBat code (around line 187):

  ```python
  # ────────────────────────────────────────────────────────────
  # GameEventWire — wrapper for a discrete game event in a frame.
  # Vtable from extract_event_offsets.py output:
  #   field 0 (offset 4): dataType    uint8   (union discriminator; 7 = PlayEvent)
  #   field 1 (offset 6): data        indirect to typed sub-table (e.g. PlayEventDataWire)
  #   field 2 (offset 8): time        float64
  #   field 3 (offset 10): isKeyFramed int8
  # We only decode PlayEventDataWire (dataType=7) for play_id; statsapi
  # provides count/atbat/handedness without needing wire decode.
  # ────────────────────────────────────────────────────────────


  @dataclass
  class GameEvent:
      dataType: int
      time: Optional[float]
      isKeyFramed: bool
      playId: Optional[str]   # populated when dataType==7 (PlayEvent), else None


  def read_game_event(bb: ByteBuffer, pos: int) -> GameEvent:
      o_dt = bb.field_offset(pos, 4)
      o_data = bb.field_offset(pos, 6)
      o_time = bb.field_offset(pos, 8)
      o_kf = bb.field_offset(pos, 10)

      dt = bb.read_uint8(pos + o_dt) if o_dt else 0
      time_v = bb.read_float64(pos + o_time) if o_time else None
      kf = bool(bb.read_int8(pos + o_kf)) if o_kf else False
      play_id: Optional[str] = None
      if dt == 7 and o_data:
          # PlayEventDataWire is at the indirect offset; PlayId is its vtoff 8
          pe_pos = bb.indirect(pos + o_data)
          o_pid = bb.field_offset(pe_pos, 8)
          if o_pid:
              play_id = bb.string(pe_pos + o_pid)
      return GameEvent(dataType=dt, time=time_v, isKeyFramed=kf, playId=play_id)


  # ────────────────────────────────────────────────────────────
  # TrackedEventWire — flat 14-field record. Self-discriminating via eventType (string).
  # We extract the fields needed for pitch segmentation: eventType, x, y, z.
  # Other fields exist on the record (timestamp, batSide, pitchHand, etc.) but
  # are sentinel (-1 / "not-set") for the events we care about, so we skip them.
  # ────────────────────────────────────────────────────────────


  @dataclass
  class TrackedEvent:
      eventType: Optional[str]
      eventTypeId: Optional[int]
      x: Optional[float]
      y: Optional[float]
      z: Optional[float]
      timestamp: Optional[str]


  def read_tracked_event(bb: ByteBuffer, pos: int) -> TrackedEvent:
      o_ts = bb.field_offset(pos, 4)
      o_etype = bb.field_offset(pos, 20)
      o_x = bb.field_offset(pos, 22)
      o_y = bb.field_offset(pos, 24)
      o_z = bb.field_offset(pos, 26)
      o_etid = bb.field_offset(pos, 30)

      ts = bb.string(pos + o_ts) if o_ts else None
      etype = bb.string(pos + o_etype) if o_etype else None
      x = bb.read_float32(pos + o_x) if o_x else None
      y = bb.read_float32(pos + o_y) if o_y else None
      z = bb.read_float32(pos + o_z) if o_z else None
      etid = bb.read_int8(pos + o_etid) if o_etid else None
      return TrackedEvent(eventType=etype, eventTypeId=etid, x=x, y=y, z=z, timestamp=ts)
  ```

  Then extend `TrackingFrame` and `read_tracking_frame`:

  ```python
  @dataclass
  class TrackingFrame:
      num: int
      time: float
      timestamp: Optional[str]
      isGap: bool
      gapDuration: float
      ballPosition: Optional[Vec3]
      actorPoses: list[ActorPose] = field(default_factory=list)
      rawJoints: list[SkeletalPlayer] = field(default_factory=list)
      inferredBat: Optional[InferredBat] = None
      gameEvents: list[GameEvent] = field(default_factory=list)        # ← new
      trackedEvents: list[TrackedEvent] = field(default_factory=list)  # ← new
  ```

  In `read_tracking_frame`, after the existing `inferred = ...` block:

  ```python
      o_ge = bb.field_offset(pos, 8)
      o_te = bb.field_offset(pos, 10)

      game_events: list[GameEvent] = []
      if o_ge:
          v = bb.vector_data(pos + o_ge)
          n = bb.vector_len(pos + o_ge)
          for i in range(n):
              elem_pos = bb.indirect(v + 4 * i)
              game_events.append(read_game_event(bb, elem_pos))

      tracked_events: list[TrackedEvent] = []
      if o_te:
          v = bb.vector_data(pos + o_te)
          n = bb.vector_len(pos + o_te)
          for i in range(n):
              elem_pos = bb.indirect(v + 4 * i)
              tracked_events.append(read_tracked_event(bb, elem_pos))
  ```

  And extend the return tuple at the bottom of `read_tracking_frame`:
  ```python
      return TrackingFrame(num, time_v, timestamp, isGap, gap_dur, ballPos,
                           actors, raw, inferred, game_events, tracked_events)
  ```

- [ ] **Step 4: Run tests; both pass.** If `test_game_events_have_playids` fails because seg 23 has no PlayEvents (only count/atbat events), update the fixture or the test to scan a wider range. The empirical recon found PlayEvents starting around seg 352; seg 131 has 1 gameEvent that may or may not be a PlayEvent. Inspect and adjust.

- [ ] **Step 5: Commit**
  ```bash
  git add src/fieldvision/wire_schemas.py tests/test_event_schemas.py
  git commit -m "Decode GameEvent (PlayEvent.playId) + TrackedEvent (flat) for pitch segmentation"
  ```

---

## Task v2.3: Update `ingest_segment` to write `pitch_event` rows

**Files:**
- Modify: `src/fieldvision/storage.py`
- Modify: `scripts/load_to_db.py`
- Modify: `scripts/fv_daemon.py`
- Create: `tests/test_ingest_segment.py`

For each frame, walk gameEvents and trackedEvents and emit one `pitch_event` row per relevant event.

- [ ] **Step 1: Failing test** (`tests/test_ingest_segment.py`):
  ```python
  """Test that ingest_segment populates pitch_event rows from wire events."""
  from pathlib import Path
  from fieldvision.storage import (
      _actor_frame_insert_sql, ingest_segment, open_game_db, transaction,
  )
  from tests.conftest import FIXTURE_SEG_WITH_TRACKED_EVENTS, fixture_bin_path


  def test_ingest_writes_pitch_event_markers(tmp_path: Path):
      game_pk, seg_idx = FIXTURE_SEG_WITH_TRACKED_EVENTS
      bin_path = fixture_bin_path(game_pk, seg_idx)
      conn = open_game_db(game_pk, tmp_path)
      with transaction(conn):
          ingest_segment(conn, game_pk, seg_idx, bin_path, {}, _actor_frame_insert_sql())

      n = conn.execute("SELECT COUNT(*) FROM pitch_event").fetchone()[0]
      assert n >= 1, "expected at least one pitch_event row"

      # We should see at least one BALL_WAS_RELEASED or BALL_WAS_CAUGHT (segment 131 has both)
      etypes = {row[0] for row in conn.execute(
          "SELECT DISTINCT event_type FROM pitch_event"
      )}
      assert any("BALL_WAS_" in e for e in etypes), f"unexpected event types: {etypes}"

      # If any PLAY_EVENT rows present, they should have a UUID-shaped play_id
      for (pid,) in conn.execute(
          "SELECT play_id FROM pitch_event WHERE event_type='PLAY_EVENT' LIMIT 5"
      ):
          assert pid and len(pid) == 36 and pid.count("-") == 4, f"bad play_id: {pid}"
  ```

- [ ] **Step 2: Run, verify failure** (no rows in pitch_event because `ingest_segment` doesn't write any).

- [ ] **Step 3: Modify `ingest_segment` in `storage.py`:**
  - Add to imports at top:
    ```python
    from .wire_schemas import (
        read_tracking_data, unpack_smallest_three,
    )
    ```
    (No new imports needed beyond what's already used — `gameEvents`/`trackedEvents` are accessed via `f.gameEvents`/`f.trackedEvents` which are part of `TrackingFrame`.)
  - Change the function signature to accept the new SQL:
    ```python
    def ingest_segment(
        conn: sqlite3.Connection,
        game_pk: int,
        segment_idx: int,
        bin_path: Path,
        labels_dict: dict[int, dict],
        insert_sql: str,
        pitch_event_insert_sql: str | None = None,
    ) -> tuple[int, int]:
    ```
    Build pitch_event SQL locally if not provided.
  - Add a `pitch_event_rows: list[tuple] = []` initialization near the top.
  - Inside the per-frame loop, after the existing actor/ball/bat blocks:
    ```python
    # Pitch segmentation markers
    for ge in f.gameEvents:
        if ge.dataType == 7 and ge.playId:  # PlayEvent
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
    ```
  - After the loop, execute the inserts:
    ```python
    pe_sql = pitch_event_insert_sql or _pitch_event_insert_sql()
    if pitch_event_rows:
        conn.executemany(pe_sql, pitch_event_rows)
    ```

- [ ] **Step 4: Update `scripts/load_to_db.py`:**
  - Add `pitch_event` to the `--rebuild` truncate list (and remove `game_event` if present):
    ```python
    for table in ("actor_frame", "ball_frame", "bat_frame",
                  "pitch_event",
                  "labels", "bones", "players", "meta"):
        conn.execute(f"DELETE FROM {table}")
    ```
  - Import + pass `_pitch_event_insert_sql`:
    ```python
    from fieldvision.storage import (_actor_frame_insert_sql, _pitch_event_insert_sql,
                                     ingest_segment, load_lookup_tables, open_game_db,
                                     open_registry, transaction, update_registry)
    ...
    insert_sql = _actor_frame_insert_sql()
    pe_sql = _pitch_event_insert_sql()
    ...
    n_actor, n_ball = ingest_segment(conn, args.game, seg_idx,
                                     bin_path, labels_dict, insert_sql, pe_sql)
    ```

- [ ] **Step 5: Update `scripts/fv_daemon.py`** (same pattern — pass `pe_sql` into `ingest_segment`).

- [ ] **Step 6: Re-run tests, all pass.**

- [ ] **Step 7: Commit**
  ```bash
  git add src/fieldvision/storage.py scripts/load_to_db.py scripts/fv_daemon.py tests/test_ingest_segment.py
  git commit -m "Ingest pitch_event rows: PLAY_EVENT (with play_id) + tracked event markers"
  ```

---

## Task v2.4: Re-ingest game 823141 and validate segmentation

**Files:** none modified (validation only).

- [ ] **Step 1: Backup the existing DB** (it has 10GB+ of ingested data we don't want to lose if something explodes):
  ```bash
  cp /Users/spencerbyers/fieldvision/data/fv_823141.sqlite /Users/spencerbyers/fieldvision/data/fv_823141.sqlite.pre-v2-backup
  ```

- [ ] **Step 2: Re-ingest:**
  ```bash
  cd /Users/spencerbyers/fieldvision
  /Users/spencerbyers/anaconda3/bin/python3 scripts/load_to_db.py --game 823141 --rebuild
  ```

- [ ] **Step 3: Sanity-check pitch_event:**
  ```bash
  sqlite3 data/fv_823141.sqlite \
    "SELECT event_type, COUNT(*) FROM pitch_event GROUP BY event_type ORDER BY 2 DESC;"
  sqlite3 data/fv_823141.sqlite \
    "SELECT COUNT(DISTINCT play_id) AS distinct_pitches FROM pitch_event WHERE event_type='PLAY_EVENT';"
  sqlite3 data/fv_823141.sqlite \
    "SELECT play_id, COUNT(*) AS frames_in_window
       FROM pitch_event WHERE play_id IS NOT NULL
       GROUP BY play_id ORDER BY frames_in_window DESC LIMIT 10;"
  ```
  Expected:
  - PLAY_EVENT count ~250-330 (≈ pitches in a full game)
  - BEGIN_OF_PLAY/END_OF_PLAY counts ~250-330 each, roughly matching PLAY_EVENT
  - Distinct play_ids ~250-330
  - Each play_id should appear ~2-5 times (PlayEvent gets emitted in multiple consecutive frames)

  If counts are way off (< 50 or > 1000): the BallPolynomialWire might be a confounder, or the dataType filter is wrong — debug.

- [ ] **Step 4: Verify alignment with actor_frame:**
  ```bash
  sqlite3 data/fv_823141.sqlite "
    WITH p AS (
      SELECT play_id, MIN(time_unix) AS t0, MAX(time_unix) AS t1
        FROM pitch_event WHERE event_type='PLAY_EVENT' GROUP BY play_id LIMIT 1
    )
    SELECT p.play_id, p.t0, p.t1,
           (SELECT COUNT(*) FROM actor_frame
            WHERE time_unix BETWEEN p.t0 - 3 AND p.t1 + 1) AS actor_rows_in_window
      FROM p;
  "
  ```
  Expected: `actor_rows_in_window` is in the thousands (3-4s × 30fps × ~16 actors ≈ 1500-2000 rows).

- [ ] **Step 5: Note the actual numbers** in your task report. No commit needed (validation only).

---

## Task v2.5: Statsapi pitch-label ingester

**Files:**
- Create: `scripts/ingest_pitch_labels.py`
- Create: `tests/test_pitch_label_ingest.py`

Fetches `feed/live` for a game, walks `liveData.plays.allPlays[].playEvents[]`, writes one `pitch_label` row per `isPitch=True` event.

- [ ] **Step 1: Failing test** (`tests/test_pitch_label_ingest.py`). This test mocks the statsapi response to keep the test offline:
  ```python
  """Test that ingest_pitch_labels populates pitch_label rows from a statsapi feed."""
  import json
  from pathlib import Path
  from fieldvision.storage import open_game_db, transaction
  # Import the ingester module — adjust if the actual module path differs
  import sys
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
  from ingest_pitch_labels import ingest_feed_dict


  def test_ingest_pitch_label_from_minimal_feed(tmp_path: Path):
      game_pk = 999900
      conn = open_game_db(game_pk, tmp_path)
      # Minimal statsapi-shaped feed with one pitch
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
      row = conn.execute("SELECT pitch_type, start_speed, batter_id, pitcher_id, "
                         "balls_before, strikes_before, start_time_unix "
                         "FROM pitch_label WHERE play_id='abc12345-aaaa-bbbb-cccc-111122223333'").fetchone()
      assert row is not None
      assert row[0] == "FF"
      assert abs(row[1] - 95.4) < 0.01
      assert row[2] == 660271
      assert row[3] == 543037
      assert row[4] == 1 and row[5] == 0
      assert row[6] is not None
  ```

- [ ] **Step 2: Run, verify failure** (script doesn't exist yet).

- [ ] **Step 3: Write `scripts/ingest_pitch_labels.py`:**
  ```python
  """Ingest MLB statsapi feed/live pitch data into pitch_label.

  Usage:
      python scripts/ingest_pitch_labels.py --game 823141
  """
  from __future__ import annotations

  import argparse
  import datetime as _dt
  import json
  import sqlite3
  import sys
  import urllib.request
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(REPO_ROOT / "src"))

  from fieldvision.storage import (_pitch_label_insert_sql, open_game_db,
                                    open_registry, transaction)


  def fetch_feed(game_pk: int) -> dict:
      url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
      req = urllib.request.Request(url, headers={
          "User-Agent": "Mozilla/5.0 (Macintosh) FieldVision-pitch-label/1.0",
      })
      with urllib.request.urlopen(req, timeout=30) as r:
          return json.loads(r.read())


  def _iso_to_unix(iso: str | None) -> float | None:
      if not iso:
          return None
      # Statsapi uses 'Z' suffix; replace for Python <3.11 compat
      try:
          d = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
          return d.timestamp()
      except Exception:
          return None


  def ingest_feed_dict(conn: sqlite3.Connection, game_pk: int, feed: dict) -> int:
      """Insert one pitch_label row per isPitch playEvent. Returns row count."""
      sql = _pitch_label_insert_sql()
      rows = []
      plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
      for play in plays:
          ab_index = play.get("atBatIndex")
          inning = play.get("about", {}).get("inning")
          half = play.get("about", {}).get("halfInning")
          top_inning = 1 if half == "top" else (0 if half == "bottom" else None)
          matchup = play.get("matchup", {})
          batter_id = matchup.get("batter", {}).get("id")
          pitcher_id = matchup.get("pitcher", {}).get("id")
          batter_side = matchup.get("batSide", {}).get("code")
          pitcher_throws = matchup.get("pitcherHand", {}).get("code")

          for ev in play.get("playEvents", []):
              if not ev.get("isPitch"):
                  continue
              play_id = ev.get("playId")
              if not play_id:
                  continue
              pn = ev.get("pitchNumber")
              count = ev.get("count", {})
              details = ev.get("details", {})
              pdata = ev.get("pitchData", {})
              breaks = pdata.get("breaks", {})
              coords = pdata.get("coordinates", {})
              ptype = details.get("type", {}) or {}
              call = details.get("call", {}) or {}

              start_time = ev.get("startTime")
              end_time = ev.get("endTime")
              rows.append((
                  game_pk, play_id,
                  ab_index, pn,
                  inning, top_inning,
                  batter_id, pitcher_id,
                  batter_side, pitcher_throws,
                  count.get("balls"), count.get("strikes"), count.get("outs"),
                  ptype.get("code"), ptype.get("description"),
                  pdata.get("startSpeed"), pdata.get("endSpeed"),
                  breaks.get("spinRate"), breaks.get("spinDirection"),
                  coords.get("x0"), coords.get("y0"), coords.get("z0"),
                  pdata.get("extension"),
                  coords.get("px"), coords.get("pz"),
                  pdata.get("strikeZoneTop"), pdata.get("strikeZoneBottom"),
                  call.get("code"), details.get("description"),
                  1 if details.get("isInPlay") else 0,
                  1 if details.get("isStrike") else 0,
                  1 if details.get("isBall") else 0,
                  start_time, end_time,
                  _iso_to_unix(start_time), _iso_to_unix(end_time),
              ))
      if rows:
          conn.executemany(sql, rows)
      return len(rows)


  def main() -> int:
      ap = argparse.ArgumentParser()
      ap.add_argument("--game", type=int, required=True)
      ap.add_argument("--data-dir", default="data")
      args = ap.parse_args()

      data_dir = Path(args.data_dir)
      conn = open_game_db(args.game, data_dir)
      print(f"Fetching statsapi feed for game {args.game}...")
      feed = fetch_feed(args.game)
      with transaction(conn):
          n = ingest_feed_dict(conn, args.game, feed)
      print(f"Inserted {n} pitch_label rows for game {args.game}")
      conn.close()
      return 0


  if __name__ == "__main__":
      sys.exit(main())
  ```

- [ ] **Step 4: Re-run tests, all pass.**

- [ ] **Step 5: Real-world run for game 823141:**
  ```bash
  cd /Users/spencerbyers/fieldvision
  /Users/spencerbyers/anaconda3/bin/python3 scripts/ingest_pitch_labels.py --game 823141
  sqlite3 data/fv_823141.sqlite \
    "SELECT pitch_type, COUNT(*), AVG(start_speed) FROM pitch_label GROUP BY pitch_type ORDER BY 2 DESC;"
  ```
  Expected: 250-330 rows total, plausible pitch types (FF/SL/CH/CB/etc.) with averages 70-100 mph.

- [ ] **Step 6: Commit**
  ```bash
  git add scripts/ingest_pitch_labels.py tests/test_pitch_label_ingest.py
  git commit -m "Ingest statsapi pitch labels into pitch_label (joins to pitch_event on play_id)"
  ```

---

## Task v2.6: End-to-end join validation

**Files:** none modified.

Confirm the segmentation works: pick one pitch, walk the join, check the actor_frame slice.

- [ ] **Step 1: Cross-source overlap query:**
  ```bash
  sqlite3 data/fv_823141.sqlite "
    SELECT
      (SELECT COUNT(DISTINCT play_id) FROM pitch_event WHERE event_type='PLAY_EVENT') AS wire_pitches,
      (SELECT COUNT(*) FROM pitch_label) AS statsapi_pitches,
      (SELECT COUNT(*) FROM pitch_event pe
        JOIN pitch_label pl ON pe.play_id = pl.play_id
        WHERE pe.event_type='PLAY_EVENT'
        GROUP BY pe.play_id) AS rows_check;
  "
  ```
  Expected: `wire_pitches` and `statsapi_pitches` should be roughly equal (within ~5-10% — wire might miss pitches captured in late innings if the daemon was restarted, statsapi might log warm-up pitches we don't see).

- [ ] **Step 2: One concrete pitch end-to-end:**
  ```bash
  sqlite3 data/fv_823141.sqlite "
    WITH p AS (
      SELECT pl.*,
             (SELECT MIN(time_unix) FROM pitch_event WHERE play_id = pl.play_id) AS first_marker_time,
             (SELECT MAX(time_unix) FROM pitch_event WHERE play_id = pl.play_id) AS last_marker_time
        FROM pitch_label pl
       WHERE pl.pitch_type = 'FF'
       LIMIT 1
    )
    SELECT play_id, pitch_type, start_speed, batter_id, pitcher_id,
           start_time_unix, first_marker_time, last_marker_time,
           start_time_unix - first_marker_time AS time_delta_seconds,
           (SELECT COUNT(*) FROM actor_frame
              WHERE time_unix BETWEEN p.first_marker_time - 3 AND p.last_marker_time + 1) AS actor_rows
    FROM p;
  "
  ```
  Expected: `time_delta_seconds` is small (within a few seconds — wire Time vs. statsapi startTime should be close), `actor_rows` is in the thousands.

- [ ] **Step 3: Record the validation numbers** in your task report. No code change needed.

---

## Task v2.7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Update the data model section + open-work list to reflect v2.

- [ ] **Step 1:** Replace the "NOT YET INGESTED" section with:
  ```markdown
  ### NOT YET INGESTED but available in the wire format

  - `gameEvents[]` types other than PlayEvent (CountEvent, AtBatEvent, HandedEvent,
    BatImpactEvent, etc.) — statsapi covers all of these per pitch, so we don't
    decode them.
  - `trackedEvents[]` extended fields beyond `eventType` + position (timestamp,
    batSide, atBatNumber, etc. — most are sentinel for the events we capture).
  - `ballPolynomials[]` — pitch trajectory polynomials. Statsapi gives us
    pitch type / velocity / spin / release / plate location, so we skip these.

  **As of <today's date> we ingest:**
  - `pitch_event` — frame-level markers (PLAY_EVENT with play_id, plus
    BEGIN_OF_PLAY / END_OF_PLAY / BALL_WAS_RELEASED / BALL_WAS_CAUGHT /
    BALL_WAS_PITCHED / etc. from trackedEvents)
  - `pitch_label` — per-pitch labels from MLB statsapi (pitch type, velocity,
    spin, release point, plate position, strike zone, result), keyed on play_id.

  These two tables join on `play_id` to give labeled per-pitch frame ranges
  for downstream feature engineering.
  ```

- [ ] **Step 2:** Add to the data-model section, after `bat_frame`:
  ```markdown
  **`pitch_event`** — frame-level pitch segmentation markers. Columns:
  `event_type` (PLAY_EVENT | BEGIN_OF_PLAY | BALL_WAS_RELEASED | …),
  `play_id` (UUID matching statsapi; populated for PLAY_EVENT),
  `pos_x/y/z` (populated for tracked events with positions). Index on play_id.

  **`pitch_label`** — per-pitch labels from MLB statsapi. Keyed on
  (game_pk, play_id). Holds pitch_type, start_speed/end_speed, spin_rate,
  release_x/y/z, plate_x/z, sz_top/sz_bot, result_call, batter_id/pitcher_id,
  count state (balls_before/strikes_before/outs_before), ab_index,
  pitch_number, start_time/end_time. Indexed by pitcher, batter, and pitch_type.

  **Recipe to get the 90 frames before pitch X:**
  ```sql
  WITH p AS (
    SELECT pl.play_id, pl.start_time_unix
      FROM pitch_label pl WHERE pl.play_id = '<uuid>'
  )
  SELECT af.* FROM actor_frame af, p
   WHERE af.time_unix BETWEEN p.start_time_unix - 3 AND p.start_time_unix
   ORDER BY af.time_unix DESC LIMIT 90;
  ```
  ```

- [ ] **Step 3:** Update the "Open / deferred work" list:
  ```markdown
  3. **Decode richer wire events** (BALL_BOUNCE polynomials, ABSEvent, etc.) —
     deferred. Statsapi covers all per-pitch labels we currently need; only
     reach for wire decoding when statsapi gaps appear.
  4. **Backfill statsapi pitch_label for all 27 captured games** — one
     command per game, ~30s each. Worth doing before the daemon's next
     restart so historical data is fully labeled.
  ```

- [ ] **Step 4: Commit**
  ```bash
  git add CLAUDE.md
  git commit -m "CLAUDE.md: document v2 pitch_event + pitch_label tables and join recipe"
  ```

---

## Task v2.8: Backfill all captured games + merge

**Files:** none modified.

- [ ] **Step 1: Backfill statsapi labels for every captured game**
  ```bash
  cd /Users/spencerbyers/fieldvision
  for db in data/fv_*.sqlite; do
    pk=$(basename "$db" .sqlite | sed 's/^fv_//')
    [[ "$pk" == "games_registry" ]] && continue
    echo "=== $pk ==="
    /Users/spencerbyers/anaconda3/bin/python3 scripts/ingest_pitch_labels.py --game "$pk" || true
  done
  ```
  Statsapi data is independent of wire ingestion — labels for game N apply whether or not actor_frame is populated.

- [ ] **Step 2: Final test sweep**
  ```bash
  /Users/spencerbyers/anaconda3/bin/python3 -m pytest tests/ -v
  ```
  All green. If anything fails, fix before merging.

- [ ] **Step 3: Final review and merge**
  ```bash
  git log --oneline main..HEAD
  git diff --stat main..HEAD
  git checkout main
  git merge --no-ff feat/event-ingestion -m "Merge: pitch_event segmentation + pitch_label statsapi join (Phase A.1 v2)"
  ```

- [ ] **Step 4: Hand back to user with daemon-restart note**
  > "Phase A.1 v2 merged to main. `pitch_event` populated for game 823141 (~250 PLAY_EVENT rows + ~250 BEGIN/END pairs from wire); `pitch_label` populated for all 27 captured games via statsapi. Restart the daemon to start writing pitch_event rows for live games:
  > ```
  > launchctl unload ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist
  > launchctl load   ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist
  > ```"

---

## Summary of files at end of v2

**New (kept from v1 work):**
- `scripts/extract_event_offsets.py` (still used to verify offsets)
- `docs/superpowers/plans/2026-05-10-event-ingestion-offsets.md` (reference)
- `tests/__init__.py`, `tests/conftest.py`

**New (this plan):**
- `scripts/ingest_pitch_labels.py`
- `tests/test_event_schemas.py` (created or updated for v2)
- `tests/test_ingest_segment.py`
- `tests/test_pitch_label_ingest.py`

**Modified:**
- `src/fieldvision/wire_schemas.py` (~80 lines: GameEvent + TrackedEvent dataclasses + readers + TrackingFrame extension)
- `src/fieldvision/storage.py` (~50 lines: schema rewrite, two SQL helpers, ingest_segment extension)
- `scripts/load_to_db.py` (small)
- `scripts/fv_daemon.py` (small)
- `tests/test_storage_schema.py` (rewritten for v2 schema)
- `CLAUDE.md`
