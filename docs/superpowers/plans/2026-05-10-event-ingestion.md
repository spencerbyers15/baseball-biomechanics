# Event Ingestion (gameEvents + trackedEvents) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode `gameEvents[]` and `trackedEvents[]` from each `TrackingFrameWire`, persist them into queryable SQLite tables (`game_event` + `pitch_event`), and verify pitch counts match statsapi for at least one re-ingested game — providing the per-pitch labels needed by `PROJECT_PREDICTION.md` Phase B (feature engineering).

**Architecture:** Add new FlatBuffer table readers in `src/fieldvision/wire_schemas.py` for `GameEventWire`, `TrackedEventWire`, and the highest-value event-data tables (`PlayEventDataWire`, `CountEventDataWire`, `AtBatEventDataWire`, `HandedEventDataWire`, `BallPitchDataWire`, `BatImpactEventDataWire`). Extend `TrackingFrame` to populate parsed event lists, extend `storage.py` with a richer `game_event` schema and a new `pitch_event` table, and extend `ingest_segment` to write event rows. Validate by re-ingesting an existing sample game and comparing pitch counts to MLB statsapi.

**Tech Stack:** Python 3.11, hand-rolled FlatBuffer reader (`flatbuf_runtime.ByteBuffer`), SQLite (per-game DB), pytest for unit tests, urllib for statsapi cross-checks. Schemas come from `gd.bvg_poser.min.js` (the MLB Gameday FieldVision bundle).

**Current state pre-plan:**
- `src/fieldvision/wire_schemas.py` decodes `TrackingDataWire`, `TrackingFrameWire` (frame-level metadata + actorPoses + inferredBat + ballPosition + rawJoints), `ActorPoseWire`, `SkeletalPlayerWire`, `TrackingBatPositionWire`, `Vec3`, and the smallest-three quaternion unpacker.
- `TrackingFrameWire` vtable offsets 8 (`gameEvents`), 10 (`trackedEvents`), 14 (`ballPolynomials`) are documented in comments but **not read** — this plan reads 8 and 10 and defers 14.
- `src/fieldvision/storage.py` has an empty stub `game_event` table (`event_type TEXT, data_json TEXT`) but `ingest_segment` never inserts into it.
- Daemon `scripts/fv_daemon.py` runs against `main` via launchd; the in-memory copy is unaffected by file edits until restart. We can develop on a feature branch safely.
- 27 captured games exist in `samples/binary_capture_*/`. We will use `samples/binary_capture_823141/` (already had decoded.json work done against it) as our primary fixture.
- The JS bundle `gd.bvg_poser.min.js` is **not on disk** — the previous session decoded it but didn't save a copy. **Task 1** is recovering it via a DevTools paste snippet (same pattern as `scripts/snippets/refresh_token.js`).

**Out of scope for this plan (deferred to Phase A.2):**
- `ballPolynomials[]` (pitch trajectory polynomials)
- Less-critical event types: `InningEventDataWire`, `PositionAssignmentEventDataWire`, `ABSEventDataWire`, `TeamScoreEventDataWire`, `BattingOrderEventDataWire`, `LiveActionEventDataWire`, `GumboTimecodeEventDataWire`, `StatusEventDataWire`, `HighFrequencyBatMarkerEventDataWire`
- Tracked events other than pitch: `BallHitDataWire`, `BallThrowData`, `BallBounceDataWire`
- Backfilling existing per-game DBs from on-disk `.bin` files (only the fresh re-ingest of one validation game is in scope; a full backfill script is its own follow-up plan).
- statsapi enrichment beyond pitch-count sanity check.
- Anything in Phase B/C/D/E (features, models, market comparison, deployment).

**File map:**
- *Modify* `src/fieldvision/wire_schemas.py` — add `GameEventWire`, `TrackedEventWire`, event-data dataclasses + readers, union dispatcher, extend `TrackingFrame` + `read_tracking_frame` to populate event lists.
- *Modify* `src/fieldvision/storage.py` — replace `game_event` schema with structured columns, add `pitch_event` table, add insert SQL helpers, extend `ingest_segment` to write event/pitch rows.
- *Modify* `scripts/load_to_db.py` — include `pitch_event` in the `--rebuild` truncate list.
- *Create* `scripts/snippets/fetch_bundle.js` — DevTools paste to grab `gd.bvg_poser.min.js`.
- *Create* `scripts/extract_event_offsets.py` — one-shot tool that scans the JS bundle and emits a markdown table of vtable offsets per class (used as ground truth for the schema port).
- *Create* `tests/__init__.py`, `tests/conftest.py`, `tests/test_event_schemas.py` — pytest unit tests against `samples/binary_capture_823141/mlb_823141_segment_*.bin` fixtures. We pick segments that empirically contain the events we need (Task 4.b finds them).
- *Create* `scripts/validate_pitch_count.py` — fetch statsapi `game/feed/live` for a gamePk, count pitches, compare to `pitch_event` row count.
- *Modify* `CLAUDE.md` — update the "NOT YET INGESTED" section to reflect what's now ingested + the new `Open / deferred work` list.
- *Optional, leave for later* `PROJECT_PREDICTION.md` — Phase A note that A.1 is done.

---

## Task 0: Set up feature branch and confirm baseline

**Files:** none yet (working in repo root).

- [ ] **Step 1: Confirm clean tree and create branch**

```bash
cd /Users/spencerbyers/fieldvision
git status
git checkout -b feat/event-ingestion
```

Expected: `nothing to commit, working tree clean` then `Switched to a new branch 'feat/event-ingestion'`.

- [ ] **Step 2: Verify daemon is unaffected by branch checkout**

Daemon runs from in-memory Python; file changes don't take effect until launchctl restart. Confirm it's still alive:

```bash
launchctl list | grep fvcapture
```

Expected: a numeric PID in column 1 (e.g. `15508`). If column 1 is `-`, daemon crashed — investigate `scheduler.err` before continuing.

- [ ] **Step 3: Confirm pytest works**

```bash
~/anaconda3/envs/baseball/bin/python -m pytest --version
```

Expected: `pytest 8.3.x`.

If the `baseball` env doesn't exist: `which python3 && python3 -m pytest --version` and use that instead. Throughout the plan, "PY" refers to whichever Python has `pytest` installed; substitute the absolute path.

- [ ] **Step 4: Save the working Python path for later steps**

```bash
PY=$(/usr/bin/which python3); ~/anaconda3/envs/baseball/bin/python -c "import pytest" 2>/dev/null && PY=~/anaconda3/envs/baseball/bin/python; echo "Use: $PY"
```

Note the path printed — every subsequent `pytest` / `python` command should use it.

---

## Task 1: Recover `gd.bvg_poser.min.js` from MLB

**Files:**
- Create: `scripts/snippets/fetch_bundle.js`

This is a one-time recon step. Spencer (the user) runs the DevTools snippet in his logged-in mlb.com Chrome tab; it identifies the bundle URL from `performance.getEntriesByType('resource')`, downloads the source, and saves it to `~/Downloads`. We then move it into the repo.

- [ ] **Step 1: Write the DevTools snippet**

Create `scripts/snippets/fetch_bundle.js` with:

```javascript
(async () => {
  // Find the bvg_poser bundle URL from network resource entries.
  const entries = performance.getEntriesByType('resource');
  const candidates = entries
    .map(e => e.name)
    .filter(u => /bvg[_.-]?poser.*\.js/i.test(u) || /gd\.bvg.*\.js/i.test(u));
  if (candidates.length === 0) {
    console.error('No bvg_poser bundle found. Open a game with FieldVision/3D view first, then re-run.');
    return;
  }
  const url = candidates[0];
  console.log('Fetching:', url);

  // Same-origin fetch — uses your authenticated session cookies.
  const res = await fetch(url, { credentials: 'include' });
  if (!res.ok) { console.error('HTTP', res.status); return; }
  const src = await res.text();

  const blob = new Blob([src], { type: 'application/javascript' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'gd.bvg_poser.min.js';
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`Saved gd.bvg_poser.min.js (${src.length.toLocaleString()} chars) to ~/Downloads`);
  console.log('Source URL:', url);
})();
```

- [ ] **Step 2: Have Spencer run the snippet**

Hand off to the user with this message:

> "Open mlb.com in your normal Chrome (logged in), navigate to a recent live game's Gameday view (so the FieldVision bundle loads), open DevTools (Cmd+Opt+J), and paste the contents of `scripts/snippets/fetch_bundle.js`. It saves the bundle to `~/Downloads/gd.bvg_poser.min.js`. Then run:
>
> ```
> mv ~/Downloads/gd.bvg_poser.min.js scripts/snippets/gd.bvg_poser.min.js
> ```
>
> Confirm when done."

**Wait for confirmation before proceeding.** If Spencer says the snippet returns "No bvg_poser bundle found", the bundle name has changed; broaden the regex (e.g. `/bvg|poser|fieldvision/i`) and try again.

**Fallback if the bundle truly cannot be recovered:** the entire plan from Task 2
onward depends on having the JS source. If MLB has stopped serving it (or Spencer's
account no longer has FieldVision entitlement), an empirical-recovery path exists:
walk the vtables of `gameEvents[]` items in a sample bin, dump field count + raw
bytes per offset, and probe field types by looking at value plausibility. This is
a separate ~half-day of work; if Task 1 deadlocks, write a follow-up plan
`2026-MM-DD-empirical-event-schemas.md` rather than mixing it in here.

- [ ] **Step 3: Verify the bundle is present and looks right**

```bash
ls -lh scripts/snippets/gd.bvg_poser.min.js
grep -c 'getRootAs' scripts/snippets/gd.bvg_poser.min.js
grep -o 'getRootAs[A-Z][A-Za-z]*' scripts/snippets/gd.bvg_poser.min.js | sort -u
```

Expected:
- File size in the 100KB–2MB range.
- `getRootAs` count ≥ 10.
- The unique-name list includes (substrings): `BallPitch`, `PlayEvent`, `CountEvent`, `AtBatEvent`, `HandedEvent`, `BatImpactEvent`, `TrackingFrame`, `GameEvent`, `TrackedEvent`. If any are missing, this might be a partial bundle — STOP and re-fetch.

- [ ] **Step 4: Add the bundle to .gitignore (it's MLB's IP, don't ship it)**

```bash
grep -q 'gd.bvg_poser.min.js' .gitignore || echo 'scripts/snippets/gd.bvg_poser.min.js' >> .gitignore
```

- [ ] **Step 5: Commit**

```bash
git add scripts/snippets/fetch_bundle.js .gitignore
git commit -m "Add DevTools snippet to fetch the MLB FieldVision JS bundle"
```

---

## Task 2: Extract field offsets from the bundle

**Files:**
- Create: `scripts/extract_event_offsets.py`
- Create: `docs/superpowers/plans/2026-05-10-event-ingestion-offsets.md` (output artifact)

We need vtable field offsets for: `GameEventWire`, `TrackedEventWire`, `PlayEventDataWire`, `CountEventDataWire`, `AtBatEventDataWire`, `HandedEventDataWire`, `BallPitchDataWire`, `BatImpactEventDataWire`. The convention used by the JS FlatBuffer codegen (already documented in `wire_schemas.py:1-10`):

> Each class has static `add<FieldName>` methods. The first call's `addField0(b, v)` uses vtable offset 4, the next 6, etc. So the order of the `add*` methods (in source) determines field index — multiply (i + 1) × 2 + 2 to get the vtable offset.

The script greps for each class's `add*` method block and prints the field list in source order.

- [ ] **Step 1: Write the extractor**

Create `scripts/extract_event_offsets.py`:

```python
"""Extract vtable field offsets for FieldVision event classes from the JS bundle.

Reads scripts/snippets/gd.bvg_poser.min.js and prints, for each class of
interest, an ordered list of (vtable_offset, fieldName, addExpression).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "scripts" / "snippets" / "gd.bvg_poser.min.js"

CLASSES_OF_INTEREST = [
    "GameEventWire",
    "TrackedEventWire",
    "PlayEventDataWire",
    "CountEventDataWire",
    "AtBatEventDataWire",
    "HandedEventDataWire",
    "BallPitchDataWire",
    "BatImpactEventDataWire",
    "InningEventDataWire",
]


def find_class_block(src: str, name: str) -> str | None:
    """Return the substring covering the class definition, or None.

    The minified bundle may emit either `class Foo {...}` or a function
    constructor. We search for `getRootAs<Name>` first; the surrounding
    {} block is the class body.
    """
    pat = re.compile(rf"\bgetRootAs{re.escape(name)}\s*\(")
    m = pat.search(src)
    if not m:
        return None
    # Walk outward to find the enclosing class { ... }. Minified code has
    # everything on one line, so we use balanced-brace counting.
    # Start from m.start() and walk left to find the `class <Name>` keyword.
    class_kw = re.compile(rf"\bclass\s+{re.escape(name)}\b")
    cm = None
    for cm_candidate in class_kw.finditer(src):
        if cm_candidate.start() < m.start():
            cm = cm_candidate
        else:
            break
    if cm is None:
        # Fall back to a wide window around getRootAs<Name>.
        return src[max(0, m.start() - 200):m.start() + 8000]
    # From cm.end(), find the next '{' and walk to its matching '}'.
    brace_open = src.find("{", cm.end())
    if brace_open < 0:
        return None
    depth = 1
    i = brace_open + 1
    while i < len(src) and depth > 0:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return src[cm.start():i]


def extract_add_methods(class_body: str) -> list[tuple[int, str, str]]:
    """Find static add<FieldName>(b, ...) methods in source order. Returns
    [(vtable_offset, field_name, raw_expression), ...]."""
    # Match: static add<Name>(b, x) { b.addFieldXxx(<idx>, x, ...) }
    # Field index goes after the bracket-call `addFieldXxx(<idx>,`.
    method_pat = re.compile(
        r"static\s+add([A-Z][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{([^}]*)\}",
        re.DOTALL,
    )
    idx_pat = re.compile(r"addField\w+\s*\(\s*(\d+)\s*,")
    out: list[tuple[int, str, str]] = []
    for mm in method_pat.finditer(class_body):
        name = mm.group(1)
        body = mm.group(2)
        idx_match = idx_pat.search(body)
        if not idx_match:
            continue
        field_idx = int(idx_match.group(1))
        vtable_offset = (field_idx + 1) * 2 + 2  # field 0 → 4, field 1 → 6, ...
        out.append((vtable_offset, name, body.strip()))
    out.sort(key=lambda r: r[0])
    return out


def main() -> int:
    if not BUNDLE.exists():
        print(f"ERROR: bundle missing at {BUNDLE}. Run Task 1 first.", file=sys.stderr)
        return 1
    src = BUNDLE.read_text(errors="replace")
    print(f"# Field offsets extracted from {BUNDLE.name} ({len(src):,} chars)")
    print()
    for cls in CLASSES_OF_INTEREST:
        body = find_class_block(src, cls)
        if body is None:
            print(f"## {cls}\n\n  ⚠ NOT FOUND in bundle. Search for alternate name.\n")
            continue
        methods = extract_add_methods(body)
        print(f"## {cls}\n")
        if not methods:
            print("  (no static add* methods found — class may be a non-table.)")
            print()
            continue
        print("| vtable_offset | field |")
        print("|---|---|")
        for off, name, _expr in methods:
            print(f"| {off} | {name} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it and capture output**

```bash
$PY scripts/extract_event_offsets.py | tee docs/superpowers/plans/2026-05-10-event-ingestion-offsets.md
```

Expected: a markdown table per class, each with 3+ rows. If any class shows "NOT FOUND" or 0 methods:
- The bundle may use a different name (try `grep -o 'getRootAs[A-Z][A-Za-z]*' scripts/snippets/gd.bvg_poser.min.js | sort -u`) — adjust `CLASSES_OF_INTEREST` and re-run.
- If `add*` methods aren't matched, the regex may need broadening (e.g., minifiers sometimes drop `static` keyword in extracted form). Inspect the relevant block visually:
  ```bash
  grep -o 'getRootAsBallPitchDataWire[^}]*}' scripts/snippets/gd.bvg_poser.min.js | head -c 2000
  ```

- [ ] **Step 3: Identify the union dispatcher (gameEvents discriminator)**

`TrackingFrameWire.gameEvents[]` is a vector of `GameEventWire` tables. Each `GameEventWire` carries a `dataType` (uint enum) plus a `data` indirect table whose true type is determined by the enum. Find the union enum:

```bash
grep -oE '(?:Lue|GameEventDataType|EventDataType)[ ={a-zA-Z]+' scripts/snippets/gd.bvg_poser.min.js | head -40
```

Expected: an object literal mapping integers to class constructors. Capture this mapping into the offsets doc by hand at the bottom (3-line addition):

```markdown
## GameEvent dataType union dispatch

| dataType | class |
|---|---|
| 1 | PlayEventDataWire |
| 2 | CountEventDataWire |
| ... | ... |
```

If you can't immediately spot the dispatch table, look for the consumer side: `data(t) { return new <Class>(...).__init(...) }` patterns inside `GameEventWire`. The class names tell you the values implicitly.

- [ ] **Step 4: Identify trackedEvents discriminator the same way**

Same approach for `TrackedEventWire`. Add a second table at the bottom of the offsets doc:

```markdown
## TrackedEvent dataType union dispatch

| dataType | class |
|---|---|
| ? | BallPitchDataWire |
| ? | BallHitDataWire |
| ? | BallThrowDataWire |
| ? | BallBounceDataWire |
```

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_event_offsets.py docs/superpowers/plans/2026-05-10-event-ingestion-offsets.md
git commit -m "Extract FieldVision event vtable offsets from JS bundle"
```

---

## Task 3: Add new tables and SQL helpers to storage.py

**Files:**
- Modify: `src/fieldvision/storage.py:60-137` (the `SCHEMA` block) and immediately below.

The existing `game_event` table is too generic for queries we'll run. Replace it with a structured version, then add `pitch_event`. Schema choices justified inline.

- [ ] **Step 1: Write a failing test first**

Create `tests/__init__.py` (empty) and `tests/conftest.py`:

```python
"""Shared pytest fixtures for fieldvision tests."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
```

Create `tests/test_storage_schema.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest tests/test_storage_schema.py -v
```

Expected: 3 failures (`missing columns` for at least `play_id`, `pitch_type`, etc.; missing indexes).

- [ ] **Step 3: Update the schema in storage.py**

In `src/fieldvision/storage.py`, replace the existing `game_event` table block (lines `127-137`, the block beginning `CREATE TABLE IF NOT EXISTS game_event`) with the new schema, and append the `pitch_event` block. Edit `SCHEMA` so the relevant section becomes:

```sql
CREATE TABLE IF NOT EXISTS game_event (
    game_pk INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,
    frame_num INTEGER NOT NULL,
    time_unix REAL NOT NULL,
    event_type TEXT NOT NULL,           -- wire class name (e.g. 'CountEvent', 'PlayEvent')
    data_type INTEGER,                  -- raw union discriminator
    is_key_framed INTEGER,              -- 1 if from TrackingDataWire.eventKeyFrame
    play_id TEXT,                       -- MLB GUID; joins to statsapi pitch records
    -- Common parsed fields, populated when the event type carries them:
    balls INTEGER,
    strikes INTEGER,
    outs INTEGER,
    inning INTEGER,
    top_inning INTEGER,                 -- 1 = top of inning, 0 = bottom
    batter_id INTEGER,
    pitcher_id INTEGER,
    batter_handedness TEXT,             -- 'L' / 'R' / 'S'
    pitcher_handedness TEXT,
    data_json TEXT                      -- raw remainder for unparsed fields
);
CREATE INDEX IF NOT EXISTS idx_event_type ON game_event(event_type, time_unix);
CREATE INDEX IF NOT EXISTS idx_event_play ON game_event(play_id);
CREATE INDEX IF NOT EXISTS idx_event_time ON game_event(time_unix);

CREATE TABLE IF NOT EXISTS pitch_event (
    game_pk INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,
    frame_num INTEGER NOT NULL,
    time_unix REAL NOT NULL,
    play_id TEXT,
    release_x REAL, release_y REAL, release_z REAL,
    plate_x REAL, plate_y REAL, plate_z REAL,
    velocity_release REAL,              -- mph (we'll confirm units empirically)
    velocity_plate REAL,
    spin_rate REAL,                     -- rpm
    spin_axis_x REAL, spin_axis_y REAL, spin_axis_z REAL,
    pitch_type TEXT,                    -- 'FF', 'SL', 'CH', etc. (statsapi-compatible code)
    pitch_type_id INTEGER,              -- raw enum from BallPitchDataWire
    extension REAL,                     -- pitcher extension (ft from rubber to release)
    sz_top REAL, sz_bottom REAL,        -- strike zone for this batter (ft above ground)
    sz_left REAL, sz_right REAL,        -- strike zone left/right (ft from plate centerline)
    PRIMARY KEY (game_pk, segment_idx, frame_num)
);
CREATE INDEX IF NOT EXISTS idx_pitch_play ON pitch_event(play_id);
CREATE INDEX IF NOT EXISTS idx_pitch_time ON pitch_event(time_unix);
```

The exact `Edit` invocation: find the existing `game_event` block (12 lines starting `CREATE TABLE IF NOT EXISTS game_event`) and replace with the two-table block above.

**Note:** Existing per-game DBs created before this change will be missing the new columns. `CREATE TABLE IF NOT EXISTS` won't migrate them. The plan handles this in Task 8 (re-ingest creates a fresh DB; for the live daemon's existing DBs we just won't have history for the new columns until the next re-ingest — acceptable since pitch_event is new and game_event was empty).

- [ ] **Step 4: Run the test again to verify it passes**

```bash
$PY -m pytest tests/test_storage_schema.py -v
```

Expected: 3 passes.

- [ ] **Step 5: Add SQL helper builders for the new tables**

Add these helpers to `storage.py`, just above `def ingest_segment(...)`:

```python
def _game_event_insert_sql() -> str:
    cols = [
        "game_pk", "segment_idx", "frame_num", "time_unix",
        "event_type", "data_type", "is_key_framed", "play_id",
        "balls", "strikes", "outs",
        "inning", "top_inning",
        "batter_id", "pitcher_id",
        "batter_handedness", "pitcher_handedness",
        "data_json",
    ]
    placeholders = ", ".join("?" * len(cols))
    return f"INSERT INTO game_event ({', '.join(cols)}) VALUES ({placeholders})"


def _pitch_event_insert_sql() -> str:
    cols = [
        "game_pk", "segment_idx", "frame_num", "time_unix",
        "play_id",
        "release_x", "release_y", "release_z",
        "plate_x", "plate_y", "plate_z",
        "velocity_release", "velocity_plate",
        "spin_rate", "spin_axis_x", "spin_axis_y", "spin_axis_z",
        "pitch_type", "pitch_type_id", "extension",
        "sz_top", "sz_bottom", "sz_left", "sz_right",
    ]
    placeholders = ", ".join("?" * len(cols))
    return f"INSERT OR REPLACE INTO pitch_event ({', '.join(cols)}) VALUES ({placeholders})"
```

(Why `INSERT` for game_event but `INSERT OR REPLACE` for pitch_event: game_event can have multiple rows at the same (segment, frame) — e.g., a CountEvent and a PlayEvent in the same frame. pitch_event is keyed on (segment, frame) so re-ingesting a segment idempotently overwrites.)

- [ ] **Step 6: Commit**

```bash
git add src/fieldvision/storage.py tests/__init__.py tests/conftest.py tests/test_storage_schema.py
git commit -m "Add pitch_event table + structured game_event schema"
```

---

## Task 4: Find sample bins that exercise the events we want

**Files:** none modified — this is recon.

We need to know which segments in `samples/binary_capture_823141/` actually contain pitch events, count events, etc., to use as fixtures. We don't yet have decoders, but we can detect the *presence* of the relevant vectors by reading offsets 8 and 10 of each TrackingFrameWire.

- [ ] **Step 1: Write a one-shot recon script (no new file — inline)**

Run this in the shell to list, for the first 50 segments, how many frames have non-empty gameEvents and trackedEvents:

```bash
$PY - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from fieldvision.flatbuf_runtime import ByteBuffer
from fieldvision.wire_schemas import read_tracking_data

samples = sorted(Path("samples/binary_capture_823141").glob("mlb_823141_segment_*.bin"),
                 key=lambda p: int(p.stem.split("_")[-1]))[:200]

total_ge = total_te = 0
hits = []
for p in samples:
    data = p.read_bytes()
    bb = ByteBuffer(data)
    root = bb.read_int32(0)
    # field 1 (offset 6) of TrackingDataWire is frames vector
    o_frames = bb.field_offset(root, 6)
    if not o_frames:
        continue
    v = bb.vector_data(root + o_frames)
    n = bb.vector_len(root + o_frames)
    seg_ge = seg_te = 0
    for i in range(n):
        f_pos = bb.indirect(v + 4 * i)
        o_ge = bb.field_offset(f_pos, 8)
        o_te = bb.field_offset(f_pos, 10)
        if o_ge:
            seg_ge += bb.vector_len(f_pos + o_ge)
        if o_te:
            seg_te += bb.vector_len(f_pos + o_te)
    if seg_ge or seg_te:
        hits.append((p.name, seg_ge, seg_te))
    total_ge += seg_ge
    total_te += seg_te

print(f"scanned {len(samples)} segments")
print(f"total gameEvent items:     {total_ge}")
print(f"total trackedEvent items:  {total_te}")
print(f"segments with events: {len(hits)}")
for n, ge, te in hits[:15]:
    print(f"  {n}: gameEvents={ge}  trackedEvents={te}")
PY
```

Expected: at least a handful of segments with non-zero `gameEvents` (likely the early segments and any segment containing a pitch). Note 2-3 segment indices that have BOTH gameEvents and trackedEvents > 0 — these become our pytest fixtures.

- [ ] **Step 2: Pick fixtures and document**

From Step 1's output (the printed `hits` list), choose three segment indices and
substitute them into the integer literals below — these are the only "fill in
from prior step" placeholders in the plan, and they exist because we can't know
which segments contain events without running the recon.

- First entry where `gameEvents > 0` → use as `FIXTURE_SEG_WITH_GAME_EVENTS`
- First entry where `trackedEvents > 0` → use as `FIXTURE_SEG_WITH_TRACKED_EVENTS`
- Last entry in `hits[:15]` → use as `FIXTURE_SEG_LATE`

Add these as constants at the top of `tests/conftest.py`:

```python
# Sample segments confirmed (via Task 4 recon) to contain events.
# Format: (game_pk, segment_idx)
# REPLACE the 0 placeholders with the values from Task 4 Step 1's output.
FIXTURE_SEG_WITH_GAME_EVENTS = (823141, 0)      # ← replace 0
FIXTURE_SEG_WITH_TRACKED_EVENTS = (823141, 0)   # ← replace 0
FIXTURE_SEG_LATE = (823141, 0)                  # ← replace 0


def fixture_bin_path(game_pk: int, seg_idx: int):
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "samples" / \
        f"binary_capture_{game_pk}" / f"mlb_{game_pk}_segment_{seg_idx}.bin"
```

- [ ] **Step 3: Verify fixtures resolve**

```bash
$PY -c "from tests.conftest import fixture_bin_path, FIXTURE_SEG_WITH_GAME_EVENTS as F; print(fixture_bin_path(*F).exists())"
```

Expected: `True`. (Run from repo root.)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "Pick sample segments with events as test fixtures"
```

---

## Task 5: Implement GameEventWire + TrackedEventWire wrappers

**Files:**
- Modify: `src/fieldvision/wire_schemas.py` (append at end)

These are the outer wrappers. Each is a table with `dataType` (uint discriminator) and `data` (indirect to the typed event-data table). Plus event-level metadata (frame_num, time, play_id?).

- [ ] **Step 1: Write a failing test**

Append to `tests/test_event_schemas.py` (create the file):

```python
"""Tests for FlatBuffer event schema decoders."""

import pytest

from tests.conftest import (
    FIXTURE_SEG_WITH_GAME_EVENTS,
    FIXTURE_SEG_WITH_TRACKED_EVENTS,
    fixture_bin_path,
)
from fieldvision.wire_schemas import read_tracking_data


def test_game_events_decode_with_dataType():
    game_pk, seg = FIXTURE_SEG_WITH_GAME_EVENTS
    td = read_tracking_data(fixture_bin_path(game_pk, seg).read_bytes())
    found_any = False
    for f in td.frames:
        for ge in f.gameEvents:
            assert isinstance(ge.dataType, int) and ge.dataType >= 0
            found_any = True
    assert found_any, "expected at least one gameEvent in this fixture"


def test_tracked_events_decode_with_dataType():
    game_pk, seg = FIXTURE_SEG_WITH_TRACKED_EVENTS
    td = read_tracking_data(fixture_bin_path(game_pk, seg).read_bytes())
    found_any = False
    for f in td.frames:
        for te in f.trackedEvents:
            assert isinstance(te.dataType, int) and te.dataType >= 0
            found_any = True
    assert found_any, "expected at least one trackedEvent in this fixture"
```

- [ ] **Step 2: Run to verify failure**

```bash
$PY -m pytest tests/test_event_schemas.py -v
```

Expected: ImportError or AttributeError on `f.gameEvents` (TrackingFrame has no such field yet).

- [ ] **Step 3: Add `GameEvent` and `TrackedEvent` dataclasses + readers**

Append to `src/fieldvision/wire_schemas.py`:

```python
# ────────────────────────────────────────────────────────────
# GameEventWire — wrapper for a discrete game event in a frame.
# Vtable from gd.bvg_poser.min.js extract_event_offsets.py output:
#
#   field 0 (offset  4): dataType   uint8/uint16  (union discriminator; see Lue map below)
#   field 1 (offset  6): data       indirect to the typed event-data table
#   field 2 (offset  8): num        int32         (frame number this event applies to)
#   field 3 (offset 10): time       float64       (time_unix)
#   field 4 (offset 12): playId     string        (MLB GUID; not always present)
#
# (CONFIRM exact offsets against extract_event_offsets.py output. Adjust
# if the extracted table differs — comment block must match reality.)
# ────────────────────────────────────────────────────────────


@dataclass
class GameEvent:
    dataType: int
    num: Optional[int]
    time: Optional[float]
    playId: Optional[str]
    # Position of the inner event-data table in the buffer; lazy-decoded
    # by the dispatch function so callers can choose what to materialize.
    data_pos: Optional[int]
    # Reference back to the buffer for lazy decode.
    _bb: Optional["ByteBuffer"] = None


def read_game_event(bb: ByteBuffer, pos: int) -> GameEvent:
    o_dt = bb.field_offset(pos, 4)
    o_data = bb.field_offset(pos, 6)
    o_num = bb.field_offset(pos, 8)
    o_time = bb.field_offset(pos, 10)
    o_pid = bb.field_offset(pos, 12)

    dt = bb.read_uint8(pos + o_dt) if o_dt else 0
    num = bb.read_int32(pos + o_num) if o_num else None
    time_v = bb.read_float64(pos + o_time) if o_time else None
    pid = bb.string(pos + o_pid) if o_pid else None
    data_pos = bb.indirect(pos + o_data) if o_data else None
    return GameEvent(dataType=dt, num=num, time=time_v,
                     playId=pid, data_pos=data_pos, _bb=bb)


@dataclass
class TrackedEvent:
    dataType: int
    num: Optional[int]
    time: Optional[float]
    playId: Optional[str]
    data_pos: Optional[int]
    _bb: Optional["ByteBuffer"] = None


def read_tracked_event(bb: ByteBuffer, pos: int) -> TrackedEvent:
    # Same vtable shape as GameEventWire (confirm from extract output).
    o_dt = bb.field_offset(pos, 4)
    o_data = bb.field_offset(pos, 6)
    o_num = bb.field_offset(pos, 8)
    o_time = bb.field_offset(pos, 10)
    o_pid = bb.field_offset(pos, 12)

    dt = bb.read_uint8(pos + o_dt) if o_dt else 0
    num = bb.read_int32(pos + o_num) if o_num else None
    time_v = bb.read_float64(pos + o_time) if o_time else None
    pid = bb.string(pos + o_pid) if o_pid else None
    data_pos = bb.indirect(pos + o_data) if o_data else None
    return TrackedEvent(dataType=dt, num=num, time=time_v,
                        playId=pid, data_pos=data_pos, _bb=bb)
```

**Important:** the offsets above are educated guesses. Before declaring this task done, open `docs/superpowers/plans/2026-05-10-event-ingestion-offsets.md` and confirm each field offset for `GameEventWire` / `TrackedEventWire` matches what's documented there. If they differ, EDIT the dataclass `read_*` functions to match the doc, and update the comment block above the dataclass to reflect reality.

- [ ] **Step 4: Extend `TrackingFrame` to populate `gameEvents` and `trackedEvents`**

Edit the `TrackingFrame` dataclass and `read_tracking_frame` in `wire_schemas.py`. Add fields:

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

In `read_tracking_frame`, after the existing `inferred = ...` block and before `return TrackingFrame(...)`, add:

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

And update the `return` to include them:

```python
    return TrackingFrame(num, time_v, timestamp, isGap, gap_dur, ballPos,
                         actors, raw, inferred, game_events, tracked_events)
```

- [ ] **Step 5: Run the tests to verify pass**

```bash
$PY -m pytest tests/test_event_schemas.py -v
```

Expected: both tests pass. If they fail with `assert found_any`, the fixture choice in Task 4 was wrong — re-pick. If they fail with attribute errors, your edits to `TrackingFrame` aren't consistent.

- [ ] **Step 6: Commit**

```bash
git add src/fieldvision/wire_schemas.py tests/test_event_schemas.py
git commit -m "Decode gameEvents/trackedEvents wrapper tables (GameEventWire, TrackedEventWire)"
```

---

## Task 6: Implement event-data decoders for the highest-value types

**Files:**
- Modify: `src/fieldvision/wire_schemas.py`

We add decoders for `BallPitchDataWire`, `PlayEventDataWire`, `CountEventDataWire`, `AtBatEventDataWire`, `HandedEventDataWire`, `BatImpactEventDataWire`. Each gets a dataclass + `read_*` function. Field offsets come from the offsets doc produced in Task 2.

For each event class, follow this exact pattern:

- [ ] **Step 1: Write failing test for BallPitchDataWire**

Append to `tests/test_event_schemas.py`:

```python
def test_ball_pitch_decodes_plausible_values():
    """Find at least one BallPitchDataWire in our pitch fixture and verify
    velocity/spin are in the human-plausible MLB range."""
    from fieldvision.wire_schemas import (
        TRACKED_EVENT_DISPATCH, decode_tracked_event_data, BallPitchData,
    )
    game_pk, seg = FIXTURE_SEG_WITH_TRACKED_EVENTS
    td = read_tracking_data(fixture_bin_path(game_pk, seg).read_bytes())

    pitches = []
    for f in td.frames:
        for te in f.trackedEvents:
            data = decode_tracked_event_data(te)
            if isinstance(data, BallPitchData):
                pitches.append(data)
    assert pitches, "expected at least one BallPitchData in fixture"

    p = pitches[0]
    # Release point: roughly 6 ft tall, 50-60 ft from home plate
    assert p.releasePosition is not None
    # Velocity range: 50-110 mph (or m/s; this asserts whichever matches)
    assert (50 < (p.velocityRelease or 0) < 110) or \
        (22 < (p.velocityRelease or 0) < 50), \
        f"velocity {p.velocityRelease} not plausible for mph or m/s"
```

- [ ] **Step 2: Run to verify failure**

```bash
$PY -m pytest tests/test_event_schemas.py::test_ball_pitch_decodes_plausible_values -v
```

Expected: ImportError on `BallPitchData` etc.

- [ ] **Step 3: Add the BallPitchData dataclass + reader**

Append to `wire_schemas.py`. **Use the actual offsets from your offsets doc; this is illustrative.** Common BallPitch fields based on PROJECT_PREDICTION.md:

```python
# ────────────────────────────────────────────────────────────
# BallPitchDataWire — pitch release/trajectory/spin metadata.
# Look up actual field order in the offsets doc; common fields:
#   - pitchType (string or uint enum)
#   - releasePosition (Vec3 inline)
#   - velocityRelease (float32, mph)
#   - velocityPlate (float32)
#   - spinRate (float32, rpm)
#   - spinAxis (Vec3 inline)
#   - extension (float32, ft)
#   - platePosition (Vec3 inline)
# ────────────────────────────────────────────────────────────


@dataclass
class BallPitchData:
    pitchType: Optional[str]
    pitchTypeId: Optional[int]
    releasePosition: Optional[Vec3]
    platePosition: Optional[Vec3]
    velocityRelease: Optional[float]
    velocityPlate: Optional[float]
    spinRate: Optional[float]
    spinAxis: Optional[Vec3]
    extension: Optional[float]


def read_ball_pitch_data(bb: ByteBuffer, pos: int) -> BallPitchData:
    # Replace these field-offset numbers with the values from
    # docs/superpowers/plans/2026-05-10-event-ingestion-offsets.md
    o_type = bb.field_offset(pos, 4)
    o_type_id = bb.field_offset(pos, 6)
    o_rel = bb.field_offset(pos, 8)
    o_plate = bb.field_offset(pos, 10)
    o_vrel = bb.field_offset(pos, 12)
    o_vplate = bb.field_offset(pos, 14)
    o_spin = bb.field_offset(pos, 16)
    o_axis = bb.field_offset(pos, 18)
    o_ext = bb.field_offset(pos, 20)

    pitch_type = bb.string(pos + o_type) if o_type else None
    pitch_type_id = bb.read_uint8(pos + o_type_id) if o_type_id else None
    rel = read_vec3(bb, pos + o_rel) if o_rel else None
    plate = read_vec3(bb, pos + o_plate) if o_plate else None
    vrel = bb.read_float32(pos + o_vrel) if o_vrel else None
    vplate = bb.read_float32(pos + o_vplate) if o_vplate else None
    spin = bb.read_float32(pos + o_spin) if o_spin else None
    axis = read_vec3(bb, pos + o_axis) if o_axis else None
    ext = bb.read_float32(pos + o_ext) if o_ext else None

    return BallPitchData(pitchType=pitch_type, pitchTypeId=pitch_type_id,
                         releasePosition=rel, platePosition=plate,
                         velocityRelease=vrel, velocityPlate=vplate,
                         spinRate=spin, spinAxis=axis, extension=ext)
```

- [ ] **Step 4: Add the union dispatchers**

Below the BallPitchData reader, append:

```python
# Mapping derived from the offsets doc (`extract_event_offsets.py` output).
# Update these integer keys to match what your doc says.
TRACKED_EVENT_DISPATCH = {
    1: ("BallPitch", read_ball_pitch_data),
    # 2: ("BallHit",   read_ball_hit_data),     # deferred to Phase A.2
    # 3: ("BallThrow", read_ball_throw_data),
    # 4: ("BallBounce", read_ball_bounce_data),
}


def decode_tracked_event_data(te: TrackedEvent):
    """Resolve te.data_pos to the typed event-data dataclass. Returns
    None if dataType isn't in our dispatch (e.g. event types we haven't
    decoded yet)."""
    handler = TRACKED_EVENT_DISPATCH.get(te.dataType)
    if handler is None or te.data_pos is None or te._bb is None:
        return None
    _name, reader = handler
    return reader(te._bb, te.data_pos)


def tracked_event_type_name(te: TrackedEvent) -> str:
    """String name of the tracked event type, or 'Unknown:<dataType>'."""
    handler = TRACKED_EVENT_DISPATCH.get(te.dataType)
    return handler[0] if handler else f"Unknown:{te.dataType}"
```

- [ ] **Step 5: Run BallPitch test**

```bash
$PY -m pytest tests/test_event_schemas.py::test_ball_pitch_decodes_plausible_values -v
```

Expected: PASS. If fail with `expected at least one BallPitchData`, the dispatch key is wrong — re-check the offsets doc's TrackedEvent union table. If fail on velocity range, the field offset is wrong — try the next slot, the value should be plausible at the right offset.

- [ ] **Step 6: Commit**

```bash
git add src/fieldvision/wire_schemas.py tests/test_event_schemas.py
git commit -m "Decode BallPitchDataWire (release point, velocity, spin, pitch type)"
```

- [ ] **Step 7: Repeat the same pattern for the remaining event-data types**

The next sub-tasks add 5 game-event decoders. Before writing any of them, add the
parallel game-event dispatch infrastructure (mirrors what Step 4 added for tracked
events):

```python
GAME_EVENT_DISPATCH: dict[int, tuple[str, "callable"]] = {}  # filled in below


def decode_game_event_data(ge: GameEvent):
    handler = GAME_EVENT_DISPATCH.get(ge.dataType)
    if handler is None or ge.data_pos is None or ge._bb is None:
        return None
    _name, reader = handler
    return reader(ge._bb, ge.data_pos)


def game_event_type_name(ge: GameEvent) -> str:
    handler = GAME_EVENT_DISPATCH.get(ge.dataType)
    return handler[0] if handler else f"Unknown:{ge.dataType}"
```

Then for each of the 5 classes below, add (in order, with a separate commit per class):
1. A failing test in `tests/test_event_schemas.py` (asserting the type appears in a fixture frame + one plausible-value check)
2. The dataclass + reader function in `wire_schemas.py`
3. An entry in `GAME_EVENT_DISPATCH` (the integer key comes from the offsets doc — Task 2 Step 3)
4. Re-run pytest: the new test should pass
5. Commit with message `Decode <ClassName>`

The classes:

a. **`PlayEventDataWire`** — has `playId`, strikezone (`szTop`, `szBottom`, `szLeft`, `szRight` floats), `batterId`, `pitcherId`, `pitchNumber`. Test asserts `0 < szTop < 5` and `0 < szBottom < szTop`.

b. **`CountEventDataWire`** — has `balls`, `strikes`, `outs` (small uints 0-4). Test asserts all in [0, 4].

c. **`AtBatEventDataWire`** — has `atBatIndex`, `batterId`, `pitcherId`. Test asserts batter and pitcher are >= 0 (could be 0 if absent).

d. **`HandedEventDataWire`** — has `batterHandedness` and `pitcherHandedness` (each is a uint enum or string; if enum, common values are L=0, R=1, S=2). Test asserts each handedness when present is in {'L', 'R', 'S'}.

e. **`BatImpactEventDataWire`** — has `playId`, `position` (Vec3 inline). Test asserts `position.y > 0` (above ground).

After all five are committed individually, you'll have 6 total event-data decoders.

---

## Task 7: Wire the union dispatchers into ingest_segment

**Files:**
- Modify: `src/fieldvision/storage.py:304-382` (the `ingest_segment` function and surrounding helpers)

We extend `ingest_segment` to:
- Iterate `f.gameEvents` and write a row per event into `game_event` (with parsed columns when the event type is known).
- Iterate `f.trackedEvents`, decode each, and write a row into `pitch_event` for `BallPitchData` instances.

- [ ] **Step 1: Write a failing test**

Create `tests/test_ingest_segment.py`:

```python
"""Test that ingest_segment populates game_event and pitch_event."""

import sqlite3
from pathlib import Path

from fieldvision.storage import (
    _actor_frame_insert_sql,
    ingest_segment,
    open_game_db,
    transaction,
)
from tests.conftest import FIXTURE_SEG_WITH_TRACKED_EVENTS, fixture_bin_path


def test_ingest_writes_pitch_events(tmp_path: Path):
    game_pk, seg_idx = FIXTURE_SEG_WITH_TRACKED_EVENTS
    bin_path = fixture_bin_path(game_pk, seg_idx)

    conn = open_game_db(game_pk, tmp_path)
    insert_sql = _actor_frame_insert_sql()
    # No labels needed for events; pass empty dict.
    with transaction(conn):
        ingest_segment(conn, game_pk, seg_idx, bin_path, {}, insert_sql)

    n_pitch = conn.execute("SELECT COUNT(*) FROM pitch_event").fetchone()[0]
    n_game = conn.execute("SELECT COUNT(*) FROM game_event").fetchone()[0]
    assert n_pitch >= 1, "expected at least one pitch_event row"
    assert n_game >= 1, "expected at least one game_event row"

    # Sanity: pitch row has plausible release point.
    rx, ry, rz = conn.execute(
        "SELECT release_x, release_y, release_z FROM pitch_event "
        "WHERE release_y IS NOT NULL LIMIT 1"
    ).fetchone() or (None, None, None)
    if ry is not None:
        assert 3.0 < ry < 8.0, f"release_y {ry} not plausible (3-8 ft)"
```

- [ ] **Step 2: Run to verify failure**

```bash
$PY -m pytest tests/test_ingest_segment.py -v
```

Expected: assertion failure — `n_pitch == 0` because ingest doesn't write events yet.

- [ ] **Step 3: Update `ingest_segment` to write events**

In `src/fieldvision/storage.py`, modify `ingest_segment`:

1. Add at the top of the file (with other imports):
   ```python
   from .wire_schemas import (
       BallPitchData, GAME_EVENT_DISPATCH, TRACKED_EVENT_DISPATCH,
       decode_game_event_data, decode_tracked_event_data,
       game_event_type_name, read_tracking_data, tracked_event_type_name,
       unpack_smallest_three,
   )
   ```
   (Add only the names that exist after Task 6.)

2. Change the function signature to accept the new SQL strings (so the daemon can build them once):
   ```python
   def ingest_segment(
       conn: sqlite3.Connection,
       game_pk: int,
       segment_idx: int,
       bin_path: Path,
       labels_dict: dict[int, dict],
       insert_sql: str,
       game_event_insert_sql: str | None = None,
       pitch_event_insert_sql: str | None = None,
   ) -> tuple[int, int]:
   ```
   When `None` is passed, build them locally as a fallback.

3. After the existing `for f in td.frames:` loop body (after actor poses), add event collection. Add three list initializations near the top of the function:
   ```python
   game_event_rows: list[tuple] = []
   pitch_event_rows: list[tuple] = []
   ```

4. Inside the frame loop, after the actorPoses block, add:
   ```python
   # Game events
   for ge in f.gameEvents:
       evt_type = game_event_type_name(ge)
       data = decode_game_event_data(ge)
       balls = strikes = outs = inning = top_inning = None
       batter_id = pitcher_id = None
       bhand = phand = None
       data_json_str = None
       if data is not None:
           # Pull common columns when the parsed object has them.
           balls = getattr(data, "balls", None)
           strikes = getattr(data, "strikes", None)
           outs = getattr(data, "outs", None)
           inning = getattr(data, "inning", None)
           top_inning = getattr(data, "topInning", None)
           batter_id = getattr(data, "batterId", None)
           pitcher_id = getattr(data, "pitcherId", None)
           bhand = getattr(data, "batterHandedness", None)
           phand = getattr(data, "pitcherHandedness", None)
       game_event_rows.append((
           game_pk, segment_idx, f.num, time_unix,
           evt_type, ge.dataType, 0, ge.playId,
           balls, strikes, outs,
           inning, top_inning,
           batter_id, pitcher_id,
           bhand, phand,
           data_json_str,
       ))

   # Tracked events — only BallPitchData currently lands in pitch_event.
   for te in f.trackedEvents:
       data = decode_tracked_event_data(te)
       if isinstance(data, BallPitchData):
           rel = data.releasePosition
           plate = data.platePosition
           axis = data.spinAxis
           pitch_event_rows.append((
               game_pk, segment_idx, f.num, time_unix,
               te.playId,
               rel.x if rel else None, rel.y if rel else None, rel.z if rel else None,
               plate.x if plate else None, plate.y if plate else None, plate.z if plate else None,
               data.velocityRelease, data.velocityPlate,
               data.spinRate,
               axis.x if axis else None, axis.y if axis else None, axis.z if axis else None,
               data.pitchType, data.pitchTypeId, data.extension,
               # Strikezone fields default None here — they come from PlayEvent
               # (joined later by play_id, or in a follow-up that backfills).
               None, None, None, None,
           ))
   ```

5. After the loop, build SQL strings if not provided and execute the inserts:
   ```python
   ge_sql = game_event_insert_sql or _game_event_insert_sql()
   pe_sql = pitch_event_insert_sql or _pitch_event_insert_sql()
   if game_event_rows:
       conn.executemany(ge_sql, game_event_rows)
   if pitch_event_rows:
       conn.executemany(pe_sql, pitch_event_rows)
   ```

- [ ] **Step 4: Run the ingest test**

```bash
$PY -m pytest tests/test_ingest_segment.py -v
```

Expected: PASS. If `release_y` is way off, the BallPitch field offsets in Task 6 are wrong — fix and re-run.

- [ ] **Step 5: Update load_to_db.py to clear new tables on `--rebuild`**

In `scripts/load_to_db.py`, find the `if args.rebuild:` block (around line 51) and add `pitch_event` to the table list:

```python
        for table in ("actor_frame", "ball_frame", "bat_frame",
                      "game_event", "pitch_event",
                      "labels", "bones", "players", "meta"):
            conn.execute(f"DELETE FROM {table}")
```

Also pass the prebuilt SQLs into ingest_segment to avoid rebuilding them per segment:

```python
from fieldvision.storage import (_actor_frame_insert_sql, _game_event_insert_sql,
                                 _pitch_event_insert_sql,
                                 ingest_segment, load_lookup_tables, open_game_db,
                                 open_registry, transaction, update_registry)
...
    insert_sql = _actor_frame_insert_sql()
    ge_sql = _game_event_insert_sql()
    pe_sql = _pitch_event_insert_sql()
...
            n_actor, n_ball = ingest_segment(conn, args.game, seg_idx,
                                             bin_path, labels_dict,
                                             insert_sql, ge_sql, pe_sql)
```

Same change in `scripts/fv_daemon.py:239,244` (the live daemon must use the new path too):

```python
    insert_sql = _actor_frame_insert_sql()
    ge_sql = _game_event_insert_sql()
    pe_sql = _pitch_event_insert_sql()
    ...
    with transaction(conn):
        ingest_segment(conn, game_pk, i, bin_path, labels_dict,
                       insert_sql, ge_sql, pe_sql)
```

(And add the imports at the top of fv_daemon.py.)

- [ ] **Step 6: Commit**

```bash
git add src/fieldvision/storage.py scripts/load_to_db.py scripts/fv_daemon.py tests/test_ingest_segment.py
git commit -m "Ingest game_event + pitch_event rows from gameEvents/trackedEvents"
```

---

## Task 8: Re-ingest sample game 823141 end-to-end

**Files:** none modified — this is validation.

We rebuild the per-game DB for `823141` from on-disk samples and confirm the new tables populate.

- [ ] **Step 1: Backup the existing DB**

```bash
cp data/fv_823141.sqlite data/fv_823141.sqlite.pre-events-backup
```

- [ ] **Step 2: Re-ingest**

```bash
$PY scripts/load_to_db.py --game 823141 --rebuild
```

Expected: progress lines for each segment, finishing with sanity-query output. Watch for any error rows.

- [ ] **Step 3: Inspect the new tables**

```bash
sqlite3 data/fv_823141.sqlite "SELECT COUNT(*) FROM pitch_event;"
sqlite3 data/fv_823141.sqlite "SELECT event_type, COUNT(*) FROM game_event GROUP BY event_type ORDER BY 2 DESC;"
sqlite3 data/fv_823141.sqlite "SELECT pitch_type, COUNT(*), AVG(velocity_release), AVG(spin_rate) FROM pitch_event GROUP BY pitch_type ORDER BY 2 DESC LIMIT 20;"
sqlite3 data/fv_823141.sqlite "SELECT release_x, release_y, release_z, velocity_release, pitch_type FROM pitch_event LIMIT 5;"
```

Expected:
- `pitch_event` count in the 200-400 range (an MLB game has ~250-330 pitches; the daemon can also produce a fragment of that if 823141 was partial).
- `game_event` types include `PlayEvent`, `CountEvent`, `AtBatEvent`, possibly others.
- Pitch types are recognizable codes (`FF`, `SL`, `CH`, etc.) and average velocities are 70-100 mph (or 30-45 m/s).

If pitch_event count is implausibly low or zero, debug:
```bash
$PY scripts/decode_segment.py samples/binary_capture_823141/mlb_823141_segment_500.bin | grep -i event
```

- [ ] **Step 4: Verify no regression on existing tables**

```bash
sqlite3 data/fv_823141.sqlite "SELECT COUNT(*) FROM actor_frame; SELECT COUNT(*) FROM bat_frame;"
diff <(sqlite3 data/fv_823141.sqlite.pre-events-backup "SELECT COUNT(*) FROM actor_frame") \
     <(sqlite3 data/fv_823141.sqlite "SELECT COUNT(*) FROM actor_frame")
```

Expected: no diff on actor_frame count (we didn't change the actor pipeline).

- [ ] **Step 5: Commit nothing (validation only) — but record the result**

If everything looks right, append a sentence to your scratch notes (or the offsets doc):
> "Re-ingest of 823141 produced N pitch_event rows, M game_event rows."

---

## Task 9: Sanity-check pitch count against MLB statsapi

**Files:**
- Create: `scripts/validate_pitch_count.py`

statsapi exposes the canonical pitch count for any game. We compare to our `pitch_event` count and flag if the gap is > 5%.

- [ ] **Step 1: Write the validator**

Create `scripts/validate_pitch_count.py`:

```python
"""Compare pitch_event row count to MLB statsapi's canonical pitch count.

Usage:
    python scripts/validate_pitch_count.py --game 823141
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path


def statsapi_pitch_count(game_pk: int) -> int:
    """Count pitches across allPlays.playEvents where event isPitch."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh) FieldVision-validator/1.0",
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    n = 0
    for play in plays:
        for ev in play.get("playEvents", []):
            if ev.get("isPitch"):
                n += 1
    return n


def db_pitch_count(game_pk: int, data_dir: Path) -> int:
    db = data_dir / f"fv_{game_pk}.sqlite"
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM pitch_event").fetchone()[0]
    conn.close()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-diff-pct", type=float, default=5.0)
    args = ap.parse_args()

    n_db = db_pitch_count(args.game, Path(args.data_dir))
    n_sa = statsapi_pitch_count(args.game)
    if n_sa == 0:
        print(f"statsapi reports 0 pitches for {args.game} — game likely not yet started or missing.")
        return 0
    diff = abs(n_db - n_sa) / n_sa * 100
    print(f"game {args.game}: db pitch_event = {n_db}, statsapi pitches = {n_sa} "
          f"(diff {diff:.1f}%)")
    if diff > args.max_diff_pct:
        print(f"  ⚠ diff exceeds {args.max_diff_pct}% threshold")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

```bash
$PY scripts/validate_pitch_count.py --game 823141
```

Expected: db count within 5% of statsapi count. A ~5-10% gap is acceptable (some pitches may not have made it into the .bin segments captured live; bulk historical games should be tighter).

If diff is huge (> 50%): the BallPitchData decoder is mis-matching frequency. Most likely causes:
- We're double-counting (each pitch appears in multiple frames as it travels). FIX: dedupe by `play_id` before insert.
- We're under-counting because the wrong dataType is mapped. FIX: re-check the trackedEvents dispatch table.

- [ ] **Step 3: If diff is from double-counting, dedupe by play_id**

If Step 2 shows roughly N×M pitches where N is real and M ~10-30, modify the pitch loop in `ingest_segment` to keep only the *first* frame for each play_id within a segment:

```python
seen_play_ids = set()
for te in f.trackedEvents:
    ...
    if isinstance(data, BallPitchData):
        if te.playId and te.playId in seen_play_ids:
            continue
        if te.playId:
            seen_play_ids.add(te.playId)
        pitch_event_rows.append((...))
```

(Note: `seen_play_ids` may need to live OUTSIDE the per-frame loop and be a function-level set across all frames. Test passing/failing tells you which.)

Re-run Task 8 (re-ingest) and Task 9 (validate) until diff is under 5%.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_pitch_count.py
git commit -m "Validate pitch_event count vs statsapi"
```

---

## Task 10: Run the full pytest suite + clean lint check

**Files:** none modified — final gate.

- [ ] **Step 1: Run all tests**

```bash
$PY -m pytest tests/ -v
```

Expected: all green. If any test fails, fix before merging.

- [ ] **Step 2: Verify the daemon-restart path works**

The launchd daemon is still running the OLD code (before our changes). It will keep working against the existing schema (we only ADDED tables/columns). When Spencer restarts it, the new ingest path activates.

Confirm the new code passes the daemon's import sanity:

```bash
$PY -c "from fieldvision.storage import ingest_segment, _game_event_insert_sql, _pitch_event_insert_sql; print('import ok')"
```

Expected: `import ok` and no traceback.

- [ ] **Step 3: Don't restart the daemon yet — that's a Spencer decision**

The daemon restart is a deploy. Do not run `launchctl unload/load` without Spencer's explicit OK. Add a note to the eventual completion message that says:
> "When you're ready to start ingesting events from live games, restart the daemon: `launchctl unload ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist && launchctl load ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist`"

---

## Task 11: Update CLAUDE.md to reflect new ingestion

**Files:**
- Modify: `CLAUDE.md`

Two sections need updating:

- [ ] **Step 1: Update "NOT YET INGESTED but available" section**

In `CLAUDE.md` find the heading `### NOT YET INGESTED but available in the wire format` (around line 113). Move the now-ingested items out — leave only what we deferred. The new list should be:

```markdown
### NOT YET INGESTED but available in the wire format (Phase A.2+)

`TrackingFrameWire` ALSO contains, per frame, fields we currently skip:

- Less-critical `gameEvents[]`:
  - `InningEventDataWire`
  - `PositionAssignmentEventDataWire`
  - `ABSEventDataWire`
  - `TeamScoreEventDataWire`
  - `BattingOrderEventDataWire`
  - `LiveActionEventDataWire`, `GumboTimecodeEventDataWire`, `StatusEventDataWire`
  - `HighFrequencyBatMarkerEventDataWire`
- Less-critical `trackedEvents[]`:
  - `BallHitDataWire`
  - `BallThrowData`
  - `BallBounceDataWire`
- `ballPolynomials[]` — pitch trajectory polynomials

**As of <today's date> we ingest `gameEvents` of type Play/Count/AtBat/Handed/BatImpact, and `trackedEvents` of type BallPitch.**
```

- [ ] **Step 2: Update "Open / deferred work" section**

Find the numbered list and update item 3:

```markdown
3. **Ingest remaining `gameEvents` + `trackedEvents` + `ballPolynomials`** —
   Phase A.1 (Play/Count/AtBat/Handed/BatImpact + BallPitch) is done as of
   <today's date>. The rest are nice-to-have for richer features.
```

- [ ] **Step 3: Add a short pointer to the new tables in the data-model section**

Find the heading `### Per-game SQLite (`data/fv_<gamePk>.sqlite`)` and append, after the `bat_frame` description:

```markdown
**`game_event`** — discrete game events tied to a frame. Columns include
`event_type`, `play_id`, `balls`, `strikes`, `outs`, `inning`, `top_inning`,
`batter_id`, `pitcher_id`, `batter_handedness`, `pitcher_handedness`.
Index on (event_type, time_unix) and on play_id.

**`pitch_event`** — per-pitch metadata (release point, plate position,
velocity at release/plate, spin rate + axis, pitch type, extension).
Joins back to `game_event` on `play_id`. Strike-zone columns are nullable;
populated by joining to PlayEvent (a follow-up may backfill).
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "CLAUDE.md: document game_event + pitch_event ingestion"
```

---

## Task 12: Merge feature branch and offer next steps

**Files:** none — git operations only.

- [ ] **Step 1: Final review of changed files**

```bash
git log --oneline main..HEAD
git diff --stat main..HEAD
```

Expected: roughly 8-10 commits from this plan.

- [ ] **Step 2: Merge to main**

```bash
git checkout main
git merge --no-ff feat/event-ingestion -m "Merge: gameEvents + trackedEvents ingestion (Phase A.1)"
```

- [ ] **Step 3: Tell Spencer about the daemon-restart step**

Hand back to the user:

> "Phase A.1 merged to main. New tables `game_event` and `pitch_event` are populated for game 823141 (validated against statsapi). To start capturing events from live games, restart the daemon:
>
> ```
> launchctl unload ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist
> launchctl load   ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist
> launchctl list | grep fvcapture       # confirm new PID
> tail -f scheduler.log
> ```
>
> Existing per-game DBs (other than 823141) won't have backfilled events until each is re-ingested via `python scripts/load_to_db.py --game <pk> --rebuild`. Worth doing for the Guardians backlog before retention expires those .bin files."

---

## Summary of files at end of plan

**New:**
- `scripts/snippets/fetch_bundle.js`
- `scripts/extract_event_offsets.py`
- `scripts/validate_pitch_count.py`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_storage_schema.py`
- `tests/test_event_schemas.py`
- `tests/test_ingest_segment.py`
- `docs/superpowers/plans/2026-05-10-event-ingestion-offsets.md` (output of Task 2)

**Modified:**
- `src/fieldvision/wire_schemas.py` (~250 lines added: 6 event-data dataclasses + readers, 2 wrapper readers, 2 dispatchers, TrackingFrame extension)
- `src/fieldvision/storage.py` (~80 lines added: schema rewrite, 2 SQL helpers, ingest_segment extension)
- `scripts/load_to_db.py` (~10 lines: rebuild list + new SQL imports)
- `scripts/fv_daemon.py` (~5 lines: pass new SQLs to ingest_segment)
- `CLAUDE.md` (data model + open-work section)
- `.gitignore` (one line: bundle file)

**Not touched:** `skeleton.py`, `flatbuf_runtime.py` (sufficient as-is), the daemon's main loop, render/scrape scripts.
