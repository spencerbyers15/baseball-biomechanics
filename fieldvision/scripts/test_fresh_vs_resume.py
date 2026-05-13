"""Compare a fresh scrape vs. a resume scrape of the same Final game.

Run A: scrape from scratch into /tmp/fv_test_a/data
Run B: pre-populate /tmp/fv_test_b/data with the first half of A's data,
       then run scrape_one_game which should RESUME, fetching only the
       back half. Writes go to a suffixed file alongside the pre-populated
       canonical file.

At the end:
- Print timings
- Compare total row counts and segment coverage between A and B
- Compare the set of (segment_idx, frame_num, actor_uid) keys for equality
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PK = int(sys.argv[1]) if len(sys.argv) > 1 else 824438
TOKEN = (REPO_ROOT / ".fv_token.txt").read_text().strip()

TMP = Path(
    os.environ.get("FV_TEST_TMP", f"/media/scratch/spencer/data/fieldvision/test/fv_test_{PK}")
)
DIR_A = TMP / "a"
DIR_B = TMP / "b"

# Wipe + recreate
if TMP.exists():
    shutil.rmtree(TMP)
(DIR_A / "data").mkdir(parents=True)
(DIR_A / "samples").mkdir(parents=True)
(DIR_A / "state").mkdir(parents=True)
(DIR_B / "data").mkdir(parents=True)
(DIR_B / "samples").mkdir(parents=True)
(DIR_B / "state").mkdir(parents=True)


def configure_dirs(base: Path):
    """Point the scrape_team_history module's globals at this tree."""
    os.environ["FV_DATA_DIR"] = str(base / "data")
    os.environ["FV_SAMPLES_DIR"] = str(base / "samples")
    os.environ["FV_STATE_DIR"] = str(base / "state")
    # Reload so module-level constants re-read the env
    import importlib
    if "scrape_team_history" in sys.modules:
        importlib.reload(sys.modules["scrape_team_history"])
    import scrape_team_history
    return scrape_team_history


# ─── Run A: fresh scrape ────────────────────────────────────────────
print(f"=== Run A: fresh scrape of pk={PK} ===")
sth = configure_dirs(DIR_A)
t0 = time.time()
result_a = sth.scrape_one_game(PK, TOKEN, delete_bins=True)
t_a = time.time() - t0
print(f"Run A: {t_a:.1f}s  ({t_a/60:.1f} min)")
print(f"  result: {result_a}")

a_path = DIR_A / "data" / str(PK) / "actor_frames.parquet"
con = duckdb.connect()
a_rows, a_segs, a_max = con.execute(
    f"SELECT COUNT(*), COUNT(DISTINCT segment_idx), MAX(segment_idx) "
    f"FROM read_parquet('{a_path}')"
).fetchone()
print(f"  Run A parquet: {a_rows:,} rows, {a_segs} distinct segments, max_idx {a_max}")

# ─── Pre-populate B with the first half of A ────────────────────────
half = a_max // 2
print(f"\n=== prep B: copy A's segments 0..{half} into B ===")
b_game_dir = DIR_B / "data" / str(PK)
b_game_dir.mkdir(parents=True, exist_ok=True)

# Copy actor_frames truncated to first half
con.execute(
    f"COPY (SELECT * FROM read_parquet('{a_path}') WHERE segment_idx <= {half}) "
    f"TO '{b_game_dir / 'actor_frames.parquet'}' "
    f"(FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 100000)"
)
# Copy the other small tables verbatim so the schema is identical
for fname in ("bat_frames", "ball_frames", "pitch_events",
              "bones", "labels", "meta", "players"):
    src = DIR_A / "data" / str(PK) / f"{fname}.parquet"
    if not src.exists(): continue
    if fname in ("bat_frames", "ball_frames", "pitch_events"):
        # Trim to same time range so the comparison is fair
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{src}') WHERE segment_idx <= {half}) "
            f"TO '{b_game_dir / src.name}' "
            f"(FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 100000)"
        )
    else:
        shutil.copy(src, b_game_dir / src.name)

b_pre_rows = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{b_game_dir / 'actor_frames.parquet'}')"
).fetchone()[0]
print(f"  pre-populated {b_pre_rows:,} rows ({half}/{a_max} segments)")

# ─── Run B: resume scrape ────────────────────────────────────────────
print(f"\n=== Run B: resume scrape of pk={PK} ===")
sth = configure_dirs(DIR_B)
t0 = time.time()
result_b = sth.scrape_one_game(PK, TOKEN, delete_bins=True)
t_b = time.time() - t0
print(f"Run B: {t_b:.1f}s  ({t_b/60:.1f} min)")
print(f"  result: {result_b}")

# ─── Compare ─────────────────────────────────────────────────────────
print(f"\n=== compare ===")

# Union A's files vs union B's files
a_files = sorted((DIR_A / "data" / str(PK)).glob("actor_frames*.parquet"))
b_files = sorted((DIR_B / "data" / str(PK)).glob("actor_frames*.parquet"))
print(f"  Run A actor_frames files: {[f.name for f in a_files]}")
print(f"  Run B actor_frames files: {[f.name for f in b_files]}")

def union_count(files):
    files_sql = "[" + ", ".join(f"'{f.as_posix()}'" for f in files) + "]"
    return con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT segment_idx), MIN(segment_idx), MAX(segment_idx) "
        f"FROM read_parquet({files_sql}, union_by_name=true)"
    ).fetchone()

a_stats = union_count(a_files)
b_stats = union_count(b_files)
print(f"  Run A union: {a_stats[0]:,} rows | {a_stats[1]} segs | range {a_stats[2]}..{a_stats[3]}")
print(f"  Run B union: {b_stats[0]:,} rows | {b_stats[1]} segs | range {b_stats[2]}..{b_stats[3]}")

# Compare key sets (segment_idx, frame_num, actor_uid)
a_files_sql = "[" + ", ".join(f"'{f.as_posix()}'" for f in a_files) + "]"
b_files_sql = "[" + ", ".join(f"'{f.as_posix()}'" for f in b_files) + "]"
diff = con.execute(f"""
    WITH
      a AS (SELECT segment_idx, frame_num, actor_uid FROM read_parquet({a_files_sql}, union_by_name=true)),
      b AS (SELECT segment_idx, frame_num, actor_uid FROM read_parquet({b_files_sql}, union_by_name=true)),
      a_only AS (SELECT * FROM a EXCEPT SELECT * FROM b),
      b_only AS (SELECT * FROM b EXCEPT SELECT * FROM a)
    SELECT
      (SELECT COUNT(*) FROM a_only) AS a_only,
      (SELECT COUNT(*) FROM b_only) AS b_only
""").fetchone()
print(f"  rows only in A: {diff[0]:,}")
print(f"  rows only in B: {diff[1]:,}")

# Also hash the joint columns to make sure the actual NUMBERS match, not
# just the keys.
checksum = con.execute(f"""
    SELECT
      SUM(pelvis_x * 1e6 + pelvis_y * 1e3 + pelvis_z) AS sx
    FROM read_parquet({a_files_sql}, union_by_name=true)
""").fetchone()[0]
checksum_b = con.execute(f"""
    SELECT
      SUM(pelvis_x * 1e6 + pelvis_y * 1e3 + pelvis_z) AS sx
    FROM read_parquet({b_files_sql}, union_by_name=true)
""").fetchone()[0]
print(f"  pelvis checksum A: {checksum}")
print(f"  pelvis checksum B: {checksum_b}")
print(f"  match: {abs(checksum - checksum_b) < 1e-3}")

print(f"\n=== summary ===")
print(f"  Fresh : {t_a:.1f}s  ({a_stats[0]:,} rows)")
print(f"  Resume: {t_b:.1f}s  ({b_stats[0]:,} rows)")
speedup = t_a / max(t_b, 0.1)
print(f"  Speedup: {speedup:.1f}x")
identical = (a_stats == b_stats) and (diff[0] == diff[1] == 0)
print(f"  Data identical: {identical}")
