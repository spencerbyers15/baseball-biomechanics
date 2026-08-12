"""Optional retention pruner for per-game Parquet stores.

No-op by default. Every cut is opt-in via flag. Designed for "the NAS
is filling up; condense the corpus" scenarios — not as part of the
standard ingest pipeline.

Each cut operates on actor_frames.parquet only (the biggest table).
Other tables (bat_frames, ball_frames, pitch_label, pitch_event, ...)
are untouched and still describe the full play, so dropped frames can
be re-scraped from MLB if their retention window is still open.

Cuts (all default OFF; can combine):

  --drop-umps-coaches
      Drop every actor row with actor_type in {umpire, plate-umpire,
      coach}. Always safe — these add no biomechanics signal.

  --drop-fielders-non-x
      Drop fielder rows whose timestamp falls inside a non-X pitch's
      [start_time_unix, end_time_unix] window. "X" = ball in play.
      Pitcher/catcher/batter are always kept.

  --drop-fielders-no-touch-post-release
      For X pitches only, drop fielder rows AFTER the pitch's release
      (start_time_unix) whose hand-to-ball minimum distance never
      drops below --touch-radius (default 2.0 ft). Pre-release
      positioning frames are preserved so defensive shifts remain
      analyzable.

Usage:
  python scripts/prune_game.py --game 823141 --drop-umps-coaches
  python scripts/prune_game.py --game 823141 --drop-umps-coaches \\
      --drop-fielders-non-x --drop-fielders-no-touch-post-release
  python scripts/prune_game.py --all-games --drop-umps-coaches --dry-run

⚠ Pruning is one-way for the canonical file. Only the *suffixed* source
files are stashed to pre_prune/; the pre-prune canonical is replaced in
place. Dropped rows are recoverable only by re-scraping, and only while the
game is inside MLB's retention window.

⚠ NEVER prune a game the autopilot might still capture. Gap-aware resume
computes "missing" as a set-difference against the manifest, so it reads
prune-dropped segments as gaps and refetches every one of them — mixing
pruned and unpruned data and undoing the prune. This bit us twice. The
completion marker is the guard, and `--require-marker` enforces it; it is
mandatory for `--all-games` outside a dry run. Games whose actor file is
currently open by a capture worker are skipped automatically.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games


SKELETAL_ROLES = ("pitcher", "catcher", "batter")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_filter(args, con: duckdb.DuckDBPyConnection) -> str:
    """Compose a single SQL WHERE clause that keeps the rows we want.
    Empty string means 'no filter, copy as-is'."""
    keep = []

    if args.drop_umps_coaches:
        # Keep anything that isn't an umpire/coach.
        keep.append(
            "actor_type NOT IN ('umpire', 'plate-umpire', 'coach')"
        )

    if args.drop_fielders_non_x:
        # Need pitch_label to know which time windows are non-X.
        non_x_exists = con.execute(
            "SELECT COUNT(*) FROM pitch_label WHERE result_call != 'X' "
            "AND start_time_unix IS NOT NULL AND end_time_unix IS NOT NULL"
        ).fetchone()[0]
        if non_x_exists == 0:
            _log("  --drop-fielders-non-x: no labeled non-X pitches in this game; "
                 "skipping (run ingest_pitch_labels.py first)")
        else:
            roles = ", ".join(f"'{r}'" for r in SKELETAL_ROLES)
            keep.append(
                f"NOT (actor_type = 'fielder' AND EXISTS ("
                f"  SELECT 1 FROM pitch_label pl "
                f"   WHERE pl.result_call != 'X' "
                f"     AND af.time_unix BETWEEN pl.start_time_unix AND pl.end_time_unix"
                f"))"
            )

    if args.drop_fielders_no_touch_post_release:
        # X-pitch + post-release + fielder + no hand-ball proximity → drop.
        # We compute the keep-set: fielders who DID touch the ball during
        # *some* X pitch. Anyone in the keep-set survives even on plays
        # they weren't involved in.
        r = con.execute(
            "SELECT COUNT(*) FROM pitch_label WHERE result_call = 'X' "
            "AND start_time_unix IS NOT NULL AND end_time_unix IS NOT NULL"
        ).fetchone()[0]
        if r == 0:
            _log("  --drop-fielders-no-touch-post-release: no labeled X pitches; "
                 "skipping (run ingest_pitch_labels.py first)")
        else:
            radius = float(args.touch_radius)
            # Build a temp set of (play_id, mlb_player_id) where the player
            # got within `radius` of the ball during the play window.
            con.execute("DROP TABLE IF EXISTS _touchers")
            con.execute(f"""
                CREATE TEMP TABLE _touchers AS
                SELECT DISTINCT pl.play_id, af.mlb_player_id
                  FROM pitch_label pl
                  JOIN actor_frame af
                    ON af.time_unix BETWEEN pl.start_time_unix AND pl.end_time_unix
                  JOIN ball_frame bf
                    ON bf.time_unix = af.time_unix
                 WHERE pl.result_call = 'X'
                   AND af.actor_type = 'fielder'
                   AND (
                     ((af.hand_lt_x - bf.ball_x) * (af.hand_lt_x - bf.ball_x)
                    + (af.hand_lt_y - bf.ball_y) * (af.hand_lt_y - bf.ball_y)
                    + (af.hand_lt_z - bf.ball_z) * (af.hand_lt_z - bf.ball_z)) < {radius * radius}
                     OR
                     ((af.hand_rt_x - bf.ball_x) * (af.hand_rt_x - bf.ball_x)
                    + (af.hand_rt_y - bf.ball_y) * (af.hand_rt_y - bf.ball_y)
                    + (af.hand_rt_z - bf.ball_z) * (af.hand_rt_z - bf.ball_z)) < {radius * radius}
                   )
            """)
            keep.append(
                "NOT (actor_type = 'fielder' "
                "  AND EXISTS ("
                "    SELECT 1 FROM pitch_label pl "
                "     WHERE pl.result_call = 'X' "
                "       AND af.time_unix BETWEEN pl.start_time_unix AND pl.end_time_unix "
                "       AND af.time_unix > pl.start_time_unix "  # post-release only
                "       AND NOT EXISTS ("
                "         SELECT 1 FROM _touchers t "
                "          WHERE t.play_id = pl.play_id "
                "            AND t.mlb_player_id = af.mlb_player_id"
                "       )"
                "  )"
                ")"
            )

    return " AND ".join(f"({c})" for c in keep)


def _finalized(p: Path) -> bool:
    try:
        if p.stat().st_size < 8:
            return False
        with p.open("rb") as f:
            f.seek(-4, 2)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def _open_paths() -> set[str]:
    """Files currently held open by any visible process (Linux /proc scan).

    A game whose canonical actor file is open is being captured RIGHT NOW.
    Pruning it would rewrite the file out from under the writer.
    """
    live: set[str] = set()
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        try:
            entries = os.listdir(fd_dir)
        except OSError:
            continue
        for entry in entries:
            try:
                live.add(os.readlink(os.path.join(fd_dir, entry)))
            except OSError:
                continue
    return live


def prune_one(game_pk: int, data_dir: Path, args, live_files=frozenset()) -> dict:
    gdir = data_dir / str(game_pk)
    af_path = gdir / "actor_frames.parquet"

    # ── Safety gate ────────────────────────────────────────────────────
    # Gap-aware resume treats prune-dropped segments as missing, so a capture
    # pass over a pruned game refetches everything we just dropped (this bit
    # us twice). The completion marker is the guard: never prune a game the
    # autopilot might still touch.
    if args.require_marker:
        marker = Path(args.marker_dir) / f"complete_{game_pk}.marker"
        if not marker.exists():
            return {"game_pk": game_pk,
                    "skipped": "no completion marker (autopilot may resume it)"}
    if any(str(p) in live_files for p in gdir.glob("actor_frames*.parquet")):
        return {"game_pk": game_pk, "skipped": "actor file open — capture in flight"}
    # Union ALL finalized actor files — live/heal captures leave suffixed
    # poll-*/resume-* files beside (or instead of) the canonical one. The
    # pruned output is written as the single canonical file and the
    # suffixed sources are stashed, so this is prune + implicit compaction
    # in one rewrite.
    af_files = sorted(p for p in gdir.glob("actor_frames*.parquet")
                      if _finalized(p))
    if not af_files:
        return {"game_pk": game_pk, "skipped": "no finalized actor_frames"}

    con = duckdb.connect()
    files_sql = "[" + ", ".join(f"'{p.as_posix()}'" for p in af_files) + "]"
    con.execute(f"CREATE VIEW actor_frame AS SELECT * FROM "
                f"read_parquet({files_sql}, union_by_name=true)")
    for table, fname in [
        ("pitch_label",  "pitch_labels"),
        ("ball_frame",   "ball_frames"),
    ]:
        p = gdir / f"{fname}.parquet"
        if p.exists():
            con.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{p.as_posix()}')"
            )

    where = _build_filter(args, con)
    before = con.execute("SELECT COUNT(*) FROM actor_frame").fetchone()[0]
    if not where:
        con.close()
        return {"game_pk": game_pk, "skipped": "no cuts requested",
                "rows_before": before}

    after = con.execute(
        f"SELECT COUNT(*) FROM actor_frame af WHERE {where}"
    ).fetchone()[0]
    pct = 100.0 * after / max(before, 1)

    out = {
        "game_pk": game_pk,
        "rows_before": before,
        "rows_after": after,
        "pct_kept": pct,
        "bytes_before": sum(p.stat().st_size for p in af_files),
    }

    if args.dry_run:
        con.close()
        return out

    tmp = af_path.with_suffix(".parquet.tmp")
    con.execute(
        f"COPY (SELECT * FROM actor_frame af WHERE {where}) "
        f"TO '{tmp.as_posix()}' "
        f"(FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 100000)"
    )
    # Verify the written file BEFORE swapping it in: a CIFS write-behind
    # error (observed during a NAS ENOSPC burst 2026-08-09) can truncate
    # the output without COPY raising — reading the count both checks the
    # footer and the row total.
    try:
        got = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{tmp.as_posix()}')"
        ).fetchone()[0]
    except Exception as e:
        got = f"unreadable: {e}"
    con.close()
    if got != after:
        tmp.unlink(missing_ok=True)
        out["error"] = (f"ABORTED, original kept: tmp had {got} rows, "
                        f"expected {after}")
        return out
    # Atomic-ish swap (rename within same filesystem), then stash the
    # suffixed source files: the pruned canonical now holds their (pruned)
    # rows, so leaving them in place would make union readers double-count
    # AND resurrect pruned rows.
    tmp.replace(af_path)
    stash = gdir / "pre_prune"
    for p in af_files:
        if p != af_path:
            stash.mkdir(exist_ok=True)
            p.rename(stash / p.name)
    out["bytes_after"] = af_path.stat().st_size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, help="Single gamePk")
    ap.add_argument("--all-games", action="store_true",
                    help="Run against every game in data-dir")
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR", "data"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be cut; don't rewrite anything")
    ap.add_argument("--drop-umps-coaches", action="store_true")
    ap.add_argument("--drop-fielders-non-x", action="store_true")
    ap.add_argument("--drop-fielders-no-touch-post-release", action="store_true")
    ap.add_argument("--touch-radius", type=float, default=2.0,
                    help="Hand-to-ball distance (ft) below which a fielder is "
                         "considered to have touched the ball")
    ap.add_argument("--require-marker", action="store_true",
                    help="Only prune games with a complete_<pk>.marker. "
                         "Strongly recommended for --all-games: pruning a "
                         "game the autopilot later resumes makes gap-aware "
                         "resume refetch every dropped segment.")
    ap.add_argument("--marker-dir",
                    default=os.environ.get(
                        "FV_STATE_DIR",
                        "/media/scratch/spencer/fieldvision/state"),
                    help="Where complete_<pk>.marker files live")
    args = ap.parse_args()

    if args.all_games and not args.require_marker and not args.dry_run:
        ap.error("--all-games without --require-marker would prune games the "
                 "autopilot may still resume, and gap-aware resume would then "
                 "refetch every dropped segment. Pass --require-marker (or "
                 "--dry-run to preview).")

    if not (args.game or args.all_games):
        ap.error("specify --game <pk> or --all-games")
    if args.game and args.all_games:
        ap.error("--game and --all-games are mutually exclusive")

    data_dir = Path(args.data_dir)
    targets = [args.game] if args.game else list_games(data_dir)
    if not targets:
        _log(f"no games found under {data_dir}")
        return

    _log(f"{'DRY RUN: ' if args.dry_run else ''}pruning {len(targets)} games")
    live_files = _open_paths()
    total_before = 0
    total_after = 0
    total_bytes_before = 0
    total_bytes_after = 0
    skipped: dict[str, int] = {}
    for pk in targets:
        try:
            r = prune_one(pk, data_dir, args, live_files)
        except Exception as e:
            _log(f"  pk={pk}: FAILED ({type(e).__name__}: {e})")
            continue
        if "skipped" in r:
            skipped[r["skipped"]] = skipped.get(r["skipped"], 0) + 1
            _log(f"  pk={pk}: skipped — {r['skipped']}")
            continue
        if "error" in r:
            _log(f"  pk={pk}: {r['error']}")
            continue
        b, a = r["rows_before"], r["rows_after"]
        total_before += b
        total_after += a
        total_bytes_before += r["bytes_before"]
        if "bytes_after" in r:
            total_bytes_after += r["bytes_after"]
            _log(f"  pk={pk}: {b:,} → {a:,} rows ({r['pct_kept']:.1f}% kept), "
                 f"{r['bytes_before']/1e9:.2f} → {r['bytes_after']/1e9:.2f} GB")
        else:
            _log(f"  pk={pk}: {b:,} → {a:,} rows ({r['pct_kept']:.1f}% kept) [dry-run]")

    if skipped:
        _log("")
        _log("=== skipped ===")
        for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            _log(f"  {n:4d}  {reason}")

    if total_before:
        pct = 100.0 * total_after / total_before
        _log("")
        _log(f"=== summary ===")
        _log(f"  rows: {total_before:,} → {total_after:,} ({pct:.1f}% kept, "
             f"{100 - pct:.1f}% pruned)")
        if total_bytes_after:
            _log(f"  bytes: {total_bytes_before/1e9:.1f} → {total_bytes_after/1e9:.1f} GB "
                 f"({(total_bytes_before - total_bytes_after)/1e9:.1f} GB freed)")


if __name__ == "__main__":
    main()
