"""Render Steven Kwan's pre-pitch postures color-coded by swing/take.

The most actionable finding from the overnight analysis: Kwan's pre-pitch posture
phase predicts whether he swings (KS p=0.010 at windup_onset). This makes that
finding VISIBLE — separate red and blue panels for swing vs take pitches in his
pre-pitch oscillation window.

Usage:
    python scripts/render_kwan_swing_take_grid.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.skeleton import SKELETON_CONNECTIONS
from fieldvision.storage import JOINT_COLS
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
PRE_OSC_SECONDS = 5.0
POST_ONSET_SECONDS = 0.5
TOTAL_T = PRE_OSC_SECONDS + POST_ONSET_SECONDS
SAMPLE_HZ = 30.0


def load_pitch(conn, batter_id, play_id, pitcher_id, release_t):
    joint_cols_select = ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)
    select_cols = "time_unix, " + joint_cols_select
    p_rows_raw = load_clean_batter_actor_frames(
        conn, pitcher_id, release_t - 5, release_t + 0.5, joint_cols_select)
    # Convert to the (time, ...) shape (skip actor_uid col)
    p_rows = [(r[0],) + r[2:] for r in p_rows_raw]
    if len(p_rows) < 30: return None
    p_frames = []
    for r in p_rows:
        wp = {}
        for i, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1 + i*3], r[2 + i*3], r[3 + i*3]
            if x is not None: wp[bid] = (x, y, z)
        p_frames.append((r[0], wp))
    ev = detect_pitcher_events(p_frames, release_t, search_back=4.0)
    if ev.windup_onset_t is None: return None
    onset_t = ev.windup_onset_t
    win_lo, win_hi = onset_t - PRE_OSC_SECONDS, onset_t + POST_ONSET_SECONDS

    b_rows_raw = load_clean_batter_actor_frames(conn, batter_id, win_lo, win_hi, joint_cols_select)
    b_rows = [(r[0],) + r[2:] for r in b_rows_raw]
    if len(b_rows) < 60: return None

    b_frames = []
    for r in b_rows:
        wp = {}
        for i, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1 + i*3], r[2 + i*3], r[3 + i*3]
            if x is not None: wp[bid] = (x, y, z)
        b_frames.append((r[0], wp))
    bat_rows = conn.execute(
        "SELECT time_unix, head_x, head_y, head_z, handle_x, handle_y, handle_z "
        "FROM bat_frame WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (win_lo, win_hi),
    ).fetchall()
    bat_frames_raw = [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bat_rows]
    # Physical-sanity filter on bats (length, height-above-ground only).
    bat_frames = filter_bat_frames(bat_frames_raw)
    # Per-pitch quality: drop the pitch entirely if data is suspect
    quality = assess_pitch_quality(b_frames, bat_frames)
    if not quality["is_clean"]:
        return None
    return {"play_id": play_id, "win_lo": win_lo, "win_hi": win_hi,
            "windup_onset_t": onset_t, "b_frames": b_frames, "bat_frames": bat_frames,
            "quality": quality}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batter-id", type=int, default=680757)  # Steven Kwan
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/oscillation_report/pre_pitch_preparatory_movement/extra_kwan_swing_take_grid.mp4")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    swing_pitches = []
    take_pitches = []
    for db_path in sorted(Path(args.data_dir).glob("fv_*.sqlite")):
        if "registry" in db_path.name or "backup" in db_path.name: continue
        conn = sqlite3.connect(str(db_path))
        for r in conn.execute(
            "SELECT play_id, pitcher_id, start_time_unix, pitch_type, result_call "
            "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL",
            (args.batter_id,),
        ):
            data = load_pitch(conn, args.batter_id, r[0], r[1], r[2])
            if data is None: continue
            data.update(pitch_type=r[3], result_call=r[4])
            if r[4] in ("S", "X", "F"):
                swing_pitches.append(data)
            elif r[4] in ("B", "C", "*B"):
                take_pitches.append(data)
        conn.close()
    print(f"Swing: {len(swing_pitches)}, Take: {len(take_pitches)}")

    if not swing_pitches or not take_pitches:
        raise SystemExit("need both swings and takes")

    # Layout: two columns, swing column on left, take column on right.
    # Each column has up to 6 cells (rows).
    n_show = min(6, max(len(swing_pitches), len(take_pitches)))
    swing_show = swing_pitches[:n_show]
    take_show = take_pitches[:n_show]

    fig, axes = plt.subplots(n_show, 2, figsize=(8, 3 * n_show), facecolor="#1a1a1a")
    if n_show == 1: axes = np.array([axes])

    # Per-cell setup
    cells = []  # list of (ax, p_data, color, lines, dots, bat_line, bat_head, cur_line)
    for ri in range(n_show):
        for ci, (col_data, color, label) in enumerate([(swing_show, "#f44336", "SWING"),
                                                         (take_show, "#42a5f5", "TAKE")]):
            ax = axes[ri][ci]
            ax.set_facecolor("#2e4d2e")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(color)
                s.set_linewidth(2)
            if ri >= len(col_data):
                ax.set_visible(False); continue
            p = col_data[ri]
            # Center on median pelvis
            pxs = [wp.get(0, (None, None, None))[2] for _, wp in p["b_frames"] if 0 in wp]
            pys = [wp.get(0, (None, None, None))[1] for _, wp in p["b_frames"] if 0 in wp]
            cx = float(np.median([x for x in pxs if x is not None])) if pxs else 0
            ax.set_xlim(cx - 4, cx + 4); ax.set_ylim(0, 8)
            ax.set_title(f"{label} #{ri+1}  {p['pitch_type']}/{p['result_call']}",
                         color=color, fontsize=9)
            lines = [ax.plot([], [], "-", lw=1.8, color="white", solid_capstyle="round")[0]
                     for _ in SKELETON_CONNECTIONS]
            dots = ax.scatter([], [], s=8, c="cyan")
            bl = ax.plot([], [], "-", lw=2.6, color="#d4a04c", solid_capstyle="round")[0]
            bh = ax.scatter([], [], s=24, c="#d4a04c")
            # Onset position marker (orange line)
            onset_frac = PRE_OSC_SECONDS / TOTAL_T
            onset_x = cx - 4 + onset_frac * 8
            ax.plot([onset_x, onset_x], [0.05, 0.4], color="#ff9800", lw=2.5)
            ax.text(onset_x, 0.55, "WINDUP", color="#ff9800", ha="center", fontsize=6, fontweight="bold")
            # Sweeping cursor
            cur = ax.axvline(cx - 4, color="white", lw=0.8, alpha=0.5)
            cells.append((ax, p, color, lines, dots, bl, bh, cur))

    fig.suptitle(f"Steven Kwan (id={args.batter_id})  |  Pre-pitch postures: SWING (red, left) vs TAKE (blue, right)\n"
                 f"All cells synced to elapsed time. Orange WINDUP tick = pitcher's true motion onset.\n"
                 f"KS p=0.010 at windup_onset → posture differs significantly between swings and takes",
                 color="white", fontsize=10, y=0.995)

    n_anim = int(TOTAL_T * args.fps)
    t_grid_frac = np.linspace(0, 1, n_anim)

    def frame_at_frac(frames, frac, lo, hi):
        if not frames: return {}
        t = lo + frac * (hi - lo)
        ts = np.array([f[0] for f in frames])
        i = np.searchsorted(ts, t)
        i = max(0, min(i, len(frames) - 1))
        return frames[i][1]

    def bat_at_frac(bat_frames, frac, lo, hi):
        if not bat_frames: return None
        t = lo + frac * (hi - lo)
        ts = np.array([b[0] for b in bat_frames])
        i = np.searchsorted(ts, t)
        i = max(0, min(i, len(bat_frames) - 1))
        if abs(ts[i] - t) > 0.07: return None
        return bat_frames[i][1], bat_frames[i][2]

    def update(idx):
        frac = t_grid_frac[idx]
        artists = []
        for ax, p, color, lines, dots, bl, bh, cur in cells:
            wp = frame_at_frac(p["b_frames"], frac, p["win_lo"], p["win_hi"])
            for li, (a, b) in enumerate(SKELETON_CONNECTIONS):
                if a in wp and b in wp:
                    p1, p2 = wp[a], wp[b]
                    lines[li].set_data([p1[2], p2[2]], [p1[1], p2[1]])
                else:
                    lines[li].set_data([], [])
            if wp:
                pts = np.array([(pp[2], pp[1]) for pp in wp.values()])
                dots.set_offsets(pts)
            else:
                dots.set_offsets(np.empty((0, 2)))
            ba = bat_at_frac(p["bat_frames"], frac, p["win_lo"], p["win_hi"])
            if ba is not None:
                head, handle = ba
                bl.set_data([handle[2], head[2]], [handle[1], head[1]])
                bh.set_offsets(np.array([[head[2], head[1]]]))
            else:
                bl.set_data([], [])
                bh.set_offsets(np.empty((0, 2)))
            xmin, xmax = ax.get_xlim()
            cur_x = xmin + frac * (xmax - xmin)
            cur.set_xdata([cur_x, cur_x])
            artists.extend(lines)
            artists.extend([dots, bl, bh, cur])
        return artists

    print(f"Rendering {n_anim} frames...")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02, hspace=0.28, wspace=0.06)
    anim = FuncAnimation(fig, update, frames=n_anim, interval=1000/args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, bitrate=3000, codec="h264")
    anim.save(args.out, writer=writer, dpi=100)
    print(f"  done: {Path(args.out).stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
