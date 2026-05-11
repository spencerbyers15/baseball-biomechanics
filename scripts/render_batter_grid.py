"""Render a grid of pre-pitch batter postures (with bat) for one batter,
anchored on the per-pitch derived WINDUP_ONSET event.

For each pitch the batter saw, compute:
  - WINDUP_ONSET_t (kinematically detected from pitcher trajectories)
  - cell window = [WINDUP_ONSET - PRE_OSC, WINDUP_ONSET + POST_ONSET]
Render every cell synced to elapsed time so you can visually compare where
each batter's body is when the WINDUP_ONSET marker appears.

Usage:
    python scripts/render_batter_grid.py --game 823141 --batter-id 645277
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

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]

PRE_OSC_SECONDS = 5.0     # cell starts 5s before windup_onset
POST_ONSET_SECONDS = 0.5  # cell ends 0.5s after windup_onset (to see initial batter response)


def load_pitch_for_grid(conn, batter_id, play_id, pitcher_id, release_t):
    """Returns dict with pitch metadata, batter frames, bat frames, and derived events."""
    pre_pad, post_pad = PRE_OSC_SECONDS + 5.0, POST_ONSET_SECONDS + 0.5

    select_cols = "time_unix, " + ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)

    # Pitcher frames (for event detection)
    p_rows = conn.execute(
        f"SELECT {select_cols} FROM actor_frame WHERE mlb_player_id=? "
        f"AND time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (pitcher_id, release_t - pre_pad, release_t + 0.5),
    ).fetchall()
    if len(p_rows) < 30:
        return None

    p_frames = []
    for r in p_rows:
        wp = {}
        for i, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1 + i*3], r[2 + i*3], r[3 + i*3]
            if x is not None: wp[bid] = (x, y, z)
        p_frames.append((r[0], wp))

    ev = detect_pitcher_events(p_frames, release_t, search_back=4.0)
    if ev.windup_onset_t is None:
        return None

    onset_t = ev.windup_onset_t
    win_lo = onset_t - PRE_OSC_SECONDS
    win_hi = onset_t + POST_ONSET_SECONDS

    # Batter frames in the cell window
    b_rows = conn.execute(
        f"SELECT {select_cols} FROM actor_frame WHERE mlb_player_id=? "
        f"AND time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (batter_id, win_lo, win_hi),
    ).fetchall()
    if len(b_rows) < 60:
        return None

    b_frames = []
    for r in b_rows:
        wp = {}
        for i, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1 + i*3], r[2 + i*3], r[3 + i*3]
            if x is not None: wp[bid] = (x, y, z)
        b_frames.append((r[0], wp))

    # Bat frames
    bat_rows = conn.execute(
        "SELECT time_unix, head_x, head_y, head_z, handle_x, handle_y, handle_z "
        "FROM bat_frame WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (win_lo, win_hi),
    ).fetchall()
    bat_frames = [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bat_rows]

    return {
        "play_id": play_id,
        "release_t": release_t,
        "windup_onset_t": onset_t,
        "knee_high_t": ev.knee_high_t,
        "foot_landing_t": ev.foot_landing_t,
        "win_lo": win_lo,
        "win_hi": win_hi,
        "b_frames": b_frames,
        "bat_frames": bat_frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=823141)
    ap.add_argument("--batter-id", type=int, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    db_path = Path(args.data_dir) / f"fv_{args.game}.sqlite"
    conn = sqlite3.connect(str(db_path))
    pitches_meta = conn.execute(
        "SELECT play_id, pitch_type, result_call, pitcher_id, start_time_unix "
        "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL "
        "ORDER BY start_time_unix",
        (args.batter_id,),
    ).fetchall()
    print(f"batter {args.batter_id}: {len(pitches_meta)} candidate pitches in game {args.game}")

    n_cells = args.rows * args.cols
    pitches = []
    for play_id, ptype, call, pitcher_id, t_release in pitches_meta:
        if len(pitches) >= n_cells:
            break
        d = load_pitch_for_grid(conn, args.batter_id, play_id, pitcher_id, t_release)
        if d is None:
            continue
        d.update({"pitch_type": ptype, "result_call": call})
        pitches.append(d)
    conn.close()
    print(f"  {len(pitches)} usable cells (others lacked frames or kinematic onset)")
    if not pitches:
        raise SystemExit("no usable cells")

    # Total cell duration in time
    cell_total_t = PRE_OSC_SECONDS + POST_ONSET_SECONDS  # 5.5s

    # Per-cell pelvis median (for centering — body motion not absolute box position)
    for p in pitches:
        pxs = [wp.get(0, (None, None, None))[2] for _, wp in p["b_frames"] if 0 in wp]
        pys = [wp.get(0, (None, None, None))[1] for _, wp in p["b_frames"] if 0 in wp]
        p["center_x"] = float(np.median([x for x in pxs if x is not None])) if pxs else 0
        p["center_y"] = float(np.median([y for y in pys if y is not None])) if pys else 4

    CELL_HALF_W = 4.0  # ft (slightly wider than annotated to fit bat extent)
    CELL_Y_LO, CELL_Y_HI = 0.0, 8.0

    fig_w = 3.2 * args.cols
    fig_h = 3.6 * args.rows
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a1a")

    axes, line_sets, dot_sets, bat_lines, bat_heads, progress_bars, onset_lines = [], [], [], [], [], [], []
    for ci, p in enumerate(pitches):
        ax = fig.add_subplot(args.rows, args.cols, ci + 1)
        ax.set_facecolor("#2e4d2e")
        ax.set_aspect("equal")
        ax.set_xlim(p["center_x"] - CELL_HALF_W, p["center_x"] + CELL_HALF_W)
        ax.set_ylim(CELL_Y_LO, CELL_Y_HI)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#444")

        # Title with concise info
        title = f"#{ci+1}  {p['pitch_type'] or '?'}/{p['result_call'] or '?'}"
        ax.set_title(title, color="white", fontsize=9, pad=3)

        lines = [ax.plot([], [], "-", lw=1.8, color="white", solid_capstyle="round")[0]
                 for _ in SKELETON_CONNECTIONS]
        dots = ax.scatter([], [], s=8, c="cyan")
        bat_line = ax.plot([], [], "-", lw=2.6, color="#d4a04c", solid_capstyle="round")[0]
        bat_head = ax.scatter([], [], s=24, c="#d4a04c")

        # Static onset marker on the cell — small "WINDUP→" tag that will appear when cursor reaches it
        # This is just a visual reference at the cell location; the cursor we plot will sweep across.
        # Mark the onset position as a vertical orange line at the bottom of the cell, fixed.
        # Onset fraction across the cell window = PRE_OSC / total
        onset_frac = PRE_OSC_SECONDS / cell_total_t
        onset_x = ax.get_xlim()[0] + onset_frac * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.plot([onset_x, onset_x], [CELL_Y_LO + 0.05, CELL_Y_LO + 0.4],
                color="#ff9800", lw=2.5, solid_capstyle="round")
        ax.text(onset_x, CELL_Y_LO + 0.55, "WINDUP", color="#ff9800",
                ha="center", fontsize=6, fontweight="bold")

        # Progress bar at the very bottom (yellow); will grow with frame
        bar = ax.plot([ax.get_xlim()[0], ax.get_xlim()[0]],
                      [CELL_Y_LO + 0.02, CELL_Y_LO + 0.02],
                      color="#fbc02d", lw=3)[0]

        # Vertical sweeping cursor (white) that crosses the orange onset line
        cursor_line = ax.axvline(ax.get_xlim()[0], color="white", lw=1.0, alpha=0.5)

        axes.append(ax); line_sets.append(lines); dot_sets.append(dots)
        bat_lines.append(bat_line); bat_heads.append(bat_head)
        progress_bars.append(bar); onset_lines.append(cursor_line)

    fig.suptitle(
        f"Batter {args.batter_id}  |  game {args.game}  |  "
        f"{len(pitches)} pre-pitch postures (5s before kinematic windup-onset → 0.5s after)  |  "
        f"orange WINDUP marker = pitcher's true motion onset for that pitch",
        color="white", fontsize=11, y=0.995
    )

    n_anim_frames = int(cell_total_t * args.fps)
    t_grid_frac = np.linspace(0, 1, n_anim_frames)

    def frame_at_frac(frames, frac, win_lo, win_hi):
        if not frames: return {}
        t_target = win_lo + frac * (win_hi - win_lo)
        ts = np.array([f[0] for f in frames])
        i = np.searchsorted(ts, t_target)
        i = max(0, min(i, len(frames) - 1))
        return frames[i][1]

    def bat_at_frac(bat_frames, frac, win_lo, win_hi):
        if not bat_frames: return None
        t_target = win_lo + frac * (win_hi - win_lo)
        ts = np.array([b[0] for b in bat_frames])
        i = np.searchsorted(ts, t_target)
        i = max(0, min(i, len(bat_frames) - 1))
        if abs(ts[i] - t_target) > 0.07:
            return None
        return bat_frames[i][1], bat_frames[i][2]

    def update(idx):
        frac = t_grid_frac[idx]
        artists = []
        for ci, p in enumerate(pitches):
            wp = frame_at_frac(p["b_frames"], frac, p["win_lo"], p["win_hi"])
            for li, (a, b) in enumerate(SKELETON_CONNECTIONS):
                if a in wp and b in wp:
                    p1, p2 = wp[a], wp[b]
                    line_sets[ci][li].set_data([p1[2], p2[2]], [p1[1], p2[1]])
                else:
                    line_sets[ci][li].set_data([], [])
            if wp:
                pts = np.array([(pp[2], pp[1]) for pp in wp.values()])
                dot_sets[ci].set_offsets(pts)
            else:
                dot_sets[ci].set_offsets(np.empty((0, 2)))

            ba = bat_at_frac(p["bat_frames"], frac, p["win_lo"], p["win_hi"])
            if ba is not None:
                head, handle = ba
                bat_lines[ci].set_data([handle[2], head[2]], [handle[1], head[1]])
                bat_heads[ci].set_offsets(np.array([[head[2], head[1]]]))
            else:
                bat_lines[ci].set_data([], [])
                bat_heads[ci].set_offsets(np.empty((0, 2)))

            xmin, xmax = axes[ci].get_xlim()
            cur_x = xmin + frac * (xmax - xmin)
            progress_bars[ci].set_data([xmin, cur_x],
                                       [CELL_Y_LO + 0.02, CELL_Y_LO + 0.02])
            onset_lines[ci].set_xdata([cur_x, cur_x])

            artists.extend(line_sets[ci])
            artists.extend([dot_sets[ci], bat_lines[ci], bat_heads[ci],
                           progress_bars[ci], onset_lines[ci]])
        return artists

    out_path = Path(args.out) if args.out else (
        Path("data") / f"batter_{args.batter_id}_{args.game}_grid.mp4"
    )
    print(f"Rendering {n_anim_frames} frames @ {args.fps}fps → {out_path}")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, hspace=0.22, wspace=0.08)
    anim = FuncAnimation(fig, update, frames=n_anim_frames, interval=1000/args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, bitrate=3000, codec="h264")
    anim.save(str(out_path), writer=writer, dpi=100)
    print(f"  done: {out_path.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
