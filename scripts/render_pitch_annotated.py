"""Render a single pitch with KINEMATIC event markers + bat overlay.

Wire labels (PITCHER_FIRST_MOVEMENT, BALL_WAS_RELEASED) proved unreliable.
This version derives the delivery sub-events from joint trajectories:
  - WINDUP_ONSET: first sustained throwing-hand motion
  - KNEE_HIGH:    peak of front-knee y
  - FOOT_LANDING: first frame after KNEE_HIGH where front foot is grounded
  - RELEASE:      from statsapi start_time_unix (verified vs ball_frame)

Pre-pitch oscillation window = N seconds before WINDUP_ONSET (default 5s).
Wire PFM is shown faded for reference.

Usage:
    python scripts/render_pitch_annotated.py --game 823141 --play-id 88a54a0d-fbc1-30ee-8cd2-32e3c95e5e35
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

PRE_OSC_SECONDS = 5.0   # pre-windup-onset oscillation window (where the interesting batter movement is)
POST_RELEASE_SECONDS = 1.0


def load_play(db_path: Path, play_id: str):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    pl = conn.execute("SELECT * FROM pitch_label WHERE play_id=?", (play_id,)).fetchone()
    if pl is None:
        raise SystemExit(f"play_id {play_id} not in pitch_label")
    pl = dict(pl)
    release_t = pl["start_time_unix"]

    # Time window: pre-osc window for batter analysis through release+1s
    t_lo = release_t - (PRE_OSC_SECONDS + 5.0)  # extra 5s in case windup_onset is well before release
    t_hi = release_t + POST_RELEASE_SECONDS

    # Load actors (batter + pitcher) over a generous window
    select_cols = "mlb_player_id, time_unix, " + ", ".join(
        f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS
    )
    rows = conn.execute(
        f"SELECT {select_cols} FROM actor_frame "
        "WHERE mlb_player_id IN (?, ?) AND time_unix BETWEEN ? AND ? "
        "ORDER BY mlb_player_id, time_unix",
        (pl["batter_id"], pl["pitcher_id"], t_lo, t_hi),
    ).fetchall()
    by_player = defaultdict(list)
    for r in rows:
        mlb_id = r[0]; t = r[1]
        wp = {}
        for i, bid in enumerate(JOINT_BIDS):
            x, y, z = r[2 + i*3], r[3 + i*3], r[4 + i*3]
            if x is not None:
                wp[bid] = (x, y, z)
        by_player[mlb_id].append((t, wp))

    # Load bat trajectory (canonical inferredBat)
    bats = conn.execute(
        "SELECT time_unix, head_x, head_y, head_z, handle_x, handle_y, handle_z "
        "FROM bat_frame WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (t_lo, t_hi),
    ).fetchall()
    bat_frames = [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bats]

    # Wire PFM (shown faded for reference only — unreliable)
    pfm_t = None
    r = conn.execute(
        "SELECT time_unix FROM pitch_event WHERE event_type='PITCHER_FIRST_MOVEMENT' "
        "AND time_unix BETWEEN ? AND ? ORDER BY time_unix LIMIT 1",
        (t_lo, t_hi),
    ).fetchone()
    if r:
        pfm_t = r[0]

    # Wire BEGIN_OF_PLAY (also faded, kept for context)
    bop_t = None
    r = conn.execute(
        "SELECT time_unix FROM pitch_event WHERE event_type='BEGIN_OF_PLAY' "
        "AND time_unix BETWEEN ? AND ? ORDER BY time_unix LIMIT 1",
        (t_lo, t_hi),
    ).fetchone()
    if r:
        bop_t = r[0]

    conn.close()
    return pl, dict(by_player), bat_frames, pfm_t, bop_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=823141)
    ap.add_argument("--play-id", type=str, default="88a54a0d-fbc1-30ee-8cd2-32e3c95e5e35")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    db_path = Path(args.data_dir) / f"fv_{args.game}.sqlite"
    pl, actors, bats, pfm_t, bop_t = load_play(db_path, args.play_id)
    pitcher_id, batter_id = pl["pitcher_id"], pl["batter_id"]
    p_frames = actors.get(pitcher_id, [])
    b_frames = actors.get(batter_id, [])
    release_t = pl["start_time_unix"]

    if not p_frames or not b_frames:
        raise SystemExit("missing pitcher or batter frames")

    # Derive pitcher kinematic events
    ev = detect_pitcher_events(p_frames, release_t, search_back=4.0)
    print(f"play_id {args.play_id[:8]}…  {pl['pitch_type']} {pl['start_speed']:.1f}mph  call={pl['result_call']}")
    print(f"  release_t (statsapi):     T+0.00s")
    if ev.windup_onset_t:  print(f"  windup_onset_t (derived):  T{ev.windup_onset_t - release_t:+.2f}s")
    if pfm_t is not None:  print(f"  PFM (wire, unreliable):    T{pfm_t - release_t:+.2f}s")
    if ev.knee_high_t:     print(f"  knee_high_t (derived):     T{ev.knee_high_t - release_t:+.2f}s  ({ev.knee_high_height:.2f}ft, leg={ev.front_leg_side})")
    if ev.foot_landing_t:  print(f"  foot_landing_t (derived):  T{ev.foot_landing_t - release_t:+.2f}s")
    if bop_t is not None:  print(f"  BEGIN_OF_PLAY (wire):      T{bop_t - release_t:+.2f}s")

    # Define render window
    if ev.windup_onset_t is not None:
        t_lo = ev.windup_onset_t - PRE_OSC_SECONDS
    else:
        t_lo = release_t - 7.0  # fallback
    t_hi = release_t + POST_RELEASE_SECONDS
    print(f"  render window: [{t_lo - release_t:+.2f}s, {t_hi - release_t:+.2f}s]  ({t_hi - t_lo:.1f}s)")

    # Render at uniform 30fps
    t_grid = np.arange(t_lo, t_hi, 1.0 / args.fps)
    n_frames = len(t_grid)

    p_times = np.array([f[0] for f in p_frames])
    b_times = np.array([f[0] for f in b_frames])
    bat_times = np.array([b[0] for b in bats]) if bats else np.array([])

    def frame_at(t, times, frames):
        if len(times) == 0: return None
        i = np.searchsorted(times, t)
        i = max(0, min(i, len(times) - 1))
        return frames[i][1]

    def bat_at(t):
        if len(bat_times) == 0: return None
        i = np.searchsorted(bat_times, t)
        i = max(0, min(i, len(bat_times) - 1))
        # Only use if within ~50ms of target time (otherwise the bat probably wasn't tracked here)
        if abs(bat_times[i] - t) > 0.07:
            return None
        return bats[i][1], bats[i][2]  # (head, handle)

    # Figure layout: tall timeline (event labels inside), pitcher panel, batter panel, phase strip
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a1a")
    gs = fig.add_gridspec(4, 2, height_ratios=[2.6, 6, 6, 0.4],
                          hspace=0.18, wspace=0.1, left=0.05, right=0.97, top=0.92, bottom=0.05)

    ax_tl = fig.add_subplot(gs[0, :])
    ax_tl.set_xlim(t_lo, t_hi)
    ax_tl.set_ylim(0, 1)
    ax_tl.set_facecolor("#1a1a1a")
    ax_tl.set_yticks([])
    tick_step = 1.0
    tick_locs = np.arange(np.ceil(t_lo - release_t), t_hi - release_t, tick_step) + release_t
    ax_tl.set_xticks(tick_locs)
    ax_tl.set_xticklabels([f"T{t-release_t:+.0f}s" for t in tick_locs], color="#aaa", fontsize=8)
    ax_tl.tick_params(colors="#aaa")
    for s in ax_tl.spines.values(): s.set_color("#444")

    # ── Phase regions ──
    if ev.windup_onset_t:
        # Pre-pitch oscillation: PRE_OSC_SECONDS before onset
        ax_tl.axvspan(t_lo, ev.windup_onset_t, color="#4caf50", alpha=0.18)
        ax_tl.text((t_lo + ev.windup_onset_t) / 2, 0.10,
                   f"PRE-PITCH OSCILLATION ({PRE_OSC_SECONDS:.0f}s before windup onset)",
                   ha="center", va="center", color="#a5d6a7", fontsize=9, fontweight="bold")
        # Windup phase: onset → knee_high
        if ev.knee_high_t:
            ax_tl.axvspan(ev.windup_onset_t, ev.knee_high_t, color="#ff9800", alpha=0.15)
            ax_tl.text((ev.windup_onset_t + ev.knee_high_t) / 2, 0.10, "WINDUP",
                       ha="center", va="center", color="#ffb74d", fontsize=9, fontweight="bold")
        # Stride phase: knee_high → foot_landing
        if ev.knee_high_t and ev.foot_landing_t:
            ax_tl.axvspan(ev.knee_high_t, ev.foot_landing_t, color="#e91e63", alpha=0.15)
            ax_tl.text((ev.knee_high_t + ev.foot_landing_t) / 2, 0.10, "STRIDE",
                       ha="center", va="center", color="#f8bbd0", fontsize=8, fontweight="bold")
        # Throw phase: foot_landing → release
        if ev.foot_landing_t:
            ax_tl.axvspan(ev.foot_landing_t, release_t, color="#9c27b0", alpha=0.18)
            ax_tl.text((ev.foot_landing_t + release_t) / 2, 0.10, "THROW",
                       ha="center", va="center", color="#ce93d8", fontsize=8, fontweight="bold")

    # ── Event markers (derived = bright; wire = faded) ──
    derived = []
    if ev.windup_onset_t: derived.append(("WINDUP_ONSET",  ev.windup_onset_t,  "#4caf50"))
    if ev.knee_high_t:    derived.append(("KNEE_HIGH",     ev.knee_high_t,     "#ff9800"))
    if ev.foot_landing_t: derived.append(("FOOT_LANDING",  ev.foot_landing_t,  "#e91e63"))
    derived.append(("RELEASE",       release_t,         "#9c27b0"))
    levels = [0.92, 0.72, 0.52, 0.32]
    for li, (lbl, t, color) in enumerate(derived):
        ax_tl.axvline(t, color=color, lw=2.0, alpha=0.95)
        ax_tl.text(t + 0.02, levels[li % 4], lbl, ha="left", va="center", color=color,
                   fontsize=8, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.18", facecolor="#1a1a1a", edgecolor=color, lw=0.6, alpha=0.9))
    # Faded wire markers
    if pfm_t is not None:
        ax_tl.axvline(pfm_t, color="#888", lw=1.0, alpha=0.5, ls=":")
        ax_tl.text(pfm_t + 0.02, 0.72, "wire PFM\n(unreliable)", ha="left", va="center",
                   color="#888", fontsize=7, style="italic",
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a1a1a", edgecolor="#555", lw=0.5, alpha=0.7))
    if bop_t is not None:
        ax_tl.axvline(bop_t, color="#888", lw=1.0, alpha=0.5, ls=":")
        ax_tl.text(bop_t + 0.02, 0.92, "wire BEGIN_OF_PLAY", ha="left", va="center",
                   color="#888", fontsize=7, style="italic",
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a1a1a", edgecolor="#555", lw=0.5, alpha=0.7))

    cursor = ax_tl.axvline(t_lo, color="white", lw=2.5, alpha=0.9)

    # ── Skeleton panels ──
    ax_p = fig.add_subplot(gs[1:3, 0])
    ax_b = fig.add_subplot(gs[1:3, 1])

    def actor_bounds(frames, padding=2.0):
        xs, ys, zs = [], [], []
        for _, wp in frames:
            for (x, y, z) in wp.values(): xs.append(x); ys.append(y); zs.append(z)
        if not xs: return (-5, 5, 0, 8)
        return (min(zs)-padding, max(zs)+padding, max(0, min(ys)-padding), max(ys)+padding)

    pb = actor_bounds(p_frames)
    bb_ = actor_bounds(b_frames)
    ax_p.set_xlim(pb[0], pb[1]); ax_p.set_ylim(pb[2], pb[3])
    ax_b.set_xlim(bb_[0], bb_[1]); ax_b.set_ylim(bb_[2], bb_[3])
    for ax, title in ((ax_p, f"PITCHER (id={pitcher_id})"),
                      (ax_b, f"BATTER (id={batter_id})")):
        ax.set_facecolor("#2e4d2e")
        ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xlabel("Z (ft, depth)", color="#888", fontsize=8)
        for s in ax.spines.values(): s.set_color("#444")
    ax_p.set_ylabel("Y (ft, up)", color="#888", fontsize=8)

    p_lines = [ax_p.plot([], [], "-", lw=2.5, color="white", solid_capstyle="round")[0]
               for _ in SKELETON_CONNECTIONS]
    b_lines = [ax_b.plot([], [], "-", lw=2.5, color="white", solid_capstyle="round")[0]
               for _ in SKELETON_CONNECTIONS]
    p_dots = ax_p.scatter([], [], s=18, c="cyan")
    b_dots = ax_b.scatter([], [], s=18, c="cyan")
    bat_line = ax_b.plot([], [], "-", lw=4, color="#d4a04c", solid_capstyle="round")[0]
    bat_head_dot = ax_b.scatter([], [], s=60, c="#d4a04c")

    # Phase label strip at bottom
    ax_phase = fig.add_subplot(gs[3, :])
    ax_phase.set_xlim(0, 1); ax_phase.set_ylim(0, 1); ax_phase.axis("off")
    phase_text = ax_phase.text(0.5, 0.5, "", color="white", fontsize=12, fontweight="bold",
                               ha="center", va="center",
                               bbox=dict(boxstyle="round,pad=0.4", facecolor="#333", edgecolor="#666"))

    def cur_phase(t):
        if ev.windup_onset_t and t < ev.windup_onset_t: return "PRE-PITCH OSCILLATION (batter intrinsic movement, no pitcher motion)", "#a5d6a7"
        if ev.knee_high_t and t < ev.knee_high_t:        return "WINDUP (pitcher arm rises, leg lifts)", "#ffb74d"
        if ev.foot_landing_t and t < ev.foot_landing_t:  return "STRIDE (knee descending, front foot striding)", "#f8bbd0"
        if t < release_t:                                 return "THROW (foot planted, arm whip toward release)", "#ce93d8"
        return "post-release", "#999"

    def update(i):
        t = t_grid[i]
        cursor.set_xdata([t, t])

        phase, color = cur_phase(t)
        phase_text.set_text(f"t={t-release_t:+.2f}s   |   {phase}")
        phase_text.set_color(color)

        # Pitcher
        p_wp = frame_at(t, p_times, p_frames) or {}
        for li, (a, b) in enumerate(SKELETON_CONNECTIONS):
            if a in p_wp and b in p_wp:
                p1, p2 = p_wp[a], p_wp[b]
                p_lines[li].set_data([p1[2], p2[2]], [p1[1], p2[1]])
            else:
                p_lines[li].set_data([], [])
        pts = np.array([(p[2], p[1]) for p in p_wp.values()]) if p_wp else np.empty((0,2))
        p_dots.set_offsets(pts)

        # Batter
        b_wp = frame_at(t, b_times, b_frames) or {}
        for li, (a, b) in enumerate(SKELETON_CONNECTIONS):
            if a in b_wp and b in b_wp:
                p1, p2 = b_wp[a], b_wp[b]
                b_lines[li].set_data([p1[2], p2[2]], [p1[1], p2[1]])
            else:
                b_lines[li].set_data([], [])
        pts = np.array([(p[2], p[1]) for p in b_wp.values()]) if b_wp else np.empty((0,2))
        b_dots.set_offsets(pts)

        # Bat
        ba = bat_at(t)
        if ba is not None:
            head, handle = ba
            bat_line.set_data([handle[2], head[2]], [handle[1], head[1]])
            bat_head_dot.set_offsets(np.array([[head[2], head[1]]]))
        else:
            bat_line.set_data([], [])
            bat_head_dot.set_offsets(np.empty((0, 2)))

        return [cursor, p_dots, b_dots, bat_line, bat_head_dot, phase_text, *p_lines, *b_lines]

    fig.suptitle(
        f"Pitch {args.play_id[:8]}…  |  {pl['pitch_type']} {pl['start_speed']:.1f} mph  |  "
        f"call={pl['result_call']}  |  game {args.game}  |  pitcher {pitcher_id}, batter {batter_id}",
        color="white", fontsize=11, y=0.985
    )

    out_path = Path(args.out) if args.out else Path("data") / f"pitch_{args.play_id[:8]}_kinematic.mp4"
    print(f"\nRendering {n_frames} frames @ {args.fps}fps → {out_path}")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, bitrate=2400, codec="h264")
    anim.save(str(out_path), writer=writer, dpi=110)
    print(f"  done: {out_path.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
