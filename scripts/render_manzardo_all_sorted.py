"""Render ALL of Manzardo's clean pre-osc pitches in one video, sorted
top-to-bottom by the distance of the windup_onset frame from origin in
PC1×PC2. Top rows = batters whose end-of-pre-osc posture was farthest from
their own stance centroid (most likely contaminated by setup motion);
bottom rows = batters who were solidly in stance at windup_onset.

Layout per row: PITCHER | BATTER+BAT | PC1×PC2 trajectory
(no phase circle — redundant for this diagnostic; no ball panel — keeps cells small)
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
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

PRE_OSC_SECONDS = 5.0
SAMPLE_HZ = 30.0
MANZARDO = 700932

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
JOINT_NAMES = {bid: name for bid, name in JOINT_COLS}

T_POST_KH = 2.2   # seconds after knee_high


def load_pitch_full(conn, batter_id, play_id, pitcher_id, release_t):
    joint_cols_select = ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)
    p_rows_raw = load_clean_batter_actor_frames(
        conn, pitcher_id, release_t - 5, release_t + 2.5, joint_cols_select)
    p_rows = [(r[0],) + r[2:] for r in p_rows_raw]
    if len(p_rows) < 30: return None
    pitcher_frames = []
    for r in p_rows:
        wp = {}
        for i, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1+i*3], r[2+i*3], r[3+i*3]
            if x is not None: wp[bid] = (x, y, z)
        pitcher_frames.append((r[0], wp))
    ev = detect_pitcher_events(pitcher_frames, release_t, search_back=4.0)
    if ev.windup_onset_t is None or ev.knee_high_t is None: return None
    kh = ev.knee_high_t
    cell_lo = ev.windup_onset_t - PRE_OSC_SECONDS
    cell_hi = kh + T_POST_KH
    qa_lo = cell_lo
    qa_hi = max(release_t + 2.5, kh + T_POST_KH + 0.3)
    b_rows_raw = load_clean_batter_actor_frames(conn, batter_id, qa_lo, qa_hi, joint_cols_select)
    b_rows = [(r[0],) + r[2:] for r in b_rows_raw]
    if len(b_rows) < 60: return None
    batter_frames = []
    for r in b_rows:
        wp = {}
        for i, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1+i*3], r[2+i*3], r[3+i*3]
            if x is not None: wp[bid] = (x, y, z)
        batter_frames.append((r[0], wp))
    bat_rows = conn.execute(
        "SELECT time_unix, head_x, head_y, head_z, handle_x, handle_y, handle_z "
        "FROM bat_frame WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (qa_lo, qa_hi)).fetchall()
    bat_frames = filter_bat_frames(
        [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bat_rows])
    quality = assess_pitch_quality(batter_frames, bat_frames)
    if not quality["is_clean"]: return None

    return {
        "play_id": play_id, "windup_onset_t": ev.windup_onset_t,
        "knee_high_t": kh, "release_t": release_t,
        "cell_lo": cell_lo, "cell_hi": cell_hi,
        "pitcher_frames": pitcher_frames,
        "batter_frames": batter_frames,
        "bat_frames": bat_frames,
    }


def frame_at(t, frames):
    if not frames: return None
    ts = np.array([f[0] for f in frames])
    i = int(np.argmin(np.abs(ts - t)))
    return frames[i][1]


def bat_at(t, bat_frames):
    if not bat_frames: return None
    ts = np.array([b[0] for b in bat_frames])
    i = int(np.argmin(np.abs(ts - t)))
    if abs(ts[i] - t) > 0.07: return None
    return bat_frames[i][1], bat_frames[i][2]


def project_pc_full(p, pc12, mean_pose):
    """Project the pre-osc frames onto PC1+PC2. Returns (ts, pc_coords)."""
    pre_lo = p["windup_onset_t"] - PRE_OSC_SECONDS
    pre_hi = p["windup_onset_t"]
    bat_times = np.array([b[0] for b in p["bat_frames"]])
    rows = []
    for t, wp in p["batter_frames"]:
        if not (pre_lo <= t <= pre_hi): continue
        joint_vec = np.full(len(JOINT_BIDS) * 3, np.nan)
        for j, bid in enumerate(JOINT_BIDS):
            if bid in wp: joint_vec[j*3:j*3+3] = wp[bid]
        if np.any(np.isnan(joint_vec)): continue
        if len(bat_times) == 0: continue
        bi = int(np.argmin(np.abs(bat_times - t)))
        if abs(bat_times[bi] - t) > 0.07: continue
        head, handle = p["bat_frames"][bi][1], p["bat_frames"][bi][2]
        rows.append((t, np.concatenate([joint_vec, head, handle])))
    if len(rows) < 5: return None, None
    ts = np.array([r[0] for r in rows])
    vecs = np.array([r[1] for r in rows])
    proj = (vecs - mean_pose) @ pc12.T
    return ts, proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/oscillation_report/pre_pitch_preparatory_movement/manzardo_all_sorted.mp4")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    census = pickle.load(open("data/oscillation_report/pre_pitch_preparatory_movement/census.pkl", "rb"))
    manzardo_key = f"{MANZARDO}_L"
    cres = census["results"][manzardo_key]
    pc12 = cres["pca_components"][:2]
    var_explained = cres["pca_var_explained"]
    per_pitch_mean_pose = cres.get("per_pitch_mean_pose", {})
    print(f"PC1+PC2 = {var_explained[0]*100:.0f}% + {var_explained[1]*100:.0f}%")

    # Load all Manzardo's pitches that survived the census's data-quality check
    keep_play_ids = set(per_pitch_mean_pose.keys())
    pitches = []
    for db_path in sorted(Path(args.data_dir).glob("fv_*.sqlite")):
        if "registry" in db_path.name or "backup" in db_path.name: continue
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT play_id, pitcher_id, start_time_unix, pitch_type, result_call "
                "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL",
                (MANZARDO,)).fetchall()
        except sqlite3.OperationalError:
            conn.close(); continue
        for play_id, pitcher_id, t_rel, pt, call in rows:
            if play_id not in keep_play_ids: continue
            d = load_pitch_full(conn, MANZARDO, play_id, pitcher_id, t_rel)
            if d is None: continue
            d.update(pitch_type=pt, result_call=call)
            mp = per_pitch_mean_pose[play_id]
            ts, proj = project_pc_full(d, pc12, mp)
            if proj is None: continue
            d["pc_times"] = ts; d["pc_coords"] = proj
            d["final_dist"] = float(np.linalg.norm(proj[-1]))
            pitches.append(d)
        conn.close()

    # Sort top→bottom by descending final_dist
    pitches.sort(key=lambda p: -p["final_dist"])
    n_rows = len(pitches)
    print(f"Loaded {n_rows} clean Manzardo pitches; sorted by ||PC_end||")

    # Global PC bounds (99th percentile)
    all_pc = np.vstack([p["pc_coords"] for p in pitches])
    pc_lim = float(np.percentile(np.abs(all_pc), 99)) * 1.1
    print(f"PC axis range: ±{pc_lim:.2f}")

    # Animation length = max cell duration
    total_t = max(p["cell_hi"] - p["cell_lo"] for p in pitches)
    n_anim = int(total_t * args.fps)
    t_grid = np.linspace(0, 1, n_anim)
    print(f"  cell duration: {total_t:.1f}s; rendering {n_anim} frames @ {args.fps}fps")

    # Figure: 3 columns, n_rows rows. Make rows small (0.85") so the whole grid fits.
    row_h = 0.85
    fig_w = 13
    fig_h = row_h * n_rows + 1.6
    if int(fig_h * 100) % 2 != 0: fig_h += 1/100.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a1a")
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[1.0, 1.4, 1.1],
                          hspace=0.45, wspace=0.18,
                          left=0.04, right=0.99, top=1 - 1.2 / fig_h, bottom=0.6 / fig_h)

    cells = []
    for ri, p in enumerate(pitches):
        is_inplay = p["result_call"] == "X"
        is_whiff = p["result_call"] == "S"
        is_foul = p["result_call"] == "F"
        border_color = ("#4caf50" if is_inplay else
                        "#f44336" if is_whiff else
                        "#fbc02d" if is_foul else "#42a5f5")
        label = {"X": "INPLAY", "S": "WHIFF", "F": "FOUL"}.get(p["result_call"], p["result_call"])

        ax_p = fig.add_subplot(gs[ri, 0])
        ax_b = fig.add_subplot(gs[ri, 1])
        ax_pc = fig.add_subplot(gs[ri, 2])

        for ax in (ax_p, ax_b):
            ax.set_facecolor("#2e4d2e"); ax.set_aspect("equal")
            ax.tick_params(colors="#888", labelsize=5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color(border_color); s.set_linewidth(1.6)

        # Pitcher bounds
        xs, ys, zs = [], [], []
        for _, wp in p["pitcher_frames"]:
            for (x, y, z) in wp.values():
                xs.append(x); ys.append(y); zs.append(z)
        if zs: ax_p.set_xlim(min(zs)-2, max(zs)+2); ax_p.set_ylim(max(0, min(ys)-1), max(ys)+2)
        # Batter bounds
        pxs = [wp.get(0, (None,None,None))[2] for _, wp in p["batter_frames"] if 0 in wp]
        cx = float(np.median([x for x in pxs if x is not None])) if pxs else 0
        ax_b.set_xlim(cx - 6, cx + 5); ax_b.set_ylim(0, 8.5)

        # PC panel
        ax_pc.set_facecolor("#1f1f1f")
        ax_pc.set_xlim(-pc_lim, pc_lim); ax_pc.set_ylim(-pc_lim, pc_lim)
        ax_pc.set_aspect("equal")
        ax_pc.axhline(0, color="#444", lw=0.4); ax_pc.axvline(0, color="#444", lw=0.4)
        ax_pc.tick_params(colors="#888", labelsize=5)
        for s in ax_pc.spines.values(): s.set_color(border_color); s.set_linewidth(1.6)

        # Title: pitch index + label + pitch type + final_dist
        ax_p.set_title(f"#{ri+1} {label} {p['pitch_type'] or '?'}  ||PC_end||={p['final_dist']:.2f}",
                       color=border_color, fontsize=7, loc="left", pad=2)

        # Artists
        p_lines = [ax_p.plot([], [], "-", lw=1.2, color="#ccc")[0] for _ in SKELETON_CONNECTIONS]
        b_lines = [ax_b.plot([], [], "-", lw=1.2, color="#ccc")[0] for _ in SKELETON_CONNECTIONS]
        bat_line = ax_b.plot([], [], "-", lw=2.2, color="#d4a04c", zorder=4)[0]
        bat_head = ax_b.scatter([], [], s=20, c="#d4a04c", edgecolor="white", lw=0.3, zorder=5)

        # Static full PC trajectory (faint) + end-dot marker
        pc_t = p["pc_times"]; pc_c = p["pc_coords"]
        ax_pc.plot(pc_c[:, 0], pc_c[:, 1], "-", color="#555", lw=0.5, alpha=0.55)
        ax_pc.scatter([pc_c[-1, 0]], [pc_c[-1, 1]], s=30, marker="*",
                      c="#ff9800", edgecolor="white", lw=0.4, zorder=3)
        pc_trail = ax_pc.plot([], [], "-", color=border_color, lw=1.0, alpha=0.85, zorder=2)[0]
        pc_dot = ax_pc.scatter([], [], s=30, c=border_color, edgecolor="white", lw=0.4, zorder=5)

        cells.append({
            "p": p, "ax_p": ax_p, "ax_b": ax_b, "ax_pc": ax_pc,
            "p_lines": p_lines, "b_lines": b_lines,
            "bat_line": bat_line, "bat_head": bat_head,
            "pc_trail": pc_trail, "pc_dot": pc_dot,
        })

    fig.suptitle(
        f"Kyle Manzardo — all {n_rows} clean pitches, sorted top→bottom by "
        f"||PC1×PC2 position at windup_onset||  •  orange star = end of pre-osc",
        color="white", fontsize=10, y=1 - 0.25 / fig_h
    )

    def update(idx):
        frac = t_grid[idx]
        artists = []
        for cell in cells:
            p = cell["p"]
            lo, hi = p["cell_lo"], p["cell_hi"]
            t = lo + frac * (hi - lo)

            p_wp = frame_at(t, p["pitcher_frames"]) or {}
            for li, (a, b) in enumerate(SKELETON_CONNECTIONS):
                ln = cell["p_lines"][li]
                if a in p_wp and b in p_wp:
                    p1, p2 = p_wp[a], p_wp[b]
                    ln.set_data([p1[2], p2[2]], [p1[1], p2[1]])
                else: ln.set_data([], [])

            b_wp = frame_at(t, p["batter_frames"]) or {}
            for li, (a, b) in enumerate(SKELETON_CONNECTIONS):
                ln = cell["b_lines"][li]
                if a in b_wp and b in b_wp:
                    p1, p2 = b_wp[a], b_wp[b]
                    ln.set_data([p1[2], p2[2]], [p1[1], p2[1]])
                else: ln.set_data([], [])
            ba = bat_at(t, p["bat_frames"])
            if ba is not None:
                head, handle = ba
                cell["bat_line"].set_data([handle[2], head[2]], [handle[1], head[1]])
                cell["bat_head"].set_offsets(np.array([[head[2], head[1]]]))
            else:
                cell["bat_line"].set_data([], [])
                cell["bat_head"].set_offsets(np.empty((0, 2)))

            # PC trajectory — animate up to current pre-osc time, freeze at windup_onset
            t_eff = min(t, p["windup_onset_t"])
            upto = np.searchsorted(p["pc_times"], t_eff, side="right")
            if upto > 0:
                cell["pc_trail"].set_data(p["pc_coords"][:upto, 0], p["pc_coords"][:upto, 1])
                cur = p["pc_coords"][upto - 1]
                cell["pc_dot"].set_offsets(np.array([[cur[0], cur[1]]]))
            else:
                cell["pc_trail"].set_data([], [])
                cell["pc_dot"].set_offsets(np.empty((0, 2)))

            artists.extend(cell["p_lines"]); artists.extend(cell["b_lines"])
            artists.extend([cell["bat_line"], cell["bat_head"],
                            cell["pc_trail"], cell["pc_dot"]])
        return artists

    print(f"Rendering → {args.out}")
    anim = FuncAnimation(fig, update, frames=n_anim, interval=1000/args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, bitrate=4000, codec="h264")
    anim.save(args.out, writer=writer, dpi=100)
    print(f"  done: {Path(args.out).stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
