"""Render Kyle Manzardo's INPLAY vs WHIFF swings — full pitcher+batter+ball
view, all cells time-aligned to knee_high.

Each row = one pitch. Left panel = pitcher, right panel = batter + bat + ball.
Cells are synced to elapsed time so knee_high is at the same instant in every
row. Yellow joints/bat = the top discriminating joints between mean inplay-pose
and mean whiff-pose at knee_high.

Layout:
  TOP    INPLAY rows (up to 4, green border)
  BOTTOM WHIFF rows (up to 3, red border)
  Each row: PITCHER | BATTER+BAT+BALL
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games, open_game
from fieldvision.skeleton import SKELETON_CONNECTIONS
from fieldvision.storage import JOINT_COLS
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

PRE_OSC_SECONDS = 5.0   # must match what was used in multi_batter_census.py

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
JOINT_NAMES = {bid: name for bid, name in JOINT_COLS}

MANZARDO = 700932

# Time window. Cell starts at pre_lo = windup_onset - PRE_OSC_SECONDS so the full
# preparatory waggle is animated. Cell ends T_POST_KH after knee_high. Cell
# starts T_PRE_KH BEFORE knee_high (a fixed offset), so knee_high lands at
# exactly the same frac in every row regardless of each pitch's
# windup_onset→knee_high gap. T_PRE_KH must cover the longest plausible gap
# (~2.5s) plus the desired pre-windup waggle (5s) = 7.5s, so every pitch shows
# at least 5s of pre-windup data.
T_PRE_KH = 7.5
T_POST_KH = 2.2
SAMPLE_HZ = 30.0
N_HIGHLIGHT = 6


def load_pitch_full(conn, batter_id, play_id, pitcher_id, release_t):
    """Load pitcher_frames, batter_frames, bat_frames, ball_frames, and events
    for one pitch — full data needed for the rich renderer."""
    joint_cols_select = ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)
    # Pull enough trailing data to cover knee_high + T_POST_KH (~2 s past release)
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
    # Knee_high–anchored cell window: fixed offsets from kh so every row aligns
    # at knee_high. (Earlier design anchored at windup_onset and let the cell
    # length vary; that made knee_high land at different fracs per row.)
    cell_lo = kh - T_PRE_KH
    cell_hi = kh + T_POST_KH

    # Batter frames for the cell window (= pre_lo to past follow-through).
    qa_lo = ev.windup_onset_t - PRE_OSC_SECONDS
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
    bat_frames_raw = [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bat_rows]
    bat_frames = filter_bat_frames(bat_frames_raw)

    # Quality check
    quality = assess_pitch_quality(batter_frames, bat_frames)
    if not quality["is_clean"]: return None

    # Ball frames during the full cell window (incoming pitch lives in T+0 to T+0.5 from release)
    ball_rows = conn.execute(
        "SELECT time_unix, ball_x, ball_y, ball_z FROM ball_frame "
        "WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (cell_lo, cell_hi)).fetchall()
    ball_frames = [(r[0], (r[1], r[2], r[3])) for r in ball_rows
                   if r[1] is not None and r[2] is not None and r[3] is not None]

    # Pre-osc-window mean pose (used to center the PC projection — matches census).
    # 66-dim with per-frame body-centroid normalization AND hands-on-bat gating
    # (both wrists within 0.6 ft of bat handle, else frame dropped).
    HAND_RT_BID, HAND_LT_BID = 28, 67
    HAND_TO_HANDLE_MAX_FT = 0.6
    pre_lo, pre_hi = ev.windup_onset_t - PRE_OSC_SECONDS, ev.windup_onset_t
    bat_times = np.array([b[0] for b in bat_frames])
    pre_vecs = []
    for t, wp in batter_frames:
        if not (pre_lo <= t <= pre_hi): continue
        joints = np.full((len(JOINT_BIDS), 3), np.nan)
        for j, bid in enumerate(JOINT_BIDS):
            if bid in wp: joints[j] = wp[bid]
        if np.any(np.isnan(joints)): continue
        if len(bat_times) == 0: continue
        bi = int(np.argmin(np.abs(bat_times - t)))
        if abs(bat_times[bi] - t) > 0.07: continue
        head = np.array(bat_frames[bi][1]); handle = np.array(bat_frames[bi][2])
        # Hands-on-bat
        hand_rt = wp.get(HAND_RT_BID); hand_lt = wp.get(HAND_LT_BID)
        if hand_rt is None or hand_lt is None: continue
        if (np.linalg.norm(np.array(hand_rt) - handle) > HAND_TO_HANDLE_MAX_FT or
            np.linalg.norm(np.array(hand_lt) - handle) > HAND_TO_HANDLE_MAX_FT):
            continue
        body_centroid = joints.mean(axis=0)
        joints_norm = (joints - body_centroid).flatten()
        head_norm = head - body_centroid
        handle_norm = handle - body_centroid
        pre_vecs.append(np.concatenate([joints_norm, head_norm, handle_norm]))
    mean_pose = np.mean(pre_vecs, axis=0) if pre_vecs else None

    return {
        "play_id": play_id,
        "windup_onset_t": ev.windup_onset_t,
        "knee_high_t": kh,
        "foot_landing_t": ev.foot_landing_t,
        "release_t": release_t,
        "cell_lo": cell_lo, "cell_hi": cell_hi,
        "pitcher_frames": pitcher_frames,
        "batter_frames": batter_frames,
        "bat_frames": bat_frames,
        "ball_frames": ball_frames,
        "mean_pose": mean_pose,
    }


def joint_pose_at(frames, target_t):
    if not frames: return None
    times = np.array([t for t, _ in frames])
    i = int(np.argmin(np.abs(times - target_t)))
    return frames[i][1]


def compute_discriminating_joints(inplay, whiff, top_k=N_HIGHLIGHT):
    def group_poses(swings):
        poses = []
        for s in swings:
            wp = joint_pose_at(s["batter_frames"], s["knee_high_t"])
            if wp is None: continue
            pelv = wp.get(0)
            if pelv is None: continue
            recentered = {bid: (pos[0]-pelv[0], pos[1]-pelv[1], pos[2]-pelv[2])
                          for bid, pos in wp.items()}
            poses.append(recentered)
        return poses
    inp = group_poses(inplay)
    whi = group_poses(whiff)
    dists = []
    for bid in JOINT_BIDS:
        a = [p[bid] for p in inp if bid in p]
        b = [p[bid] for p in whi if bid in p]
        if len(a) < 2 or len(b) < 2: continue
        d = float(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0)))
        dists.append((bid, d))
    dists.sort(key=lambda x: -x[1])
    return dists[:top_k], dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/oscillation_report/pre_pitch_preparatory_movement/manzardo_inplay_vs_whiff.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=4,
                    help="max INPLAY rows; whiff column always shows all 3")
    args = ap.parse_args()

    inplay, whiff = [], []
    for game_pk in list_games(Path(args.data_dir)):
        conn = open_game(game_pk, Path(args.data_dir))
        try:
            rows = conn.execute(
                "SELECT play_id, pitcher_id, start_time_unix, pitch_type, result_call "
                "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL",
                (MANZARDO,)).fetchall()
        except Exception:
            conn.close(); continue
        for play_id, pitcher_id, t_rel, pt, call in rows:
            if call not in ("X", "S"): continue
            d = load_pitch_full(conn, MANZARDO, play_id, pitcher_id, t_rel)
            if d is None: continue
            d.update(pitch_type=pt, result_call=call)
            (inplay if call == "X" else whiff).append(d)
        conn.close()
    print(f"Manzardo: {len(inplay)} INPLAY, {len(whiff)} WHIFF clean pitches with full data")
    if not inplay or not whiff:
        raise SystemExit("need both")

    top_disc, all_disc = compute_discriminating_joints(inplay, whiff)
    highlight_bids = set(bid for bid, _ in top_disc)
    print(f"\nTop {N_HIGHLIGHT} discriminating joints at knee_high:")
    for bid, d in top_disc:
        print(f"  {JOINT_NAMES[bid]:14s}  Δ={d:.2f} ft")

    # Load Manzardo's PC components from the census so the trajectory + phase
    # plots match the analysis pipeline. Components are fit on STANCE-only
    # frames (setup/walking-to-the-box outliers removed via MAD filter in PC space).
    census = pickle.load(open("data/oscillation_report/pre_pitch_preparatory_movement/census.pkl", "rb"))
    manzardo_key = f"{MANZARDO}_L"
    if manzardo_key not in census["results"]:
        raise SystemExit(f"{manzardo_key} not in census — run multi_batter_census.py first")
    cres = census["results"][manzardo_key]
    pca_components = cres["pca_components"]  # (5, 66)
    pc12 = pca_components[:2]                # (2, 66) — kept for compatibility
    var_explained = cres["pca_var_explained"]
    per_pitch_mean_pose = cres.get("per_pitch_mean_pose", {})
    stance_frac = cres.get("stance_frac", 1.0)
    # jPCA: rotational dynamics in PC-5 space. The dominant rotation plane is
    # the natural 2D oscillation axis for the waggle (much more meaningful than
    # PC1×PC2, which is the variance axis regardless of dynamics).
    jpca = cres.get("jpca")
    if jpca is None or not jpca["rotation_planes"]:
        raise SystemExit("jpca not in census — re-run multi_batter_census.py")
    jp_plane = jpca["rotation_planes"][0]
    jp_e1 = jp_plane["e1"]          # (5,) basis in PC-5 space
    jp_e2 = jp_plane["e2"]          # (5,)
    jp_freq = jp_plane["freq_hz"]
    jp_xmean = jpca["x_mean"]       # (5,) mean across pooled PC-trajectories
    print(f"\nLoaded jPCA from census  ({len(jpca['rotation_planes'])} rotation planes; "
          f"dominant: ω={jp_plane['omega']:.2f} rad/s = {jp_freq:.2f} Hz)")
    print(f"  Underlying PCA: PC1+PC2 = {var_explained[0]*100:.0f}%+{var_explained[1]*100:.0f}% of stance variance, "
          f"stance frames = {stance_frac*100:.0f}% of pre-osc")
    pc5 = pca_components[:5]  # (5, 66)

    inplay_show = inplay[:args.max_rows]
    whiff_show = whiff[:3]
    n_inplay = len(inplay_show); n_whiff = len(whiff_show)
    n_rows = n_inplay + n_whiff

    # Pre-compute per-pitch jPCA rotation-plane projection.
    # Pipeline: posture → centered (per-pitch mean) → PC-5 → recentered (jpca
    # x_mean) → dotted with (e1, e2) → 2D rotation-plane coords.
    def project_pc(p):
        """Returns (times, jpca_coords) where jpca_coords is (n_frames, 2).

        Projects pre-osc-waggle frames into the jPCA dominant rotation plane.
        Within this plane the waggle is, by construction, the most rotational
        component of the dynamics — angle = preparatory phase.
        """
        # Build the 66-dim posture sequence within pre-osc window — with
        # per-frame body-centroid normalization AND hands-on-bat gating.
        HAND_RT_BID, HAND_LT_BID = 28, 67
        HAND_TO_HANDLE_MAX_FT = 1.0
        pre_lo = p["windup_onset_t"] - PRE_OSC_SECONDS
        pre_hi = p["windup_onset_t"]
        bat_times = np.array([b[0] for b in p["bat_frames"]])
        rows = []
        for t, wp in p["batter_frames"]:
            if not (pre_lo <= t <= pre_hi): continue
            joints = np.full((len(JOINT_BIDS), 3), np.nan)
            for j, bid in enumerate(JOINT_BIDS):
                if bid in wp: joints[j] = wp[bid]
            if np.any(np.isnan(joints)): continue
            if len(bat_times) == 0: continue
            bi = int(np.argmin(np.abs(bat_times - t)))
            if abs(bat_times[bi] - t) > 0.07: continue
            head = np.array(p["bat_frames"][bi][1]); handle = np.array(p["bat_frames"][bi][2])
            hand_rt = wp.get(HAND_RT_BID); hand_lt = wp.get(HAND_LT_BID)
            if hand_rt is None or hand_lt is None: continue
            if (np.linalg.norm(np.array(hand_rt) - handle) > HAND_TO_HANDLE_MAX_FT or
                np.linalg.norm(np.array(hand_lt) - handle) > HAND_TO_HANDLE_MAX_FT):
                continue
            body_centroid = joints.mean(axis=0)
            joints_norm = (joints - body_centroid).flatten()
            head_norm = head - body_centroid
            handle_norm = handle - body_centroid
            rows.append((t, np.concatenate([joints_norm, head_norm, handle_norm])))
        if len(rows) < 5: return None
        ts = np.array([t for t, _ in rows])
        vecs = np.array([v for _, v in rows])

        # Use the census's per-pitch stance mean_pose if available; otherwise
        # fall back to the pitch's own all-pre-osc mean.
        mp = per_pitch_mean_pose.get(p["play_id"])
        if mp is None: mp = p["mean_pose"]
        if mp is None: return None
        # Project: 66-dim posture → 5D PC → 2D jPCA rotation-plane coords.
        pc5_traj = (vecs - mp) @ pc5.T                 # (T, 5)
        centered = pc5_traj - jp_xmean[None, :]        # (T, 5)
        proj = np.stack([centered @ jp_e1, centered @ jp_e2], axis=1)  # (T, 2)

        # Template-matching stance filter: keep frames whose centroid-normalized
        # posture has cosine similarity > STANCE_SIM_THRESHOLD to the per-pitch
        # median template. Median is robust to sporadic divergent frames
        # (plate-tap, bat-point, breath).
        STANCE_SIM_THRESHOLD = 0.95
        if len(vecs) < 10:
            return ts, proj
        template = np.median(vecs, axis=0)
        t_norm = np.linalg.norm(template) + 1e-9
        v_norm = np.linalg.norm(vecs, axis=1) + 1e-9
        sims = (vecs @ template) / (v_norm * t_norm)
        keep = sims > STANCE_SIM_THRESHOLD
        if keep.sum() < 5:
            return ts, proj
        return ts[keep], proj[keep]

    for p in inplay_show + whiff_show:
        result = project_pc(p)
        p["pc_times"] = result[0] if result else None
        p["pc_coords"] = result[1] if result else None
        if result is not None:
            # phase per frame
            p["phases"] = np.arctan2(result[1][:, 1], result[1][:, 0])
        else:
            p["phases"] = None

    # Single shared axis bound across every pitch — these PCs are the SAME basis
    # (fit once on all of Manzardo's pre-osc frames in the census), so plotting
    # them in the same state space is meaningful. Use the 99th percentile of
    # |coord| (square, equal PC1/PC2) so one outlier frame can't squash everything.
    all_pc = np.vstack([p["pc_coords"] for p in inplay_show + whiff_show
                        if p["pc_coords"] is not None])
    pc_lim = float(np.percentile(np.abs(all_pc), 99)) * 1.1
    print(f"  PC axis range: ±{pc_lim:.2f} (shared across all rows)")

    # Compute phase at windup_onset and at knee_high for every shown pitch by
    # projecting that single frame onto the saved (stance-only) PCs. The phase
    # is the angle of the projected point in PC1×PC2.
    def _nearest_frame(frames, target_t, max_dt=0.07):
        if not frames: return None
        ts_arr = np.array([f[0] for f in frames])
        i = int(np.argmin(np.abs(ts_arr - target_t)))
        if abs(ts_arr[i] - target_t) > max_dt: return None
        return frames[i][1]

    def project_frame_at(p, target_t):
        target_frame = _nearest_frame(p["batter_frames"], target_t)
        if target_frame is None: return None
        joints = np.full((len(JOINT_BIDS), 3), np.nan)
        for j, bid in enumerate(JOINT_BIDS):
            if bid in target_frame: joints[j] = target_frame[bid]
        if np.any(np.isnan(joints)): return None
        # Bat: same lookup pattern (returns the (head, handle) tuple)
        if not p["bat_frames"]: return None
        bat_ts = np.array([b[0] for b in p["bat_frames"]])
        bi = int(np.argmin(np.abs(bat_ts - target_t)))
        if abs(bat_ts[bi] - target_t) > 0.07: return None
        head = np.array(p["bat_frames"][bi][1]); handle = np.array(p["bat_frames"][bi][2])
        body_centroid = joints.mean(axis=0)
        joints_norm = (joints - body_centroid).flatten()
        head_norm = head - body_centroid
        handle_norm = handle - body_centroid
        vec = np.concatenate([joints_norm, head_norm, handle_norm])
        mp = per_pitch_mean_pose.get(p["play_id"])
        if mp is None: mp = p["mean_pose"]
        if mp is None: return None
        pc5_v = (vec - mp) @ pc5.T              # (5,)
        c = pc5_v - jp_xmean                    # (5,)
        return np.array([c @ jp_e1, c @ jp_e2])  # (2,)

    for p in inplay_show + whiff_show:
        pc_wuo = project_frame_at(p, p["windup_onset_t"])
        pc_kh = project_frame_at(p, p["knee_high_t"])
        p["phase_wuo"] = float(np.arctan2(pc_wuo[1], pc_wuo[0])) if pc_wuo is not None else None
        p["phase_kh"] = float(np.arctan2(pc_kh[1], pc_kh[0])) if pc_kh is not None else None

    # Figure: 4 columns (pitcher | batter | PC trajectory | phase circle), plus
    # a final summary row spanning all columns with whiff-vs-inplay phase plots.
    fig_w = 18
    summary_h = 4.5
    DPI = 100  # at dpi=100, fig_h * 100 = pixels; integer pixel arithmetic
    fig_h = 3.4 * n_rows + summary_h + 2
    # Round up to nearest even tenth so dpi=100 gives even pixel height
    import math
    fig_h = math.ceil(fig_h * 5.0) / 5.0  # nearest 0.2 inch → 20-pixel granularity
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a1a")
    gs = fig.add_gridspec(n_rows + 1, 4, width_ratios=[1.0, 1.4, 1.0, 0.85],
                          height_ratios=[1.0] * n_rows + [summary_h / 3.4],
                          hspace=0.32, wspace=0.18,
                          left=0.03, right=0.98, top=0.93, bottom=0.04)

    cells = []
    for ri, p in enumerate(inplay_show + whiff_show):
        is_inplay = ri < n_inplay
        border_color = "#4caf50" if is_inplay else "#f44336"
        label = "INPLAY" if is_inplay else "WHIFF"
        ax_p = fig.add_subplot(gs[ri, 0])
        ax_b = fig.add_subplot(gs[ri, 1])
        ax_pc = fig.add_subplot(gs[ri, 2])
        ax_phase = fig.add_subplot(gs[ri, 3], projection="polar")

        for ax, who, title in ((ax_p, "pitcher", "PITCHER"), (ax_b, "batter", "BATTER + BAT + BALL")):
            ax.set_facecolor("#2e4d2e"); ax.set_aspect("equal")
            ax.tick_params(colors="#888", labelsize=6)
            for s in ax.spines.values():
                s.set_color(border_color); s.set_linewidth(2)
            ax.set_xticks([]); ax.set_yticks([])

        # PC trajectory panel — shared scale across rows
        ax_pc.set_facecolor("#1f1f1f")
        ax_pc.set_xlim(-pc_lim, pc_lim); ax_pc.set_ylim(-pc_lim, pc_lim)
        ax_pc.set_aspect("equal")
        ax_pc.axhline(0, color="#444", lw=0.6); ax_pc.axvline(0, color="#444", lw=0.6)
        ax_pc.tick_params(colors="#888", labelsize=6)
        for s in ax_pc.spines.values():
            s.set_color(border_color); s.set_linewidth(2)
        ax_pc.set_xlabel("jPC1 (rot. plane)", color="#aaa", fontsize=7, labelpad=1)
        ax_pc.set_ylabel("jPC2 (rot. plane)", color="#aaa", fontsize=7, labelpad=1)
        ax_pc.set_title(f"jPCA rotation plane ({jp_freq:.2f} Hz)",
                        color=border_color, fontsize=9, loc="left")

        # Phase panel (polar)
        ax_phase.set_facecolor("#1f1f1f")
        ax_phase.set_ylim(0, 1.1)
        ax_phase.set_yticks([])
        ax_phase.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax_phase.set_xticklabels(["0", "", "π/2", "", "π", "", "3π/2", ""],
                                 color="#888", fontsize=6)
        ax_phase.tick_params(colors="#888")
        ax_phase.grid(color="#333", lw=0.5)
        ax_phase.spines["polar"].set_color(border_color)
        ax_phase.spines["polar"].set_linewidth(2)
        ax_phase.set_title("phase", color=border_color, fontsize=9, loc="left", pad=2)

        # Pitcher cell auto-bounds
        def actor_bounds(frames):
            xs, ys, zs = [], [], []
            for _, wp in frames:
                for (x, y, z) in wp.values():
                    xs.append(x); ys.append(y); zs.append(z)
            if not xs: return (-5, 5, 0, 8)
            return (min(zs)-2, max(zs)+2, max(0, min(ys)-1), max(ys)+2)
        pb = actor_bounds(p["pitcher_frames"])
        ax_p.set_xlim(pb[0], pb[1]); ax_p.set_ylim(pb[2], pb[3])

        # Batter cell: center on median pelvis Z, fixed range
        pxs = [wp.get(0, (None,None,None))[2] for _, wp in p["batter_frames"] if 0 in wp]
        cx = float(np.median([x for x in pxs if x is not None])) if pxs else 0
        # Include ball flight (ball comes from negative z toward 0 from pitcher) — extend left
        ax_b.set_xlim(cx - 8, cx + 5)
        ax_b.set_ylim(0, 8.5)

        ax_p.set_title(f"#{ri+1} {label}  {p['pitch_type'] or '?'}  PITCHER",
                       color=border_color, fontsize=9, loc="left")
        ax_b.set_title(f"BATTER  •  knee_high anchor",
                       color=border_color, fontsize=9, loc="left")

        # Build artists
        p_lines = [ax_p.plot([], [], "-", lw=1.6, color="#ccc", solid_capstyle="round")[0]
                   for _ in SKELETON_CONNECTIONS]
        p_dots = ax_p.scatter([], [], s=10, c="#80cbc4")

        # Batter: highlight discriminating joints
        b_lines = []
        for a, b in SKELETON_CONNECTIONS:
            is_hi = a in highlight_bids or b in highlight_bids
            ln = ax_b.plot([], [], "-",
                           lw=2.2 if is_hi else 1.4,
                           color="#fbc02d" if is_hi else "#bbb",
                           solid_capstyle="round")[0]
            b_lines.append(ln)
        b_normal_dots = ax_b.scatter([], [], s=10, c="#888")
        b_high_dots = ax_b.scatter([], [], s=45, c="#fbc02d",
                                    edgecolor="white", lw=0.5, zorder=5)
        bat_line = ax_b.plot([], [], "-", lw=3.0, color="#d4a04c",
                              solid_capstyle="round", zorder=4)[0]
        bat_head = ax_b.scatter([], [], s=40, c="#d4a04c",
                                 edgecolor="white", lw=0.4, zorder=5)

        # Ball — show 6-frame trail
        ball_trail = ax_b.plot([], [], "o-", lw=1.2, color="#ff6f00",
                                markersize=6, alpha=0.85, zorder=6)[0]

        # knee_high marker on bottom of each cell
        kh = p["knee_high_t"]
        win_lo, win_hi = p["cell_lo"], p["cell_hi"]
        kh_frac = (kh - win_lo) / (win_hi - win_lo)
        for ax in (ax_p, ax_b):
            xm, xx = ax.get_xlim()
            kh_x = xm + kh_frac * (xx - xm)
            ax.axvline(kh_x, color="#ff9800", lw=1.2, ls=":", alpha=0.4, zorder=1)
        # Progress bar for each panel
        prog_p = ax_p.plot([ax_p.get_xlim()[0], ax_p.get_xlim()[0]],
                           [ax_p.get_ylim()[0]+0.1, ax_p.get_ylim()[0]+0.1],
                           color="white", lw=1.8, zorder=2)[0]
        prog_b = ax_b.plot([ax_b.get_xlim()[0], ax_b.get_xlim()[0]],
                           [0.08, 0.08], color="white", lw=1.8, zorder=2)[0]

        # PC trajectory artists: static faded full trail + animated bright trail + current dot.
        # Also a marker at the knee_high frame as reference.
        pc_t = p.get("pc_times")
        pc_c = p.get("pc_coords")
        # PC artists: animated trail + current dot + windup_onset reference star (no pre-drawn trace).
        if pc_t is not None and pc_c is not None and len(pc_c) > 1:
            wu_i = int(np.argmin(np.abs(pc_t - p["windup_onset_t"])))
            ax_pc.scatter([pc_c[wu_i, 0]], [pc_c[wu_i, 1]], s=40, marker="*",
                          c="#ff9800", edgecolor="white", lw=0.4, zorder=3)
        pc_trail = ax_pc.plot([], [], "-", color=border_color, lw=1.6, alpha=0.85, zorder=2)[0]
        pc_dot = ax_pc.scatter([], [], s=60, c=border_color,
                                edgecolor="white", lw=0.6, zorder=5)

        # Phase artists: windup_onset reference star + animated cursor (no static scatter).
        ph = p.get("phases")
        if ph is not None and len(ph) > 0:
            wu_i = int(np.argmin(np.abs(p["pc_times"] - p["windup_onset_t"])))
            ax_phase.scatter([ph[wu_i]], [0.95], s=70, marker="*",
                              c="#ff9800", edgecolor="white", lw=0.4, zorder=3)
        phase_trail = ax_phase.plot([], [], "-", color=border_color, lw=1.4, alpha=0.7, zorder=2)[0]
        phase_dot = ax_phase.scatter([], [], s=80, c=border_color,
                                       edgecolor="white", lw=0.6, zorder=5)

        cells.append({
            "p": p, "is_inplay": is_inplay,
            "ax_p": ax_p, "ax_b": ax_b, "ax_pc": ax_pc, "ax_phase": ax_phase,
            "p_lines": p_lines, "p_dots": p_dots,
            "b_lines": b_lines, "b_normal_dots": b_normal_dots, "b_high_dots": b_high_dots,
            "bat_line": bat_line, "bat_head": bat_head, "ball_trail": ball_trail,
            "pc_trail": pc_trail, "pc_dot": pc_dot,
            "phase_trail": phase_trail, "phase_dot": phase_dot,
            "prog_p": prog_p, "prog_b": prog_b,
        })

    # ===== Summary row: aggregate phase distribution, whiff vs inplay =====
    # Use ALL clean Manzardo pitches (not just rows shown above) for statistical
    # weight: pulled from the census's phases_per_pitch.
    def collect_phases(event_name):
        ip_phases, wh_phases = [], []
        # From shown pitches (precomputed above):
        for p in inplay_show:
            phi = p[f"phase_{'wuo' if event_name == 'windup_onset' else 'kh'}"]
            if phi is not None: ip_phases.append(phi)
        for p in whiff_show:
            phi = p[f"phase_{'wuo' if event_name == 'windup_onset' else 'kh'}"]
            if phi is not None: wh_phases.append(phi)
        # Augment with remaining inplay/whiff pitches present in the data but
        # not rendered (only first 4 inplay + 3 whiff are shown as rows).
        extra_inplay = inplay[len(inplay_show):]
        extra_whiff = whiff[len(whiff_show):]
        for p in extra_inplay + extra_whiff:
            pc = project_frame_at(p, p["windup_onset_t"] if event_name == "windup_onset"
                                  else p["knee_high_t"])
            if pc is None: continue
            phi = float(np.arctan2(pc[1], pc[0]))
            (ip_phases if p["result_call"] == "X" else wh_phases).append(phi)
        return np.array(ip_phases), np.array(wh_phases)

    def mean_resultant(phases):
        if len(phases) == 0: return 0.0, 0.0
        z = np.mean(np.exp(1j * phases))
        return float(np.abs(z)), float(np.angle(z))

    def draw_phase_panel(ax, ip_phases, wh_phases, title):
        ax.set_facecolor("#1f1f1f")
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax.set_xticklabels(["0", "", "π/2", "", "π", "", "3π/2", ""],
                           color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa")
        ax.grid(color="#333", lw=0.4)
        ax.spines["polar"].set_color("#666")
        # Pitch dots
        if len(ip_phases):
            ax.scatter(ip_phases, np.full_like(ip_phases, 0.85), s=70,
                       c="#4caf50", edgecolor="white", lw=0.6, alpha=0.9, zorder=3,
                       label=f"inplay (n={len(ip_phases)})")
        if len(wh_phases):
            ax.scatter(wh_phases, np.full_like(wh_phases, 1.0), s=70,
                       c="#f44336", edgecolor="white", lw=0.6, alpha=0.9, zorder=3,
                       label=f"whiff (n={len(wh_phases)})")
        # Mean resultant vectors
        ip_R, ip_theta = mean_resultant(ip_phases)
        wh_R, wh_theta = mean_resultant(wh_phases)
        ax.annotate("", xy=(ip_theta, ip_R), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#4caf50", lw=2.4))
        ax.annotate("", xy=(wh_theta, wh_R), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#f44336", lw=2.4))
        ax.set_title(
            f"phase @ {title}\n"
            f"inplay R={ip_R:.2f} θ={np.degrees(ip_theta):.0f}°  ·  "
            f"whiff R={wh_R:.2f} θ={np.degrees(wh_theta):.0f}°",
            color="white", fontsize=10, pad=12
        )
        ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.05),
                  facecolor="#1a1a1a", edgecolor="#555",
                  labelcolor="white", fontsize=8)

    ip_wuo, wh_wuo = collect_phases("windup_onset")
    ip_kh, wh_kh = collect_phases("knee_high")
    # Two polar plots in the summary row: span cols 0-1 and cols 2-3
    ax_sum_l = fig.add_subplot(gs[n_rows, 0:2], projection="polar")
    ax_sum_r = fig.add_subplot(gs[n_rows, 2:4], projection="polar")
    draw_phase_panel(ax_sum_l, ip_wuo, wh_wuo, "windup_onset")
    draw_phase_panel(ax_sum_r, ip_kh, wh_kh, "knee_high")

    fig.suptitle(
        f"Kyle Manzardo  |  INPLAY (green, n={n_inplay} shown of {len(inplay)}) vs WHIFF (red, n={n_whiff})\n"
        f"Each row time-aligned so knee_high coincides — orange dotted vertical line.\n"
        f"Yellow joints/lines = top {N_HIGHLIGHT} discriminators at knee_high "
        f"({', '.join(JOINT_NAMES[bid] for bid, _ in top_disc)})\n"
        f"Bottom row: per-pitch phase angle in the jPCA rotation plane ({jp_freq:.2f} Hz) at the event "
        f"— arrow = mean resultant vector, R = concentration.",
        color="white", fontsize=11, y=0.985
    )

    # Animation: cells synced by elapsed-fraction over their own [cell_lo, cell_hi]
    # window. Cell durations vary (pre_osc + windup_onset→knee_high gap + T_POST_KH)
    # so we use the max as the wall-clock animation length. Shorter cells just play
    # at slightly slower wall-clock time, which is barely perceptible.
    total_t = max(p["cell_hi"] - p["cell_lo"] for p in inplay_show + whiff_show)
    n_anim = int(total_t * args.fps)
    t_grid = np.linspace(0, 1, n_anim)
    print(f"  cell duration: {total_t:.1f}s (= {PRE_OSC_SECONDS}s waggle + variable windup→knee_high gap + {T_POST_KH}s post-knee_high)")

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

    def ball_trail_to(t, ball_frames, n=6):
        """Return positions of ball at frames whose times ∈ [t - 0.2s, t]."""
        if not ball_frames: return []
        ts = np.array([b[0] for b in ball_frames])
        mask = (ts > t - 0.2) & (ts <= t)
        return [ball_frames[i][1] for i in np.where(mask)[0]][-n:]

    def update(idx):
        frac = t_grid[idx]
        artists = []
        for cell in cells:
            p = cell["p"]
            lo, hi = p["cell_lo"], p["cell_hi"]
            t = lo + frac * (hi - lo)

            # Pitcher
            p_wp = frame_at(t, p["pitcher_frames"]) or {}
            for li, (a, b) in enumerate(SKELETON_CONNECTIONS):
                ln = cell["p_lines"][li]
                if a in p_wp and b in p_wp:
                    p1, p2 = p_wp[a], p_wp[b]
                    ln.set_data([p1[2], p2[2]], [p1[1], p2[1]])
                else: ln.set_data([], [])
            pts = np.array([(pp[2], pp[1]) for pp in p_wp.values()]) if p_wp else np.empty((0,2))
            cell["p_dots"].set_offsets(pts)

            # Batter
            b_wp = frame_at(t, p["batter_frames"]) or {}
            for li, (a, b) in enumerate(SKELETON_CONNECTIONS):
                ln = cell["b_lines"][li]
                if a in b_wp and b in b_wp:
                    p1, p2 = b_wp[a], b_wp[b]
                    ln.set_data([p1[2], p2[2]], [p1[1], p2[1]])
                else: ln.set_data([], [])
            norm_pts = []; high_pts = []
            for bid, pp in b_wp.items():
                if bid in highlight_bids: high_pts.append((pp[2], pp[1]))
                else: norm_pts.append((pp[2], pp[1]))
            cell["b_normal_dots"].set_offsets(np.array(norm_pts) if norm_pts else np.empty((0,2)))
            cell["b_high_dots"].set_offsets(np.array(high_pts) if high_pts else np.empty((0,2)))

            ba = bat_at(t, p["bat_frames"])
            if ba is not None:
                head, handle = ba
                cell["bat_line"].set_data([handle[2], head[2]], [handle[1], head[1]])
                cell["bat_head"].set_offsets(np.array([[head[2], head[1]]]))
            else:
                cell["bat_line"].set_data([], [])
                cell["bat_head"].set_offsets(np.empty((0,2)))

            # Ball trail (in flight ~0.4s after release)
            trail = ball_trail_to(t, p["ball_frames"])
            if trail:
                # Render as series of (Z, Y) points
                zs = [tt[2] for tt in trail]
                ys = [tt[1] for tt in trail]
                cell["ball_trail"].set_data(zs, ys)
            else:
                cell["ball_trail"].set_data([], [])

            # PC trajectory + phase: trail up to current time, clamped at windup_onset.
            # After windup_onset the cursor freezes — analysis ignores everything past it.
            pc_t = p.get("pc_times"); pc_c = p.get("pc_coords"); ph = p.get("phases")
            if pc_t is not None and pc_c is not None and len(pc_c) > 0:
                t_eff = min(t, p["windup_onset_t"])
                upto = np.searchsorted(pc_t, t_eff, side="right")
                if upto > 0:
                    cell["pc_trail"].set_data(pc_c[:upto, 0], pc_c[:upto, 1])
                    cur = pc_c[upto-1]
                    cell["pc_dot"].set_offsets(np.array([[cur[0], cur[1]]]))
                    if ph is not None:
                        # Short rolling window to avoid ugly 2π wrap-arounds
                        lo_i = max(0, upto - 15)
                        cell["phase_trail"].set_data(ph[lo_i:upto],
                                                     np.full(upto - lo_i, 0.7))
                        cell["phase_dot"].set_offsets(np.array([[ph[upto-1], 0.7]]))
                else:
                    cell["pc_trail"].set_data([], [])
                    cell["pc_dot"].set_offsets(np.empty((0, 2)))
                    cell["phase_trail"].set_data([], [])
                    cell["phase_dot"].set_offsets(np.empty((0, 2)))

            # Progress bars
            for ax_key, prog in (("ax_p", "prog_p"), ("ax_b", "prog_b")):
                ax = cell[ax_key]
                xm, xx = ax.get_xlim()
                cur_x = xm + frac * (xx - xm)
                ym = (ax.get_ylim()[0] + 0.1) if ax_key == "ax_p" else 0.08
                cell[prog].set_data([xm, cur_x], [ym, ym])

            artists.extend(cell["p_lines"]); artists.append(cell["p_dots"])
            artists.extend(cell["b_lines"])
            artists.extend([cell["b_normal_dots"], cell["b_high_dots"],
                            cell["bat_line"], cell["bat_head"], cell["ball_trail"],
                            cell["pc_trail"], cell["pc_dot"],
                            cell["phase_trail"], cell["phase_dot"],
                            cell["prog_p"], cell["prog_b"]])
        return artists

    print(f"\nRendering {n_anim} frames @ {args.fps}fps → {args.out}")
    anim = FuncAnimation(fig, update, frames=n_anim, interval=1000/args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, bitrate=3500, codec="h264")
    anim.save(args.out, writer=writer, dpi=DPI)
    print(f"  done: {Path(args.out).stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
