"""Diagnostic: render a heatmap of Manzardo's PC1×PC2 occupancy across ALL
his pre-osc frames, pooled from every pitch. Overlays each pitch as a faint
trajectory line so we can see how setup arms extend from the dense stance
cluster.

Output: data/oscillation_report/pre_pitch_preparatory_movement/manzardo_pc_heatmap.png
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games, open_game
from fieldvision.storage import JOINT_COLS
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

PRE_OSC_SECONDS = 5.0
SAMPLE_HZ = 30.0
MANZARDO = 700932
JOINT_BIDS = [bid for bid, _ in JOINT_COLS]


def load_pitch_posture(conn, batter_id, pitcher_id, release_t):
    """Return (times, posture_vecs (n, 66)) for the pre-osc window of one pitch."""
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
    if ev.windup_onset_t is None: return None
    pre_lo, pre_hi = ev.windup_onset_t - PRE_OSC_SECONDS, ev.windup_onset_t

    b_rows_raw = load_clean_batter_actor_frames(
        conn, batter_id, pre_lo - 0.2, pre_hi + 0.2, joint_cols_select)
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
        (pre_lo - 0.2, pre_hi + 0.2)).fetchall()
    bat_records = filter_bat_frames(
        [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bat_rows])
    quality = assess_pitch_quality(batter_frames, bat_records)
    if not quality["is_clean"]: return None

    bat_times = np.array([b[0] for b in bat_records])
    rows = []
    for t, wp in batter_frames:
        if not (pre_lo <= t <= pre_hi): continue
        joints = np.full((len(JOINT_BIDS), 3), np.nan)
        for j, bid in enumerate(JOINT_BIDS):
            if bid in wp: joints[j] = wp[bid]
        if np.any(np.isnan(joints)): continue
        if len(bat_times) == 0: continue
        bi = int(np.argmin(np.abs(bat_times - t)))
        if abs(bat_times[bi] - t) > 0.07: continue
        head = np.array(bat_records[bi][1]); handle = np.array(bat_records[bi][2])
        body_centroid = joints.mean(axis=0)
        joints_norm = (joints - body_centroid).flatten()
        head_norm = head - body_centroid
        handle_norm = handle - body_centroid
        rows.append((t, np.concatenate([joints_norm, head_norm, handle_norm])))
    if len(rows) < 5: return None
    ts = np.array([r[0] for r in rows])
    vecs = np.array([r[1] for r in rows])
    return ts, vecs


def main():
    out_path = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_pc_heatmap.png"

    census = pickle.load(open("data/oscillation_report/pre_pitch_preparatory_movement/census.pkl", "rb"))
    manzardo_key = f"{MANZARDO}_L"
    cres = census["results"][manzardo_key]
    pca_components = cres["pca_components"]
    pc12 = pca_components[:2]  # (2, 66)
    var_explained = cres["pca_var_explained"]
    per_pitch_mean_pose = cres.get("per_pitch_mean_pose", {})
    print(f"PC1+PC2 = {var_explained[0]*100:.0f}% + {var_explained[1]*100:.0f}% "
          f"= {(var_explained[0]+var_explained[1])*100:.0f}% of stance variance")

    # Iterate all of Manzardo's pitches and collect projections
    per_pitch_proj = []  # list of (call, ts, proj)
    pitch_meta = cres["pitches_meta"]
    play_id_to_call = {pm["play_id"]: pm["result_call"] for pm in pitch_meta}

    data_dir = Path(os.environ.get("FV_DATA_DIR", "data"))
    for game_pk in list_games(data_dir):
        conn = open_game(game_pk, data_dir)
        try:
            rows = conn.execute(
                "SELECT play_id, pitcher_id, start_time_unix "
                "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL",
                (MANZARDO,)).fetchall()
        except Exception:
            conn.close(); continue
        for play_id, pitcher_id, t_rel in rows:
            if play_id not in play_id_to_call: continue
            r = load_pitch_posture(conn, MANZARDO, pitcher_id, t_rel)
            if r is None: continue
            ts, vecs = r
            mp = per_pitch_mean_pose.get(play_id)
            if mp is None: continue
            proj = (vecs - mp) @ pc12.T
            per_pitch_proj.append((play_id_to_call[play_id], ts, proj))
        conn.close()

    if not per_pitch_proj:
        raise SystemExit("no pitches loaded")

    all_pts = np.vstack([proj for (_, _, proj) in per_pitch_proj])
    n_total = len(all_pts)
    n_pitches = len(per_pitch_proj)
    print(f"Pooled {n_total} pre-osc frames from {n_pitches} pitches")

    # 2D heatmap with light smoothing
    lim = float(np.percentile(np.abs(all_pts), 99)) * 1.1
    bins = 60
    h, xe, ye = np.histogram2d(all_pts[:, 0], all_pts[:, 1],
                                bins=bins, range=[[-lim, lim], [-lim, lim]])
    h_log = np.log1p(h)
    h_smooth = gaussian_filter(h_log, sigma=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="#1a1a1a")

    # Panel 1: heatmap
    ax = axes[0]
    ax.set_facecolor("#0a0a0a")
    im = ax.imshow(h_smooth.T, origin="lower", extent=[-lim, lim, -lim, lim],
                   cmap="inferno", aspect="equal", interpolation="bilinear")
    ax.set_xlabel("PC1", color="white"); ax.set_ylabel("PC2", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.axhline(0, color="#555", lw=0.5); ax.axvline(0, color="#555", lw=0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("log(1 + frame count)", color="white")
    cb.ax.tick_params(colors="white")
    ax.set_title(f"Manzardo PC1×PC2 occupancy heatmap\n"
                 f"{n_total} pre-osc frames pooled across {n_pitches} pitches",
                 color="white", fontsize=10)

    # Panel 2: per-pitch trajectories overlaid, colored by outcome
    ax = axes[1]
    ax.set_facecolor("#0a0a0a")
    color_for = {"X": "#4caf50", "S": "#f44336", "F": "#fbc02d",
                 "B": "#42a5f5", "C": "#90caf9", "*B": "#42a5f5"}
    for (call, ts, proj) in per_pitch_proj:
        c = color_for.get(call, "#aaaaaa")
        ax.plot(proj[:, 0], proj[:, 1], "-", color=c, lw=0.5, alpha=0.5)
        ax.scatter([proj[-1, 0]], [proj[-1, 1]], s=15, c=c, edgecolor="white",
                   lw=0.3, zorder=3)  # mark windup_onset position
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#555", lw=0.5); ax.axvline(0, color="#555", lw=0.5)
    ax.set_xlabel("PC1", color="white"); ax.set_ylabel("PC2", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.set_title("Per-pitch trajectories (dot = windup_onset position)\n"
                 "green=inplay  red=whiff  yellow=foul  blue=take",
                 color="white", fontsize=10)

    fig.suptitle(f"Manzardo PC space — diagnostic for stance vs setup separation",
                 color="white", fontsize=12)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
