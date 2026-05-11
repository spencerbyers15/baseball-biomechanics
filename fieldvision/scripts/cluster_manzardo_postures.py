"""Behavioral-segmentation diagnostic on Manzardo's pooled pre-osc PC
projection. Inspired by B-SOiD's approach: cluster the trajectory in
low-dim space, then pick out the dense behavioral mode that corresponds
to the in-stance preparatory waggle.

HDBSCAN with a min-cluster-size of ~5% of pooled frames so the stance
cluster (largest, densest, centered near origin) emerges and setup
arms are either separated into their own clusters or labeled as noise.

Outputs:
  - data/oscillation_report/pre_pitch_preparatory_movement/manzardo_clusters.png  (cluster scatter +
    heatmap with stance-cluster highlighted)
  - prints per-pitch "stance entry" timing — the first frame after which
    the batter is durably in the stance cluster.
"""

from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
    return ts, vecs, ev.windup_onset_t


def main():
    out_png = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_clusters.png"

    census = pickle.load(open("data/oscillation_report/pre_pitch_preparatory_movement/census.pkl", "rb"))
    cres = census["results"][f"{MANZARDO}_L"]
    pc12 = cres["pca_components"][:2]
    per_pitch_mean_pose = cres.get("per_pitch_mean_pose", {})
    keep_play_ids = set(per_pitch_mean_pose.keys())

    per_pitch = []  # list of (play_id, ts, proj, windup_onset)
    for db_path in sorted(Path("data").glob("fv_*.sqlite")):
        if "registry" in db_path.name or "backup" in db_path.name: continue
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT play_id, pitcher_id, start_time_unix "
                "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL",
                (MANZARDO,)).fetchall()
        except sqlite3.OperationalError:
            conn.close(); continue
        for play_id, pitcher_id, t_rel in rows:
            if play_id not in keep_play_ids: continue
            r = load_pitch_posture(conn, MANZARDO, pitcher_id, t_rel)
            if r is None: continue
            ts, vecs, wuo = r
            mp = per_pitch_mean_pose[play_id]
            proj = (vecs - mp) @ pc12.T
            per_pitch.append((play_id, ts, proj, wuo))
        conn.close()

    all_proj = np.vstack([proj for (_, _, proj, _) in per_pitch])
    pitch_ids = np.concatenate([[pi] * len(p) for pi, (_, _, p, _) in enumerate(per_pitch)])
    print(f"Pooled {len(all_proj)} frames from {len(per_pitch)} pitches")

    # Cluster pooled PC1×PC2 with HDBSCAN. min_cluster_size = 5% of frames.
    min_cs = max(40, len(all_proj) // 20)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=10)
    labels = clusterer.fit_predict(all_proj)
    n_clusters = labels.max() + 1
    print(f"HDBSCAN: {n_clusters} clusters, "
          f"{(labels == -1).sum()} noise ({(labels == -1).sum() / len(labels) * 100:.1f}%)")

    # Identify the stance cluster: largest cluster nearest origin.
    cluster_info = []
    for c in range(n_clusters):
        mask = labels == c
        center = all_proj[mask].mean(axis=0)
        dist_from_origin = float(np.linalg.norm(center))
        cluster_info.append((c, mask.sum(), dist_from_origin, center))
        print(f"  cluster {c}: n={mask.sum()}, center=({center[0]:.2f}, {center[1]:.2f}), "
              f"|center|={dist_from_origin:.2f}")
    if cluster_info:
        # Stance = nearest-to-origin cluster (assumes origin = per-pitch stance centroid)
        stance_id = min(cluster_info, key=lambda x: x[2])[0]
        print(f"\nStance cluster: {stance_id}")
    else:
        stance_id = -1
        print("\nNo clusters found")

    # === Render diagnostic ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor="#1a1a1a")
    lim = float(np.percentile(np.abs(all_proj), 99)) * 1.1

    # Panel 1: heatmap (log-scaled)
    ax = axes[0]
    ax.set_facecolor("#0a0a0a")
    h, xe, ye = np.histogram2d(all_proj[:, 0], all_proj[:, 1],
                                bins=60, range=[[-lim, lim], [-lim, lim]])
    h_log = np.log1p(h)
    h_smooth = gaussian_filter(h_log, sigma=1.2)
    im = ax.imshow(h_smooth.T, origin="lower", extent=[-lim, lim, -lim, lim],
                   cmap="inferno", aspect="equal", interpolation="bilinear")
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(colors="white")
    ax.set_title(f"Heatmap ({len(all_proj)} frames, {len(per_pitch)} pitches)",
                 color="white", fontsize=10)
    ax.set_xlabel("PC1", color="white"); ax.set_ylabel("PC2", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")

    # Panel 2: HDBSCAN cluster labels
    ax = axes[1]
    ax.set_facecolor("#0a0a0a")
    cmap = plt.get_cmap("tab10")
    for c in range(n_clusters):
        mask = labels == c
        color = "#fbc02d" if c == stance_id else cmap(c % 10)
        marker = "*" if c == stance_id else "o"
        s = 18 if c == stance_id else 6
        ax.scatter(all_proj[mask, 0], all_proj[mask, 1], s=s, c=[color],
                   alpha=0.7, edgecolor="none", marker=marker)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(all_proj[noise_mask, 0], all_proj[noise_mask, 1], s=4, c="#555",
                   alpha=0.4, edgecolor="none")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#555", lw=0.5); ax.axvline(0, color="#555", lw=0.5)
    ax.set_title(f"HDBSCAN clusters ({n_clusters}; stance = ★ gold)",
                 color="white", fontsize=10)
    ax.set_xlabel("PC1", color="white"); ax.set_ylabel("PC2", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")

    # Panel 3: per-pitch trajectories with stance-cluster frames highlighted
    ax = axes[2]
    ax.set_facecolor("#0a0a0a")
    for pi, (play_id, ts, proj, wuo) in enumerate(per_pitch):
        m = pitch_ids == pi
        pitch_labels = labels[m]
        # Faint full trajectory
        ax.plot(proj[:, 0], proj[:, 1], "-", color="#444", lw=0.5, alpha=0.4)
        # Highlight stance-cluster frames
        stance_frames = pitch_labels == stance_id
        if stance_frames.any():
            ax.scatter(proj[stance_frames, 0], proj[stance_frames, 1],
                       s=3, c="#4caf50", alpha=0.6, edgecolor="none")
        # Setup frames (not in stance cluster, not noise)
        setup_frames = (pitch_labels != stance_id) & (pitch_labels != -1)
        if setup_frames.any():
            ax.scatter(proj[setup_frames, 0], proj[setup_frames, 1],
                       s=3, c="#f44336", alpha=0.5, edgecolor="none")
        # End-of-pre-osc marker
        ax.scatter([proj[-1, 0]], [proj[-1, 1]], s=18, c="#ff9800",
                   marker="*", edgecolor="white", lw=0.3, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#555", lw=0.5); ax.axvline(0, color="#555", lw=0.5)
    ax.set_title("Per-frame: green=stance  red=setup  gray=noise  ★=windup_onset",
                 color="white", fontsize=10)
    ax.set_xlabel("PC1", color="white"); ax.set_ylabel("PC2", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")

    fig.suptitle("Manzardo posture clustering — finding the stance behavioral mode",
                 color="white", fontsize=12)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=110, facecolor="#1a1a1a")
    print(f"\nWrote {out_png}")

    # Per-pitch stance-entry timing
    print("\nStance-entry timing per pitch (seconds before windup_onset):")
    entries = []
    for pi, (play_id, ts, proj, wuo) in enumerate(per_pitch):
        m = pitch_ids == pi
        pitch_labels = labels[m]
        # Stance "entry": first frame where label==stance_id AND it stays stance/noise (not setup) thereafter
        stance_idx = np.where(pitch_labels == stance_id)[0]
        if len(stance_idx) == 0:
            entry_t = None; secs_before = None
        else:
            # Most robust: the first frame after the last frame labeled non-stance,non-noise
            non_stance_idx = np.where((pitch_labels != stance_id) & (pitch_labels != -1))[0]
            if len(non_stance_idx) == 0:
                entry_idx = stance_idx[0]
            else:
                entry_idx = int(non_stance_idx.max() + 1)
                if entry_idx >= len(pitch_labels): entry_idx = len(pitch_labels) - 1
            entry_t = ts[entry_idx]
            secs_before = wuo - entry_t
            entries.append(secs_before)
        print(f"  pitch {pi+1}: entry={secs_before:.2f}s before windup" if secs_before is not None
              else f"  pitch {pi+1}: no clean stance entry")
    if entries:
        print(f"\nSummary: median stance window = {np.median(entries):.2f}s, "
              f"min={min(entries):.2f}s, max={max(entries):.2f}s")


if __name__ == "__main__":
    main()
