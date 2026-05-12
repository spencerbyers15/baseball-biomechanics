"""Test two higher-dim clustering schemes on Manzardo's pre-osc data to see
whether either resolves the stance / non-stance distinction Spencer sees
by eye but PC1×PC2 misses.

Variants tested on the same 66-dim centroid-normalized + hands-on-bat-gated data:
  - HDBSCAN in PC1-PC5 (5-dim)
  - HDBSCAN in PC1-PC10 (10-dim)
  - UMAP(66 → 2) + HDBSCAN (B-SOiD style)

Renders a comparison panel and prints per-method cluster + noise counts.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import hdbscan
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
MANZARDO = 700932
HAND_TO_HANDLE_MAX_FT = 1.0
JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
N_JOINTS = len(JOINT_BIDS)


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
        joints = np.full((N_JOINTS, 3), np.nan)
        for j, bid in enumerate(JOINT_BIDS):
            if bid in wp: joints[j] = wp[bid]
        if np.any(np.isnan(joints)): continue
        if len(bat_times) == 0: continue
        bi = int(np.argmin(np.abs(bat_times - t)))
        if abs(bat_times[bi] - t) > 0.07: continue
        head = np.array(bat_records[bi][1]); handle = np.array(bat_records[bi][2])
        # hands-on-bat
        hand_rt = joints[JOINT_BIDS.index(28)]
        hand_lt = joints[JOINT_BIDS.index(67)]
        if (np.linalg.norm(hand_rt - handle) > HAND_TO_HANDLE_MAX_FT or
            np.linalg.norm(hand_lt - handle) > HAND_TO_HANDLE_MAX_FT):
            continue
        body_centroid = joints.mean(axis=0)
        joints_norm = (joints - body_centroid).flatten()
        head_norm = head - body_centroid
        handle_norm = handle - body_centroid
        rows.append(np.concatenate([joints_norm, head_norm, handle_norm]))
    if len(rows) < 5: return None
    return np.array(rows)


def main():
    out_png = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_clustering_methods.png"

    census = pickle.load(open("data/oscillation_report/pre_pitch_preparatory_movement/census.pkl", "rb"))
    cres = census["results"][f"{MANZARDO}_L"]
    pca_comps = cres["pca_components"]  # (5, 66)
    per_pitch_mean_pose = cres.get("per_pitch_mean_pose", {})
    keep_play_ids = set(per_pitch_mean_pose.keys())

    all_vecs, all_proj2 = [], []
    for game_pk in list_games(Path(os.environ.get("FV_DATA_DIR", "data"))):
        conn = open_game(game_pk, Path(os.environ.get("FV_DATA_DIR", "data")))
        try:
            rows = conn.execute(
                "SELECT play_id, pitcher_id, start_time_unix "
                "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL",
                (MANZARDO,)).fetchall()
        except Exception:
            conn.close(); continue
        for play_id, pitcher_id, t_rel in rows:
            if play_id not in keep_play_ids: continue
            vecs = load_pitch_posture(conn, MANZARDO, pitcher_id, t_rel)
            if vecs is None: continue
            mp = per_pitch_mean_pose[play_id]
            all_vecs.append(vecs - mp)  # centered
        conn.close()
    all_vecs = np.vstack(all_vecs)
    print(f"Loaded {len(all_vecs)} stance-candidate frames")

    # Project onto 5 PCs and 10 PCs (we only have 5 stored — refit for 10)
    proj5 = all_vecs @ pca_comps.T  # (n, 5)
    # For 10 PCs, refit
    U, S, Vt = np.linalg.svd(all_vecs, full_matrices=False)
    proj10 = all_vecs @ Vt[:10].T
    proj_pc12 = proj5[:, :2]

    var10 = (S**2) / np.sum(S**2)
    print(f"PCs variance: {[f'{v*100:.0f}%' for v in var10[:10]]}")

    # 2D PC clustering (baseline)
    lab2 = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(proj_pc12)
    # 5D PC clustering
    lab5 = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(proj5)
    # 10D PC clustering
    lab10 = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(proj10)

    # UMAP from 66D -> 2D then HDBSCAN
    print("Running UMAP...")
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=42)
    umap_proj = reducer.fit_transform(all_vecs)
    lab_umap = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(umap_proj)

    methods = [
        ("PC1-PC2 (baseline 2D)", lab2, proj_pc12),
        ("PC1-PC5 (5D)", lab5, proj_pc12),  # show in PC1xPC2 for comparison
        ("PC1-PC10 (10D)", lab10, proj_pc12),
        ("UMAP(66D→2D)", lab_umap, umap_proj),
    ]

    print("\nCluster summary:")
    print(f"{'method':<22}  n_clusters  noise %")
    for name, labels, _ in methods:
        n_clusters = labels.max() + 1
        noise_pct = (labels == -1).mean() * 100
        # largest cluster
        if (labels >= 0).any():
            sizes = np.bincount(labels[labels >= 0])
            largest_pct = sizes.max() / len(labels) * 100
        else:
            largest_pct = 0
        print(f"  {name:<22}  {n_clusters:>10d}  {noise_pct:>5.1f}%  "
              f"largest-cluster={largest_pct:.1f}%")

    # Render
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), facecolor="#1a1a1a")
    cmap = plt.get_cmap("tab10")
    for ax, (name, labels, coords) in zip(axes.ravel(), methods):
        ax.set_facecolor("#0a0a0a")
        n_clusters = labels.max() + 1
        for c in range(n_clusters):
            m = labels == c
            ax.scatter(coords[m, 0], coords[m, 1], s=5, c=[cmap(c % 10)],
                       alpha=0.6, edgecolor="none", label=f"cluster {c} (n={m.sum()})")
        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1], s=4,
                       c="#555", alpha=0.4, edgecolor="none",
                       label=f"noise (n={noise_mask.sum()})")
        ax.set_aspect("equal")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#666")
        ax.set_title(f"{name}  —  {n_clusters} clusters, {noise_mask.mean()*100:.0f}% noise",
                     color="white", fontsize=11)
        if "PC" in name and coords is proj_pc12:
            ax.set_xlabel("PC1", color="#ccc"); ax.set_ylabel("PC2", color="#ccc")
        else:
            ax.set_xlabel("dim 1", color="#ccc"); ax.set_ylabel("dim 2", color="#ccc")
        ax.legend(facecolor="#1a1a1a", edgecolor="#666", labelcolor="white", fontsize=7,
                  loc="lower right")

    fig.suptitle("Manzardo: clustering at different feature-space dimensionalities\n"
                 "(scatter colored by cluster, gray = HDBSCAN noise)",
                 color="white", fontsize=12)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=110, facecolor="#1a1a1a", bbox_inches="tight")
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
