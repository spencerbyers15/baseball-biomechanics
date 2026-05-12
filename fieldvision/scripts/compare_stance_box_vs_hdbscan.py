"""Compare two candidate stance-frame filters on Manzardo's pooled pre-osc data:

  1. SPATIAL BOX:  in-stance iff |PC1| ≤ BOX and |PC2| ≤ BOX
  2. HDBSCAN:      in-stance iff HDBSCAN label ≠ -1 (noise)

Renders a 2×2 grid:
  - heatmap with the box outlined
  - HDBSCAN labels with the box outlined
  - frame-level confusion table (box × hdbscan)
  - bar chart of fraction inside box per HDBSCAN cluster id

Prints agreement statistics. Useful to confirm Spencer's "PC1, PC2 ≤ 1"
intuition matches the data-driven HDBSCAN noise rejection.
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
BOX = 1.0  # |PC1|, |PC2| <= BOX defines the candidate "stance box"
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
        body_centroid = joints.mean(axis=0)
        joints_norm = (joints - body_centroid).flatten()
        head_norm = head - body_centroid
        handle_norm = handle - body_centroid
        rows.append(np.concatenate([joints_norm, head_norm, handle_norm]))
    if len(rows) < 5: return None
    return np.array(rows)


def main():
    out_png = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_box_vs_hdbscan.png"

    census = pickle.load(open("data/oscillation_report/pre_pitch_preparatory_movement/census.pkl", "rb"))
    cres = census["results"][f"{MANZARDO}_L"]
    pc12 = cres["pca_components"][:2]
    per_pitch_mean_pose = cres.get("per_pitch_mean_pose", {})
    keep_play_ids = set(per_pitch_mean_pose.keys())

    all_proj = []
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
            proj = (vecs - mp) @ pc12.T
            all_proj.append(proj)
        conn.close()
    all_proj = np.vstack(all_proj)
    n = len(all_proj)
    print(f"Pooled {n} frames")

    # HDBSCAN — use a smaller min_cluster_size here so it can find the dense
    # core on the saved (pass-2, stance-tight) PCs. The census itself runs
    # HDBSCAN on pass-1 PCs where structure is coarser, so the same params
    # produce different counts.
    min_cs = 40
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=10).fit_predict(all_proj)
    n_clusters = labels.max() + 1
    noise_mask = labels == -1
    print(f"HDBSCAN: {n_clusters} clusters, noise={noise_mask.sum()} ({noise_mask.mean()*100:.1f}%)")

    # Spatial box
    in_box = (np.abs(all_proj[:, 0]) <= BOX) & (np.abs(all_proj[:, 1]) <= BOX)
    print(f"Box |PC|≤{BOX}: in={in_box.sum()} ({in_box.mean()*100:.1f}%)")

    # Confusion
    cluster_in = (~noise_mask) & in_box
    cluster_out = (~noise_mask) & (~in_box)
    noise_in = noise_mask & in_box
    noise_out = noise_mask & (~in_box)
    print()
    print(f"               in_box        out_box")
    print(f"  CLUSTER     {cluster_in.sum():6d}        {cluster_out.sum():6d}")
    print(f"  NOISE       {noise_in.sum():6d}        {noise_out.sum():6d}")
    print()
    cluster_frac_in = cluster_in.sum() / max(1, (~noise_mask).sum()) * 100
    noise_frac_out = noise_out.sum() / max(1, noise_mask.sum()) * 100
    box_frac_cluster = cluster_in.sum() / max(1, in_box.sum()) * 100
    print(f"Of HDBSCAN clusters: {cluster_frac_in:.1f}% inside box")
    print(f"Of HDBSCAN noise:    {noise_frac_out:.1f}% outside box")
    print(f"Of box:              {box_frac_cluster:.1f}% are in clusters")

    # Per-cluster fraction-in-box
    print("\nPer-cluster fraction inside box:")
    for c in range(n_clusters):
        m = labels == c
        if m.sum() == 0: continue
        f = (m & in_box).sum() / m.sum() * 100
        print(f"  cluster {c} (n={m.sum()}): {f:.0f}% inside box")

    # === Render ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#1a1a1a")
    lim = float(np.percentile(np.abs(all_proj), 99)) * 1.1

    def draw_box(ax):
        ax.plot([-BOX, BOX, BOX, -BOX, -BOX], [-BOX, -BOX, BOX, BOX, -BOX],
                "-", color="#fbc02d", lw=2.0, label=f"|PC|≤{BOX} box")

    # Panel 1: heatmap with box
    ax = axes[0, 0]
    ax.set_facecolor("#0a0a0a")
    h, xe, ye = np.histogram2d(all_proj[:, 0], all_proj[:, 1], bins=60,
                                range=[[-lim, lim], [-lim, lim]])
    h_s = gaussian_filter(np.log1p(h), sigma=1.2)
    im = ax.imshow(h_s.T, origin="lower", extent=[-lim, lim, -lim, lim],
                   cmap="inferno", aspect="equal", interpolation="bilinear")
    draw_box(ax)
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(colors="white")
    ax.set_title(f"Heatmap ({n} frames, centroid-normalized)", color="white", fontsize=11)
    ax.set_xlabel("PC1", color="white"); ax.set_ylabel("PC2", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.legend(loc="lower right", facecolor="#1a1a1a", edgecolor="#666",
              labelcolor="white", fontsize=9)

    # Panel 2: HDBSCAN labels with box
    ax = axes[0, 1]
    ax.set_facecolor("#0a0a0a")
    cmap = plt.get_cmap("tab10")
    for c in range(n_clusters):
        m = labels == c
        ax.scatter(all_proj[m, 0], all_proj[m, 1], s=4, c=[cmap(c % 10)],
                   alpha=0.6, edgecolor="none", label=f"cluster {c} (n={m.sum()})")
    ax.scatter(all_proj[noise_mask, 0], all_proj[noise_mask, 1], s=4, c="#666",
               alpha=0.4, edgecolor="none", label=f"noise (n={noise_mask.sum()})")
    draw_box(ax)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#444", lw=0.4); ax.axvline(0, color="#444", lw=0.4)
    ax.set_title(f"HDBSCAN labels ({n_clusters} clusters)", color="white", fontsize=11)
    ax.set_xlabel("PC1", color="white"); ax.set_ylabel("PC2", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.legend(loc="lower right", facecolor="#1a1a1a", edgecolor="#666",
              labelcolor="white", fontsize=8)

    # Panel 3: confusion as text
    ax = axes[1, 0]
    ax.set_facecolor("#0a0a0a")
    ax.axis("off")
    table_text = (
        f"AGREEMENT TABLE\n\n"
        f"                   in box  ({in_box.mean()*100:5.1f}%)     out box ({(~in_box).mean()*100:5.1f}%)\n"
        f"  HDBSCAN cluster   {cluster_in.sum():6d}              {cluster_out.sum():6d}\n"
        f"  HDBSCAN noise     {noise_in.sum():6d}              {noise_out.sum():6d}\n\n"
        f"  Box thinks stance: {in_box.sum()} ({in_box.mean()*100:.1f}%)\n"
        f"  HDBSCAN thinks stance: {(~noise_mask).sum()} ({(~noise_mask).mean()*100:.1f}%)\n\n"
        f"  HDBSCAN clusters inside box: {cluster_frac_in:.1f}%\n"
        f"  HDBSCAN noise outside box:   {noise_frac_out:.1f}%\n"
        f"  Box contents that are in HDBSCAN clusters: {box_frac_cluster:.1f}%"
    )
    ax.text(0.05, 0.95, table_text, color="white", fontsize=11,
            family="monospace", va="top", transform=ax.transAxes)
    ax.set_title("Box vs HDBSCAN — frame-level agreement",
                 color="white", fontsize=11, loc="left")

    # Panel 4: per-cluster fraction inside box
    ax = axes[1, 1]
    ax.set_facecolor("#0a0a0a")
    cluster_labels_x = []
    cluster_fracs = []
    cluster_ns = []
    for c in range(n_clusters):
        m = labels == c
        if m.sum() == 0: continue
        cluster_labels_x.append(f"cluster {c}\n(n={m.sum()})")
        cluster_fracs.append((m & in_box).sum() / m.sum() * 100)
        cluster_ns.append(m.sum())
    cluster_labels_x.append(f"noise\n(n={noise_mask.sum()})")
    cluster_fracs.append((noise_mask & in_box).sum() / max(1, noise_mask.sum()) * 100)
    cluster_ns.append(noise_mask.sum())
    colors = [cmap(c % 10) for c in range(n_clusters)] + ["#666"]
    bars = ax.bar(range(len(cluster_labels_x)), cluster_fracs,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(50, color="#666", lw=0.5, ls="--")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(cluster_labels_x)))
    ax.set_xticklabels(cluster_labels_x, color="white", fontsize=9)
    ax.set_ylabel("% of cluster's frames inside |PC|≤1 box", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    for bar, frac in zip(bars, cluster_fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{frac:.0f}%", ha="center", va="bottom", color="white", fontsize=9)
    ax.set_title("Per-cluster: fraction of frames inside box",
                 color="white", fontsize=11)

    fig.suptitle(
        f"Manzardo: does HDBSCAN-noise = pre-stance setup match the spatial |PC|≤{BOX} rule?",
        color="white", fontsize=12, y=0.995
    )
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=110, facecolor="#1a1a1a", bbox_inches="tight")
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
