"""Render the two HDBSCAN-discovered behavioral clusters as side-by-side
skeleton + bat images so we can SEE what posture each cluster represents.

For each cluster, the rendered figure is the mean 66-dim posture vector
across all frames assigned to that cluster (joints + bat in real-world
coordinates, averaged). A few sampled poses are overlaid faintly to show
the spread.

Output: data/oscillation_report/pre_pitch_preparatory_movement/manzardo_cluster_postures.png
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games, open_game
from fieldvision.skeleton import SKELETON_CONNECTIONS
from fieldvision.storage import JOINT_COLS
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

PRE_OSC_SECONDS = 5.0
MANZARDO = 700932
JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
JOINT_NAMES = {bid: name for bid, name in JOINT_COLS}
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
        joint_vec = np.full(N_JOINTS * 3, np.nan)
        for j, bid in enumerate(JOINT_BIDS):
            if bid in wp: joint_vec[j*3:j*3+3] = wp[bid]
        if np.any(np.isnan(joint_vec)): continue
        if len(bat_times) == 0: continue
        bi = int(np.argmin(np.abs(bat_times - t)))
        if abs(bat_times[bi] - t) > 0.07: continue
        head, handle = bat_records[bi][1], bat_records[bi][2]
        rows.append(np.concatenate([joint_vec, head, handle]))
    if len(rows) < 5: return None
    return np.array(rows)  # (n, 66) RAW (un-centered) postures


def draw_pose(ax, vec, color="#bbbbbb", alpha=1.0, lw=1.6, bat_color="#d4a04c"):
    joints = vec[:N_JOINTS * 3].reshape(N_JOINTS, 3)  # each row (x, y, z)
    bat_head = vec[N_JOINTS * 3: N_JOINTS * 3 + 3]
    bat_handle = vec[N_JOINTS * 3 + 3:]
    joint_lookup = {bid: joints[i] for i, bid in enumerate(JOINT_BIDS)}
    for a, b in SKELETON_CONNECTIONS:
        if a in joint_lookup and b in joint_lookup:
            p1, p2 = joint_lookup[a], joint_lookup[b]
            ax.plot([p1[2], p2[2]], [p1[1], p2[1]], "-",
                    color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.scatter(joints[:, 2], joints[:, 1], s=18, c=color, alpha=alpha, edgecolor="none")
    ax.plot([bat_handle[2], bat_head[2]], [bat_handle[1], bat_head[1]],
            "-", color=bat_color, lw=lw + 1.4, alpha=alpha, solid_capstyle="round")
    ax.scatter([bat_head[2]], [bat_head[1]], s=50, c=bat_color, alpha=alpha,
               edgecolor="white", lw=0.4)


def main():
    out_png = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_cluster_postures.png"

    census = pickle.load(open("data/oscillation_report/pre_pitch_preparatory_movement/census.pkl", "rb"))
    cres = census["results"][f"{MANZARDO}_L"]
    pc12 = cres["pca_components"][:2]
    per_pitch_mean_pose = cres.get("per_pitch_mean_pose", {})
    keep_play_ids = set(per_pitch_mean_pose.keys())

    all_vecs, all_proj = [], []
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
            all_vecs.append(vecs)
            all_proj.append(proj)
        conn.close()

    all_vecs = np.vstack(all_vecs)
    all_proj = np.vstack(all_proj)
    print(f"Pooled {len(all_vecs)} frames")

    # Re-run identical clustering as the diagnostic
    min_cs = max(40, len(all_proj) // 20)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=10).fit_predict(all_proj)
    print(f"  clusters: {labels.max() + 1}, noise: {(labels == -1).sum()}")

    # Identify clusters
    n_clusters = labels.max() + 1
    cluster_info = []
    for c in range(n_clusters):
        mask = labels == c
        center_pc = all_proj[mask].mean(axis=0)
        cluster_info.append((c, mask.sum(), float(np.linalg.norm(center_pc)), center_pc))
    cluster_info.sort(key=lambda x: x[2])  # closest-to-origin first
    stance_id = cluster_info[0][0]
    other_ids = [c for (c, _, _, _) in cluster_info[1:]]
    print(f"Stance cluster: {stance_id} (n={cluster_info[0][1]})")
    for c, n, d, _ in cluster_info[1:]:
        print(f"  other cluster {c}: n={n}, |center_pc|={d:.2f}")

    # Compute mean posture per cluster (in real-world coordinates)
    def cluster_mean(c):
        m = labels == c
        return all_vecs[m].mean(axis=0), m.sum()

    stance_mean, stance_n = cluster_mean(stance_id)
    other_means = [(c, *cluster_mean(c)) for c in other_ids]

    # === Figure ===
    n_panels = 1 + len(other_means)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 7.5), facecolor="#1a1a1a")
    if n_panels == 1: axes = [axes]

    def setup_ax(ax, title, color):
        ax.set_facecolor("#1a3a1a")
        ax.set_aspect("equal")
        ax.tick_params(colors="#aaa")
        for s in ax.spines.values(): s.set_color(color); s.set_linewidth(2)
        ax.set_xlabel("Z (ft)", color="#aaa")
        ax.set_ylabel("Y / height (ft)", color="#aaa")
        ax.set_title(title, color=color, fontsize=11)

    # Find common bounds (Z and Y) so the panels are scaled identically
    def pose_bounds(vec):
        joints = vec[:N_JOINTS * 3].reshape(N_JOINTS, 3)
        head = vec[N_JOINTS * 3: N_JOINTS * 3 + 3]
        handle = vec[N_JOINTS * 3 + 3:]
        zs = np.concatenate([joints[:, 2], [head[2], handle[2]]])
        ys = np.concatenate([joints[:, 1], [head[1], handle[1]]])
        return zs, ys
    all_zs, all_ys = [], []
    for v in [stance_mean] + [om for _, om, _ in other_means]:
        z, y = pose_bounds(v); all_zs.append(z); all_ys.append(y)
    zmin = min(z.min() for z in all_zs) - 1
    zmax = max(z.max() for z in all_zs) + 1
    ymin = 0
    ymax = max(y.max() for y in all_ys) + 1

    # Stance panel: 30 random sampled poses + mean overlay
    ax = axes[0]
    setup_ax(ax, f"STANCE  (cluster {stance_id}, n={stance_n} frames, "
                  f"{stance_n / len(labels) * 100:.0f}% of pre-osc)",
             "#4caf50")
    stance_idxs = np.where(labels == stance_id)[0]
    sample_idxs = np.random.default_rng(7).choice(stance_idxs, size=min(30, len(stance_idxs)),
                                                   replace=False)
    for i in sample_idxs:
        draw_pose(ax, all_vecs[i], color="#9bd49b", alpha=0.18, lw=0.8, bat_color="#d4a04c")
    draw_pose(ax, stance_mean, color="white", alpha=1.0, lw=2.2, bat_color="#ffd54f")
    ax.set_xlim(zmin, zmax); ax.set_ylim(ymin, ymax)

    # Other-cluster panels
    for pi_panel, (c, om, on) in enumerate(other_means, start=1):
        ax = axes[pi_panel]
        setup_ax(ax,
                 f"NON-STANCE cluster {c}  (n={on} frames, {on / len(labels) * 100:.0f}%)\n"
                 f"PC-center = ({cluster_info[pi_panel][3][0]:.2f}, {cluster_info[pi_panel][3][1]:.2f})",
                 "#f44336")
        other_idxs = np.where(labels == c)[0]
        sample_idxs = np.random.default_rng(7).choice(other_idxs, size=min(30, len(other_idxs)),
                                                       replace=False)
        for i in sample_idxs:
            draw_pose(ax, all_vecs[i], color="#e89b9b", alpha=0.18, lw=0.8, bat_color="#d4a04c")
        draw_pose(ax, om, color="white", alpha=1.0, lw=2.2, bat_color="#ffd54f")
        ax.set_xlim(zmin, zmax); ax.set_ylim(ymin, ymax)

    fig.suptitle("Manzardo posture clusters — mean pose (bright) + sampled poses (faint)\n"
                 "White skeleton = cluster mean, gold = bat",
                 color="white", fontsize=12, y=0.99)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=110, facecolor="#1a1a1a", bbox_inches="tight")
    print(f"\nWrote {out_png}")

    # Numerical comparison
    print("\n=== Differences (NON-stance mean minus STANCE mean) ===")
    print("Joint / Bat anchor          ΔX(ft)  ΔY(ft)  ΔZ(ft)  ||Δ||")
    s_j = stance_mean[:N_JOINTS * 3].reshape(N_JOINTS, 3)
    s_h = stance_mean[N_JOINTS * 3: N_JOINTS * 3 + 3]
    s_hd = stance_mean[N_JOINTS * 3 + 3:]
    for c, om, on in other_means:
        o_j = om[:N_JOINTS * 3].reshape(N_JOINTS, 3)
        o_h = om[N_JOINTS * 3: N_JOINTS * 3 + 3]
        o_hd = om[N_JOINTS * 3 + 3:]
        print(f"  --- cluster {c} ---")
        deltas = []
        for j, bid in enumerate(JOINT_BIDS):
            d = o_j[j] - s_j[j]
            deltas.append((JOINT_NAMES[bid], d, np.linalg.norm(d)))
        deltas.append(("bat_head", o_h - s_h, np.linalg.norm(o_h - s_h)))
        deltas.append(("bat_handle", o_hd - s_hd, np.linalg.norm(o_hd - s_hd)))
        deltas.sort(key=lambda x: -x[2])
        for name, d, m in deltas[:8]:
            print(f"    {name:14s}  {d[0]:+.2f}   {d[1]:+.2f}   {d[2]:+.2f}   {m:.2f}")

        # Bat angle (handle→head direction) per cluster
        def bat_pitch_yaw(h, head):
            v = np.array(head) - np.array(h)
            n = np.linalg.norm(v) + 1e-9
            pitch = np.degrees(np.arcsin(v[1] / n))  # vertical angle
            yaw = np.degrees(np.arctan2(v[2], v[0]))  # horizontal angle in xz
            return pitch, yaw
        sp, sy = bat_pitch_yaw(s_hd, s_h)
        op, oy = bat_pitch_yaw(o_hd, o_h)
        print(f"    bat orientation: stance pitch={sp:.0f}°/yaw={sy:.0f}°, "
              f"cluster pitch={op:.0f}°/yaw={oy:.0f}°  "
              f"(Δpitch={op - sp:+.0f}°, Δyaw={oy - sy:+.0f}°)")


if __name__ == "__main__":
    main()
