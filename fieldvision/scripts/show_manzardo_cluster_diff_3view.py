"""Render both HDBSCAN cluster means OVERLAID in three orthogonal views
(side / front / top) plus a per-joint Δ chart. Makes visible the
direction along which the two 'stance' clusters actually differ, which
is hidden in a single side-view projection.
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
    return np.array(rows)


def draw_pose_view(ax, vec, axis_a, axis_b, color, alpha=1.0, lw=1.8, bat_color="#d4a04c"):
    """axis_a / axis_b are 0,1,2 → x,y,z. Plot joint coords (axis_a vs axis_b)."""
    joints = vec[:N_JOINTS * 3].reshape(N_JOINTS, 3)
    bat_head = vec[N_JOINTS * 3: N_JOINTS * 3 + 3]
    bat_handle = vec[N_JOINTS * 3 + 3:]
    joint_lookup = {bid: joints[i] for i, bid in enumerate(JOINT_BIDS)}
    for a, b in SKELETON_CONNECTIONS:
        if a in joint_lookup and b in joint_lookup:
            p1, p2 = joint_lookup[a], joint_lookup[b]
            ax.plot([p1[axis_a], p2[axis_a]], [p1[axis_b], p2[axis_b]], "-",
                    color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.scatter(joints[:, axis_a], joints[:, axis_b], s=22, c=color, alpha=alpha,
               edgecolor="none")
    ax.plot([bat_handle[axis_a], bat_head[axis_a]],
            [bat_handle[axis_b], bat_head[axis_b]], "-",
            color=bat_color, lw=lw + 1.2, alpha=alpha, solid_capstyle="round")
    ax.scatter([bat_head[axis_a]], [bat_head[axis_b]], s=55, c=bat_color,
               alpha=alpha, edgecolor="white", lw=0.5)


def main():
    out_png = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_cluster_diff_3view.png"

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

    min_cs = max(40, len(all_proj) // 20)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=10).fit_predict(all_proj)
    n_clusters = labels.max() + 1
    print(f"  clusters: {n_clusters}, noise: {(labels == -1).sum()}")

    cluster_info = []
    for c in range(n_clusters):
        m = labels == c
        center_pc = all_proj[m].mean(axis=0)
        cluster_info.append((c, m.sum(), float(np.linalg.norm(center_pc)), center_pc))
    cluster_info.sort(key=lambda x: x[2])

    c0_id = cluster_info[0][0]
    c1_id = cluster_info[1][0]
    c0_mean = all_vecs[labels == c0_id].mean(axis=0)
    c1_mean = all_vecs[labels == c1_id].mean(axis=0)

    fig = plt.figure(figsize=(16, 11), facecolor="#1a1a1a")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1], hspace=0.30, wspace=0.20,
                          left=0.05, right=0.98, top=0.93, bottom=0.06)

    views = [
        ("Side view (Z vs Y)  —  X (depth) is hidden here", 2, 1, "Z (ft)", "Y (ft)"),
        ("Front view (X vs Y)  —  the depth dimension", 0, 1, "X (ft)", "Y (ft)"),
        ("Top view (X vs Z)  —  looking down from above", 0, 2, "X (ft)", "Z (ft)"),
    ]
    for i, (title, ax_a, ax_b, xlab, ylab) in enumerate(views):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#0a0a0a"); ax.set_aspect("equal")
        draw_pose_view(ax, c0_mean, ax_a, ax_b, color="#4caf50", alpha=0.95, lw=2.0,
                       bat_color="#a5d6a7")
        draw_pose_view(ax, c1_mean, ax_a, ax_b, color="#ef5350", alpha=0.95, lw=2.0,
                       bat_color="#ef9a9a")
        ax.set_xlabel(xlab, color="#ccc")
        ax.set_ylabel(ylab, color="#ccc")
        ax.tick_params(colors="#888")
        for s in ax.spines.values(): s.set_color("#555")
        ax.set_title(title, color="white", fontsize=10)
        if i == 2:
            ax.plot([], [], "-", color="#4caf50", lw=2,
                    label=f"cluster {c0_id} (closer to origin)")
            ax.plot([], [], "-", color="#ef5350", lw=2,
                    label=f"cluster {c1_id} (offset)")
            ax.legend(loc="lower right", facecolor="#1a1a1a", edgecolor="#666",
                      labelcolor="white", fontsize=8)

    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor("#0a0a0a")
    s_j = c0_mean[:N_JOINTS * 3].reshape(N_JOINTS, 3)
    o_j = c1_mean[:N_JOINTS * 3].reshape(N_JOINTS, 3)
    s_bat = c0_mean[N_JOINTS * 3:].reshape(2, 3)
    o_bat = c1_mean[N_JOINTS * 3:].reshape(2, 3)
    labels_x = [JOINT_NAMES[bid] for bid in JOINT_BIDS] + ["bat_head", "bat_handle"]
    dx = np.concatenate([o_j[:, 0] - s_j[:, 0], o_bat[:, 0] - s_bat[:, 0]])
    dy = np.concatenate([o_j[:, 1] - s_j[:, 1], o_bat[:, 1] - s_bat[:, 1]])
    dz = np.concatenate([o_j[:, 2] - s_j[:, 2], o_bat[:, 2] - s_bat[:, 2]])
    width = 0.27
    pos = np.arange(len(labels_x))
    ax.bar(pos - width, dx, width, color="#ef5350", label="ΔX (catcher direction)")
    ax.bar(pos,         dy, width, color="#ffca28", label="ΔY (height)")
    ax.bar(pos + width, dz, width, color="#42a5f5", label="ΔZ (pitcher direction)")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels_x, rotation=70, color="white", fontsize=7)
    ax.tick_params(colors="white")
    ax.axhline(0, color="#555", lw=0.5)
    for s in ax.spines.values(): s.set_color("#555")
    ax.legend(facecolor="#1a1a1a", edgecolor="#666", labelcolor="white", fontsize=9)
    ax.set_ylabel("cluster1 − cluster0 (ft)", color="#ccc")
    ax.set_title("Per-joint difference between the two clusters — ΔX dominates",
                 color="white", fontsize=11)

    fig.suptitle(
        f"Manzardo two-cluster delta — green & red are the cluster means overlaid\n"
        f"cluster {c0_id} n={cluster_info[0][1]},  cluster {c1_id} n={cluster_info[1][1]},  "
        f"noise n={(labels == -1).sum()}",
        color="white", fontsize=12, y=0.985
    )
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=110, facecolor="#1a1a1a", bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
