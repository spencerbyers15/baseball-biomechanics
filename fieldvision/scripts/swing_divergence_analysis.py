"""Swing divergence & 'fooled-ness' analysis per batter.

For each batter:
  1. Detect each swing's swing_trigger from bat speed
  2. Time-align all swings to t=swing_trigger
  3. Compute per-time-offset variance in joint space across all swings →
     divergence curve. Find divergence_onset = first time variance > 2× baseline.
  4. Cluster swings (k-means on time-flattened PC-projected windows) to find
     "swing modes" — full swing vs check-swing vs late adjust.
  5. Build canonical good-swing trajectory = mean across contact swings (result_call='X').
  6. For each swing: compute deviation-from-canonical (sum of squared joint
     differences over the swing window) + min ball-bat-axis distance.
  7. Combined "fooled" score = z-scored sum of normalized deviation + distance.

Outputs:
  data/oscillation_report/pre_pitch_preparatory_movement/swing_divergence/<batter_id>_<name>/...
    - divergence_curve.png   (variance vs time)
    - swing_clusters.png     (cluster assignments)
    - canonical_vs_pitches.png (each swing's posture deviation curve from canonical)
    - fooled_score.csv       (per-pitch fooled score + components)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games, open_game
from fieldvision.batter_kinematics import (ball_bat_min_distance,
                                            detect_batter_events,
                                            interpolate_trajectory)
from fieldvision.skeleton import SKELETON_CONNECTIONS
from fieldvision.storage import JOINT_COLS
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
JOINT_NAMES = {bid: name for bid, name in JOINT_COLS}
N_JOINTS = len(JOINT_BIDS)

SAMPLE_HZ = 100.0  # resample swings to 100 Hz for divergence analysis
PRE_TRIGGER_MS = 100   # 0.1s before trigger (where divergence should be 0)
POST_TRIGGER_MS = 300  # 0.3s after trigger covers the full swing
SWING_GRID = np.arange(-PRE_TRIGGER_MS/1000, POST_TRIGGER_MS/1000 + 1e-9, 1/SAMPLE_HZ)


def load_batter_swings(game_pks, data_dir, batter_id):
    """Return list of swing dicts for this batter across all games.

    Each dict has:
      play_id, result_call, pitch_type, pitch_speed, pitcher_id, game_pk,
      swing_trigger_t, peak_bat_speed_t, peak_bat_speed_value,
      joint_trajectory  (n_grid, n_joints*3) aligned to swing_trigger,
      bat_axis_trajectory (n_grid, 6)  -- handle xyz + head xyz
      ball_trajectory  (n_grid, 3)  -- nan-padded if absent
      ball_bat_min_d  (float)
      ball_bat_min_t  (float, rel-to-swing-trigger)
    """
    out = []
    cols = "time_unix, " + ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)
    for game_pk in game_pks:
        try:
            conn = open_game(game_pk, data_dir)
        except Exception:
            continue
        rows = conn.execute(
            "SELECT play_id, result_call, pitch_type, start_speed, pitcher_id, "
            "start_time_unix FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL "
            "ORDER BY start_time_unix",
            (batter_id,),
        ).fetchall()
        if not rows:
            conn.close(); continue
        joint_cols_select_local = ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)
        for play_id, call, ptype, speed, pitcher_id, t_rel in rows:
            # Pull batter frames near release (filtered to standing-batter actor only)
            b_rows_raw = load_clean_batter_actor_frames(
                conn, batter_id, t_rel - 1.5, t_rel + 0.9, joint_cols_select_local)
            b_rows = [(r[0],) + r[2:] for r in b_rows_raw]
            if len(b_rows) < 50: continue
            b_frames = []
            for r in b_rows:
                wp = {}
                for i, jbid in enumerate(JOINT_BIDS):
                    x, y, z = r[1+i*3], r[2+i*3], r[3+i*3]
                    if x is not None: wp[jbid] = (x, y, z)
                b_frames.append((r[0], wp))
            bat_rows = conn.execute(
                "SELECT time_unix, head_x, head_y, head_z, handle_x, handle_y, handle_z "
                "FROM bat_frame WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
                (t_rel - 1.5, t_rel + 0.9),
            ).fetchall()
            bat_frames_raw = [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bat_rows]
            bat_frames = filter_bat_frames(bat_frames_raw)
            # Per-pitch quality: skip pitches where bat isn't in batter's hands
            # or body data looks wrong
            quality = assess_pitch_quality(b_frames, bat_frames)
            if not quality["is_clean"]:
                continue
            ev = detect_batter_events(b_frames, bat_frames, t_rel)
            if not ev.has_swing:
                continue
            ball_rows = conn.execute(
                "SELECT time_unix, ball_x, ball_y, ball_z FROM ball_frame "
                "WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
                (t_rel - 0.05, ev.peak_bat_speed_t + 0.2),
            ).fetchall()
            ball_pts = [(r[0], (r[1], r[2], r[3])) for r in ball_rows if r[1] is not None]
            d, td, _, _ = ball_bat_min_distance(ball_pts, bat_frames, t_rel,
                                                 ev.peak_bat_speed_t + 0.2)

            # Resample batter joint trajectory to swing-trigger-aligned grid
            tg = SWING_GRID + ev.swing_trigger_t
            b_times = np.array([f[0] for f in b_frames])
            n_dim = N_JOINTS * 3
            joint_arr = np.full((len(b_frames), n_dim), np.nan)
            for i, (_, wp) in enumerate(b_frames):
                for j, jbid in enumerate(JOINT_BIDS):
                    if jbid in wp:
                        joint_arr[i, j*3:j*3+3] = wp[jbid]
            joint_resamp = interpolate_trajectory(b_times, joint_arr, tg)

            # Bat axis resampled
            bat_arr = np.full((len(bat_frames), 6), np.nan)
            bat_times = np.array([f[0] for f in bat_frames])
            for i, (_, head, handle) in enumerate(bat_frames):
                bat_arr[i, 0:3] = head
                bat_arr[i, 3:6] = handle
            bat_resamp = interpolate_trajectory(bat_times, bat_arr, tg)

            # Ball trajectory resampled (3 dims)
            if ball_pts:
                ball_t_arr = np.array([p[0] for p in ball_pts])
                ball_p_arr = np.array([p[1] for p in ball_pts])
                ball_resamp = interpolate_trajectory(ball_t_arr, ball_p_arr, tg)
            else:
                ball_resamp = np.full((len(tg), 3), np.nan)

            out.append({
                "play_id": play_id, "result_call": call, "pitch_type": ptype,
                "pitch_speed": speed, "pitcher_id": pitcher_id, "game_pk": game_pk,
                "swing_trigger_t": ev.swing_trigger_t,
                "peak_bat_speed_t": ev.peak_bat_speed_t,
                "peak_bat_speed_value": ev.peak_bat_speed_value,
                "joint": joint_resamp,        # (n_grid, n_dim)
                "bat": bat_resamp,             # (n_grid, 6)
                "ball": ball_resamp,           # (n_grid, 3)
                "min_ball_bat_d": d,
                "min_ball_bat_t": (td - ev.swing_trigger_t) if td else None,
            })
        conn.close()
    return out


def divergence_curve(swings):
    """For each time offset in SWING_GRID, compute mean cross-swing variance across joints."""
    if not swings: return None, None, None
    # Stack: (n_swings, n_grid, n_dim)
    n = len(swings)
    ng = len(SWING_GRID)
    nd = swings[0]["joint"].shape[1]
    cube = np.stack([s["joint"] for s in swings], axis=0)  # (n, ng, nd)
    # Per-joint variance across swings, then mean across joints
    # Use NaN-aware ops
    var_per_dim = np.nanvar(cube, axis=0)  # (ng, nd)
    var_total = np.nanmean(var_per_dim, axis=1)  # (ng,)
    return SWING_GRID, var_total, cube


def find_canonical_swing(swings, contact_calls=("X",)):
    """Canonical swing = mean joint trajectory over swings ending in contact (X or F).
    Falls back to mean of all swings if no contact-only data."""
    contact = [s for s in swings if s["result_call"] in contact_calls]
    if len(contact) < 2:
        contact = [s for s in swings if s["result_call"] in ("X", "F")]
    if len(contact) < 2:
        contact = swings
    cube = np.stack([s["joint"] for s in contact], axis=0)
    canonical = np.nanmean(cube, axis=0)  # (n_grid, n_dim)
    return canonical, len(contact)


def deviation_from_canonical(swings, canonical):
    """Per-swing deviation from canonical: time-integrated joint distance."""
    out = []
    for s in swings:
        diff = s["joint"] - canonical  # (n_grid, n_dim)
        # Reshape to (n_grid, n_joints, 3) → 3D distance per joint per time
        diff3 = diff.reshape(len(SWING_GRID), N_JOINTS, 3)
        dist_per_joint_per_t = np.linalg.norm(diff3, axis=2)  # (n_grid, n_joints)
        # Integrate over time, mean across joints
        # Use only POST-trigger frames (where the swing matters)
        post = SWING_GRID >= 0
        deviation = np.nanmean(dist_per_joint_per_t[post])
        out.append(deviation)
    return np.array(out)


def fooled_score(swings, ball_bat_distances, posture_deviations):
    """Combined z-scored fooled-ness score. Lower = looks like canonical swing,
    higher = unusual posture + far from ball."""
    # z-score within batter
    d_arr = np.array([d if d is not None else np.nan for d in ball_bat_distances])
    p_arr = np.array(posture_deviations)
    def zscore(x):
        valid = ~np.isnan(x)
        if valid.sum() < 2: return np.zeros_like(x)
        out = np.zeros_like(x)
        m, s = np.nanmean(x), np.nanstd(x)
        out[valid] = (x[valid] - m) / (s if s > 1e-6 else 1)
        return out
    return zscore(d_arr) + zscore(p_arr), zscore(d_arr), zscore(p_arr)


def cluster_swings(cube, n_clusters=3, n_top_dims=15):
    """Simple k-means on the (n_swings, n_grid * n_dim) flattened POST-trigger window.
    Use top n_top_dims highest-variance dimensions to keep it tractable."""
    n = cube.shape[0]
    if n < n_clusters * 2:
        return None
    post = SWING_GRID >= 0
    cube_post = cube[:, post, :]  # (n, ng_post, nd)
    flat = cube_post.reshape(n, -1)
    # Fill NaN with column mean
    col_mean = np.nanmean(flat, axis=0)
    inds = np.where(np.isnan(flat))
    flat[inds] = np.take(col_mean, inds[1])
    # Use top-variance dims
    var = np.nanvar(flat, axis=0)
    top_idx = np.argsort(-var)[:n_top_dims * cube_post.shape[1]]
    reduced = flat[:, top_idx]
    # K-means via simple Lloyd's
    rng = np.random.default_rng(42)
    init_idx = rng.choice(n, n_clusters, replace=False)
    centers = reduced[init_idx].copy()
    labels = np.zeros(n, dtype=int)
    for it in range(50):
        # Assign
        d = np.linalg.norm(reduced[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.all(new_labels == labels): break
        labels = new_labels
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centers[c] = reduced[mask].mean(axis=0)
    return labels


def plot_divergence(swings, grid, var_curve, out_path, name):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), facecolor="#1a1a1a", sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 1]))
    ax = axes[0]
    # Each swing's pelvis_y trajectory as background
    for s in swings:
        py = s["joint"][:, 0*3 + 1]  # pelvis y
        c = {"X": "#4caf50", "F": "#ff9800", "S": "#f44336"}.get(s["result_call"], "#888")
        ax.plot(grid, py - np.nanmean(py[grid < 0]) if np.any(grid < 0) else py, "-",
                color=c, alpha=0.35, lw=0.8)
    ax.axvline(0, color="white", lw=2, alpha=0.8, label="swing_trigger")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_ylabel("pelvis_y (centered, ft)", color="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.set_title(f"{name} — pelvis_y trajectories during swing window\n"
                 "green=X(in play), orange=F(foul), red=S(whiff)", color="white", fontsize=11)
    ax.legend(facecolor="#222", edgecolor="#666", labelcolor="white", loc="upper left")

    ax2 = axes[1]
    ax2.plot(grid, var_curve, "-", color="#4caf50", lw=2)
    # Find divergence onset (variance > 2× pre-trigger baseline)
    pre = grid < 0
    baseline = np.nanmean(var_curve[pre]) if pre.any() else var_curve[0]
    threshold = 2.0 * baseline
    div_onset_idx = np.argmax(var_curve > threshold)
    if var_curve[div_onset_idx] > threshold:
        div_onset_t = grid[div_onset_idx]
        ax2.axvline(div_onset_t, color="#ff9800", lw=1.5, ls="--",
                    label=f"divergence onset T{div_onset_t:+.2f}s")
        ax2.axhline(threshold, color="#666", lw=0.8, ls=":")
    ax2.axvline(0, color="white", lw=2, alpha=0.8)
    ax2.set_facecolor("#1a1a1a"); ax2.tick_params(colors="white")
    ax2.set_ylabel("cross-swing variance (ft²)", color="white")
    ax2.set_xlabel("time relative to swing_trigger (s)", color="white")
    for s in ax2.spines.values(): s.set_color("#666")
    ax2.set_title("Cross-swing variance vs time — where do swings start to differ?",
                  color="white", fontsize=10)
    ax2.legend(facecolor="#222", edgecolor="#666", labelcolor="white", loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)
    return div_onset_t if var_curve[div_onset_idx] > threshold else None


def plot_canonical_vs_pitches(swings, canonical, deviations, out_path, name):
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#1a1a1a")
    for i, s in enumerate(swings):
        diff = s["joint"] - canonical
        diff3 = diff.reshape(len(SWING_GRID), N_JOINTS, 3)
        d_t = np.nanmean(np.linalg.norm(diff3, axis=2), axis=1)
        c = {"X": "#4caf50", "F": "#ff9800", "S": "#f44336"}.get(s["result_call"], "#888")
        ax.plot(SWING_GRID, d_t, "-", color=c, alpha=0.5, lw=0.9,
                label=f"#{i+1} {s['pitch_type']}/{s['result_call']} dev={deviations[i]:.2f}")
    ax.axvline(0, color="white", lw=2, alpha=0.8, label="swing_trigger")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_xlabel("time relative to swing_trigger (s)", color="white")
    ax.set_ylabel("mean per-joint distance from canonical (ft)", color="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.set_title(f"{name} — each swing's posture deviation from canonical-good-swing\n"
                 "(green=X, orange=F, red=S; legend in CSV due to clutter)",
                 color="white", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def plot_fooled_score(swings, fooled, dz, pz, out_path, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="#1a1a1a")
    # Left: scatter of components
    ax = axes[0]
    colors = ["#4caf50" if s["result_call"] == "X" else
              "#ff9800" if s["result_call"] == "F" else
              "#f44336" if s["result_call"] == "S" else "#888"
              for s in swings]
    ax.scatter(dz, pz, c=colors, alpha=0.75, s=50, edgecolor="white", lw=0.4)
    for i, s in enumerate(swings):
        ax.annotate(f"#{i+1}", (dz[i], pz[i]), color="white", fontsize=6, ha="left", va="bottom")
    ax.axhline(0, color="#666", lw=0.5); ax.axvline(0, color="#666", lw=0.5)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_xlabel("ball-bat distance (z-score)", color="white")
    ax.set_ylabel("posture deviation from canonical (z-score)", color="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.set_title("Fooled-score components per swing\ngreen=X, orange=F, red=S",
                 color="white", fontsize=10)

    # Middle: fooled score histogram by outcome
    ax = axes[1]
    by_call = defaultdict(list)
    for i, s in enumerate(swings):
        if not np.isnan(fooled[i]):
            by_call[s["result_call"]].append(fooled[i])
    for call, color in [("X", "#4caf50"), ("F", "#ff9800"), ("S", "#f44336")]:
        vals = by_call.get(call, [])
        if vals:
            ax.hist(vals, bins=8, alpha=0.5, color=color, edgecolor="white",
                    label=f"{call} n={len(vals)}", lw=0.5)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_xlabel("fooled score (z-sum)", color="white")
    ax.set_ylabel("count", color="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.legend(facecolor="#222", edgecolor="#666", labelcolor="white")
    ax.set_title("Fooled score by outcome", color="white", fontsize=10)

    # Right: per-pitch bar
    ax = axes[2]
    order = np.argsort(fooled)[::-1]
    x = np.arange(len(swings))
    ax.bar(x, [fooled[i] for i in order], color=[colors[i] for i in order], alpha=0.85)
    labels = [f"#{i+1} {swings[i]['pitch_type'] or '?'}/{swings[i]['result_call']}" for i in order]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=70, ha="right",
                                          color="white", fontsize=7)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_ylabel("fooled score", color="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.set_title("Most-to-least fooled swings", color="white", fontsize=10)
    fig.suptitle(f"{name} — fooled-ness analysis", color="white", fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batter-id", type=int, required=True)
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR", "data"))
    ap.add_argument("--out-dir", default="data/oscillation_report/pre_pitch_preparatory_movement/swing_divergence")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    game_pks = list_games(data_dir)

    print(f"loading swings for batter {args.batter_id} from {len(game_pks)} games...")
    swings = load_batter_swings(game_pks, data_dir, args.batter_id)
    print(f"  {len(swings)} usable swings")
    if len(swings) < 3:
        print("  too few swings"); return

    # Look up name
    import urllib.request, json
    try:
        req = urllib.request.Request(
            f"https://statsapi.mlb.com/api/v1/people/{args.batter_id}",
            headers={"User-Agent": "FV/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            name = json.loads(r.read())["people"][0]["fullName"]
    except Exception:
        name = f"player_{args.batter_id}"
    print(f"  name: {name}")

    out_dir = Path(args.out_dir) / f"{args.batter_id}_{name.replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Divergence curve
    grid, var_curve, cube = divergence_curve(swings)
    div_onset_t = plot_divergence(swings, grid, var_curve, out_dir / "divergence_curve.png", name)
    print(f"  divergence onset: T{div_onset_t:+.2f}s relative to swing_trigger" if div_onset_t else "  no clear divergence onset")

    # 2. Canonical good swing + deviations
    canonical, n_canonical = find_canonical_swing(swings)
    deviations = deviation_from_canonical(swings, canonical)
    plot_canonical_vs_pitches(swings, canonical, deviations, out_dir / "canonical_deviation.png", name)
    print(f"  canonical built from n={n_canonical} swings")

    # 3. Fooled score
    bb_dists = [s["min_ball_bat_d"] for s in swings]
    fooled, dz, pz = fooled_score(swings, bb_dists, deviations)
    plot_fooled_score(swings, fooled, dz, pz, out_dir / "fooled_score.png", name)

    # 4. Clustering
    labels = cluster_swings(cube, n_clusters=3, n_top_dims=15)
    if labels is not None:
        from collections import Counter
        print(f"  cluster sizes: {Counter(labels)}")

    # 5. Per-pitch CSV
    csv_path = out_dir / "per_swing_features.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["swing_idx", "play_id", "pitch_type", "result_call", "pitch_speed",
                  "pitcher_id", "swing_trigger_t_rel", "peak_bat_speed",
                  "min_ball_bat_d", "min_ball_bat_t_rel",
                  "posture_deviation", "ball_bat_z", "posture_z",
                  "fooled_score", "cluster"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, s in enumerate(swings):
            w.writerow({
                "swing_idx": i+1, "play_id": s["play_id"],
                "pitch_type": s["pitch_type"], "result_call": s["result_call"],
                "pitch_speed": s["pitch_speed"], "pitcher_id": s["pitcher_id"],
                "swing_trigger_t_rel": s["swing_trigger_t"],
                "peak_bat_speed": s["peak_bat_speed_value"],
                "min_ball_bat_d": s["min_ball_bat_d"],
                "min_ball_bat_t_rel": s["min_ball_bat_t"],
                "posture_deviation": deviations[i],
                "ball_bat_z": dz[i], "posture_z": pz[i],
                "fooled_score": fooled[i],
                "cluster": int(labels[i]) if labels is not None else "",
            })
    print(f"  csv: {csv_path}")
    print(f"  outputs in {out_dir}/")


if __name__ == "__main__":
    main()
