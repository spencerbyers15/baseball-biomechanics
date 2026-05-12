"""Verify the template-matching stance filter generalizes across batters.

For each batter with enough clean pitches in the census, compute the
per-pitch similarity-to-mean-stance trace (template = last 1s of pre-osc)
and plot the aggregate. If a single similarity threshold (e.g., 0.95)
cleanly separates stance from pre-stance for all batters, the filter is
batter-agnostic and we can wire it into the census.

Output:
  - data/oscillation_report/pre_pitch_preparatory_movement/template_filter_all_batters.png  (grid of
    aggregate similarity traces, one per batter)
  - prints per-batter "stance-entry" time at three thresholds (0.95, 0.97, 0.99)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games, open_game
from fieldvision.storage import JOINT_COLS
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

PRE_OSC_SECONDS = 5.0
REFERENCE_SECONDS = 1.0  # last second before windup_onset = stance reference
JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
N_JOINTS = len(JOINT_BIDS)
HAND_RT_BID, HAND_LT_BID = 28, 67


def build_posture_vec(joints, head, handle):
    """66-dim centroid-normalized posture (60 joints + 6 bat dims)."""
    body_centroid = joints.mean(axis=0)
    joints_norm = (joints - body_centroid).flatten()
    head_norm = head - body_centroid
    handle_norm = handle - body_centroid
    return np.concatenate([joints_norm, head_norm, handle_norm])


def load_pitch_postures(conn, batter_id, pitcher_id, release_t):
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
    if not assess_pitch_quality(batter_frames, bat_records)["is_clean"]: return None

    bat_times = np.array([b[0] for b in bat_records])
    ts, vecs = [], []
    for t, wp in batter_frames:
        if not (pre_lo <= t <= pre_hi): continue
        joints = np.full((N_JOINTS, 3), np.nan)
        for j, bid in enumerate(JOINT_BIDS):
            if bid in wp: joints[j] = wp[bid]
        if np.any(np.isnan(joints)): continue
        if len(bat_times) == 0: continue
        bi = int(np.argmin(np.abs(bat_times - t)))
        if abs(bat_times[bi] - t) > 0.07: continue
        head = np.array(bat_records[bi][1])
        handle = np.array(bat_records[bi][2])
        vec = build_posture_vec(joints, head, handle)
        ts.append(t); vecs.append(vec)
    if len(ts) < 30: return None
    return np.array(ts), np.array(vecs), ev.windup_onset_t


def compute_batter_trace(batter_id, side):
    """Returns (grid, median_trace, q25, q75, n_pitches) for one batter, or None."""
    grid = np.linspace(0, PRE_OSC_SECONDS, 151)
    sims_grid = []
    for game_pk in list_games(Path(os.environ.get("FV_DATA_DIR", "data"))):
        conn = open_game(game_pk, Path(os.environ.get("FV_DATA_DIR", "data")))
        try:
            rows = conn.execute(
                "SELECT play_id, pitcher_id, start_time_unix, batter_side "
                "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL",
                (batter_id,)).fetchall()
        except Exception:
            conn.close(); continue
        for play_id, pid, t_rel, batter_side in rows:
            # Filter by side (for switch hitters)
            bs = batter_side or "?"
            if bs == "S": bs = side  # if uncertain, use census's recorded side
            if bs != side: continue
            r = load_pitch_postures(conn, batter_id, pid, t_rel)
            if r is None: continue
            ts, vecs, wuo = r
            t_before = wuo - ts
            ref_mask = (t_before >= 0) & (t_before < REFERENCE_SECONDS)
            if ref_mask.sum() < 10: continue
            ref_mean = vecs[ref_mask].mean(axis=0)
            ref_norm = np.linalg.norm(ref_mean) + 1e-9
            sims = (vecs @ ref_mean) / (np.linalg.norm(vecs, axis=1) * ref_norm + 1e-9)
            order = np.argsort(t_before)
            sims_grid.append(np.interp(grid, t_before[order], sims[order],
                                        left=np.nan, right=np.nan))
        conn.close()
    if not sims_grid: return None
    sims_grid = np.array(sims_grid)
    med = np.nanmedian(sims_grid, axis=0)
    q25 = np.nanpercentile(sims_grid, 25, axis=0)
    q75 = np.nanpercentile(sims_grid, 75, axis=0)
    return grid, med, q25, q75, len(sims_grid)


def main():
    out = "data/oscillation_report/pre_pitch_preparatory_movement/template_filter_all_batters.png"
    census = pickle.load(open("data/oscillation_report/pre_pitch_preparatory_movement/census.pkl", "rb"))

    # Pick top 12 batters by usable-pitch count
    by_n = []
    for key, r in census["results"].items():
        bid = r["batter_id"]
        side = key.split("_")[-1]
        name = census["names"][key]
        by_n.append((r["n_pitches_usable"], bid, side, name))
    by_n.sort(reverse=True)
    top = by_n[:12]
    print(f"Diagnosing top {len(top)} batters by pitch count:")
    for n, bid, side, name in top:
        print(f"  {name} ({side}) n={n}")

    fig, axes = plt.subplots(3, 4, figsize=(20, 13), facecolor="#1a1a1a")
    thresholds = [0.99, 0.97, 0.95]
    threshold_colors = ["#a5d6a7", "#ffca28", "#f44336"]

    print(f"\nStance-entry time (sec before windup_onset) per batter:")
    print(f"  threshold:           0.99    0.97    0.95")
    for ax, (n, bid, side, name) in zip(axes.flat, top):
        ax.set_facecolor("#0a0a0a")
        r = compute_batter_trace(bid, side)
        if r is None:
            ax.set_title(f"{name} ({side}) — no data", color="white", fontsize=9)
            continue
        grid, med, q25, q75, n_pitches = r
        ax.fill_between(grid, q25, q75, color="#42a5f5", alpha=0.25)
        ax.plot(grid, med, "-", color="#42a5f5", lw=1.6)
        ax.axvspan(0, REFERENCE_SECONDS, alpha=0.15, color="#fbc02d")
        for thresh, color in zip(thresholds, threshold_colors):
            below = np.where(med < thresh)[0]
            if len(below):
                t = grid[below[0]]
                ax.axhline(thresh, color=color, lw=0.5, ls=":")
                ax.axvline(t, color=color, lw=0.5, ls=":")
        ax.set_xlim(0, PRE_OSC_SECONDS); ax.invert_xaxis()
        ax.set_ylim(0.5, 1.02)
        ax.tick_params(colors="white", labelsize=7)
        for s in ax.spines.values(): s.set_color("#666")
        ax.set_title(f"{name} ({side}) — n={n_pitches}", color="white", fontsize=9)
        # print per-batter entries
        entries = []
        for thresh in thresholds:
            below = np.where(med < thresh)[0]
            entries.append(grid[below[0]] if len(below) else float("nan"))
        print(f"  {name[:22]:<22} ({side})  "
              f"{entries[0]:>5.2f}s  {entries[1]:>5.2f}s  {entries[2]:>5.2f}s")

    fig.suptitle(
        "Template-matching stance filter across batters\n"
        "blue = median cosine-similarity to per-pitch mean stance (last 1s ref window)\n"
        "horizontal lines: candidate thresholds (green=0.99, yellow=0.97, red=0.95)\n"
        "vertical: where the median first drops below each threshold",
        color="white", fontsize=11, y=0.995)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, facecolor="#1a1a1a", bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
