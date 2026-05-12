"""Template-matching diagnostic for stance entry detection.

For each Manzardo pitch:
  1. Take the LAST ~1 second of pre-osc (right before windup_onset) — we
     trust this window to be in-stance.
  2. Build the mean stance pose from those frames (66-dim, centroid-normalized).
  3. For every frame in pre-osc, compute similarity to the mean stance pose.
  4. Plot similarity vs (time before windup_onset). Walking backward, we
     expect: similarity ≈ 1 during stance, then a clear drop at the moment
     the batter enters stance, then low/noisy values earlier (pre-stance).

If the drop is consistent and sharp across pitches, that's the stance-entry
signature — a single threshold on this signal gives us a clean filter.
"""

from __future__ import annotations

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
MANZARDO = 700932
JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
N_JOINTS = len(JOINT_BIDS)
HAND_RT_BID, HAND_LT_BID = 28, 67

# Reference window: last REFERENCE_SECONDS before windup_onset
REFERENCE_SECONDS = 1.0


def build_posture_vec(joints_arr, head, handle):
    """66-dim centroid-normalized posture: 60 joint dims + bat head + bat handle.
    joints_arr is shape (N_JOINTS, 3)."""
    body_centroid = joints_arr.mean(axis=0)
    joints_norm = (joints_arr - body_centroid).flatten()
    head_norm = head - body_centroid
    handle_norm = handle - body_centroid
    return np.concatenate([joints_norm, head_norm, handle_norm])


def load_pitch_postures(conn, batter_id, pitcher_id, release_t):
    """Returns (times, posture_matrix, windup_onset_t)."""
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


def main():
    out = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_stance_template_similarity.png"

    pitch_traces = []  # list of (t_before_wuo, similarity)
    for game_pk in list_games(Path(os.environ.get("FV_DATA_DIR", "data"))):
        conn = open_game(game_pk, Path(os.environ.get("FV_DATA_DIR", "data")))
        try:
            rows = conn.execute(
                "SELECT play_id, pitcher_id, start_time_unix, result_call "
                "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL",
                (MANZARDO,)).fetchall()
        except Exception:
            conn.close(); continue
        for play_id, pid, t_rel, call in rows:
            r = load_pitch_postures(conn, MANZARDO, pid, t_rel)
            if r is None: continue
            ts, vecs, wuo = r
            t_before = wuo - ts  # 0 at windup_onset, PRE_OSC_SECONDS at pre_lo

            # Reference window: last REFERENCE_SECONDS before windup_onset
            ref_mask = (t_before >= 0) & (t_before < REFERENCE_SECONDS)
            if ref_mask.sum() < 10: continue
            ref_mean = vecs[ref_mask].mean(axis=0)

            # Cosine similarity to reference mean
            # (handles scale and is bounded in [-1, 1])
            ref_norm = np.linalg.norm(ref_mean) + 1e-9
            sims = (vecs @ ref_mean) / (np.linalg.norm(vecs, axis=1) * ref_norm + 1e-9)
            pitch_traces.append({
                "play_id": play_id, "call": call,
                "t_before": t_before, "sim": sims,
            })
        conn.close()

    print(f"Loaded {len(pitch_traces)} clean Manzardo pitches")

    # Aggregate: interpolate each pitch onto a common grid, then take median + IQR
    grid = np.linspace(0, PRE_OSC_SECONDS, 151)  # 151 pts over [0, 5s] before WUO
    sims_grid = []
    for tr in pitch_traces:
        # tr["t_before"] is in increasing order from windup_onset back
        order = np.argsort(tr["t_before"])
        tb = tr["t_before"][order]
        sm = tr["sim"][order]
        sims_grid.append(np.interp(grid, tb, sm, left=np.nan, right=np.nan))
    sims_grid = np.array(sims_grid)
    med = np.nanmedian(sims_grid, axis=0)
    q25 = np.nanpercentile(sims_grid, 25, axis=0)
    q75 = np.nanpercentile(sims_grid, 75, axis=0)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="#1a1a1a",
                              gridspec_kw={"height_ratios": [1, 1.2]})

    # Panel 1: all pitches overlaid
    ax = axes[0]
    ax.set_facecolor("#0a0a0a")
    for tr in pitch_traces:
        color = "#4caf50" if tr["call"] == "X" else "#f44336" if tr["call"] == "S" else "#fbc02d" if tr["call"] == "F" else "#42a5f5"
        ax.plot(tr["t_before"], tr["sim"], "-", color=color, lw=0.4, alpha=0.3)
    ax.axvspan(0, REFERENCE_SECONDS, alpha=0.15, color="#fbc02d",
               label=f"reference window (last {REFERENCE_SECONDS}s)")
    ax.set_xlim(0, PRE_OSC_SECONDS)
    ax.invert_xaxis()  # time goes RIGHT to LEFT (right = windup_onset, left = pre_lo)
    ax.set_xlabel("seconds before windup_onset", color="white")
    ax.set_ylabel("cosine similarity to mean stance", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.legend(facecolor="#1a1a1a", edgecolor="#666", labelcolor="white", fontsize=9)
    ax.set_title(f"Per-pitch similarity to per-pitch mean stance ({len(pitch_traces)} pitches overlaid)",
                 color="white", fontsize=11)

    # Panel 2: median + IQR
    ax = axes[1]
    ax.set_facecolor("#0a0a0a")
    ax.fill_between(grid, q25, q75, color="#42a5f5", alpha=0.25, label="IQR (25-75%)")
    ax.plot(grid, med, "-", color="#42a5f5", lw=2.2, label="median across pitches")
    ax.axvspan(0, REFERENCE_SECONDS, alpha=0.15, color="#fbc02d",
               label=f"reference window")
    ax.set_xlim(0, PRE_OSC_SECONDS)
    ax.invert_xaxis()
    ax.set_xlabel("seconds before windup_onset", color="white")
    ax.set_ylabel("cosine similarity to mean stance", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.legend(facecolor="#1a1a1a", edgecolor="#666", labelcolor="white", fontsize=9)
    ax.set_title("Aggregate similarity trace (median, IQR across pitches)",
                 color="white", fontsize=11)
    # Annotate plausible stance-entry threshold
    if not np.all(np.isnan(med)):
        for thresh, color in [(0.99, "#a5d6a7"), (0.97, "#ffca28"), (0.95, "#f44336")]:
            below_idx = np.where(med < thresh)[0]
            if len(below_idx):
                t_below = grid[below_idx[0]]
                ax.axhline(thresh, color=color, lw=0.6, ls=":")
                ax.text(0.05, thresh + 0.001, f"median crosses {thresh} at {t_below:.2f}s before WUO",
                        color=color, fontsize=8, transform=ax.transData)

    fig.suptitle(
        f"Manzardo: template-matching stance detection\n"
        f"reference = last {REFERENCE_SECONDS}s before windup_onset (definitely in stance), "
        f"track similarity walking backward",
        color="white", fontsize=12)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, facecolor="#1a1a1a", bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
