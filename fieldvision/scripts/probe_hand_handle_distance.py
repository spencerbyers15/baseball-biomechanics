"""Probe distribution of (hand_rt, hand_lt) -> bat_handle distances across
all of Manzardo's pre-osc frames. Used to calibrate the hands-on-bat
threshold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import list_games, open_game
from fieldvision.storage import JOINT_COLS
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

MANZARDO = 700932
JOINT_BIDS = [bid for bid, _ in JOINT_COLS]


def main():
    joint_cols_select = ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)
    d_rt_all, d_lt_all = [], []

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
            prr = load_clean_batter_actor_frames(
                conn, pitcher_id, t_rel - 5, t_rel + 2.5, joint_cols_select)
            pf = []
            for r0 in [(rr[0],) + rr[2:] for rr in prr]:
                wp = {}
                for i, bid in enumerate(JOINT_BIDS):
                    x, y, z = r0[1+i*3], r0[2+i*3], r0[3+i*3]
                    if x is not None: wp[bid] = (x, y, z)
                pf.append((r0[0], wp))
            if len(pf) < 30: continue
            ev = detect_pitcher_events(pf, t_rel, search_back=4.0)
            if ev.windup_onset_t is None: continue
            pre_lo, pre_hi = ev.windup_onset_t - 5, ev.windup_onset_t

            brr = load_clean_batter_actor_frames(
                conn, MANZARDO, pre_lo - 0.2, pre_hi + 0.2, joint_cols_select)
            bf = []
            for r0 in [(rr[0],) + rr[2:] for rr in brr]:
                wp = {}
                for i, bid in enumerate(JOINT_BIDS):
                    x, y, z = r0[1+i*3], r0[2+i*3], r0[3+i*3]
                    if x is not None: wp[bid] = (x, y, z)
                bf.append((r0[0], wp))
            bat_rows = conn.execute(
                "SELECT time_unix, head_x, head_y, head_z, handle_x, handle_y, handle_z "
                "FROM bat_frame WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
                (pre_lo - 0.2, pre_hi + 0.2)).fetchall()
            bat_records = filter_bat_frames(
                [(rr[0], (rr[1], rr[2], rr[3]), (rr[4], rr[5], rr[6])) for rr in bat_rows])
            if not assess_pitch_quality(bf, bat_records)["is_clean"]: continue

            bat_times = np.array([b[0] for b in bat_records])
            for t, wp in bf:
                if not (pre_lo <= t <= pre_hi): continue
                if 28 not in wp or 67 not in wp: continue
                if len(bat_times) == 0: continue
                bi = int(np.argmin(np.abs(bat_times - t)))
                if abs(bat_times[bi] - t) > 0.07: continue
                handle = np.array(bat_records[bi][2])
                hand_rt = np.array(wp[28])
                hand_lt = np.array(wp[67])
                d_rt_all.append(np.linalg.norm(hand_rt - handle))
                d_lt_all.append(np.linalg.norm(hand_lt - handle))
        conn.close()

    d_rt = np.array(d_rt_all); d_lt = np.array(d_lt_all)
    print(f"n_frames={len(d_rt)}")
    print(f"hand_rt -> handle:  median={np.median(d_rt):.2f}  "
          f"p10={np.percentile(d_rt,10):.2f}  p90={np.percentile(d_rt,90):.2f}  "
          f"p99={np.percentile(d_rt,99):.2f}")
    print(f"hand_lt -> handle:  median={np.median(d_lt):.2f}  "
          f"p10={np.percentile(d_lt,10):.2f}  p90={np.percentile(d_lt,90):.2f}  "
          f"p99={np.percentile(d_lt,99):.2f}")
    # Distance of furthest hand per frame (the one that's looser)
    further = np.maximum(d_rt, d_lt)
    print(f"max(hand) -> handle: median={np.median(further):.2f}  "
          f"p90={np.percentile(further,90):.2f}  p95={np.percentile(further,95):.2f}  "
          f"p99={np.percentile(further,99):.2f}")

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#1a1a1a")
    for ax, data, title in zip(axes, [d_rt, d_lt, further],
                                ["hand_rt → handle", "hand_lt → handle", "max(hand) → handle"]):
        ax.set_facecolor("#0a0a0a")
        ax.hist(data, bins=80, color="#42a5f5", edgecolor="white", lw=0.3)
        for q, color in [(50, "#fbc02d"), (90, "#ff9800"), (99, "#f44336")]:
            v = np.percentile(data, q)
            ax.axvline(v, color=color, lw=1.2, ls="--", label=f"p{q}={v:.2f} ft")
        ax.legend(facecolor="#1a1a1a", edgecolor="#666", labelcolor="white", fontsize=8)
        ax.set_xlabel("distance (ft)", color="white")
        ax.set_ylabel("frames", color="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#666")
        ax.set_title(title, color="white", fontsize=10)
    fig.suptitle("Manzardo: hand→handle distance distribution (all pre-osc frames, all clean pitches)",
                 color="white", fontsize=11)
    plt.tight_layout()
    out = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_hand_handle_dist.png"
    plt.savefig(out, dpi=110, facecolor="#1a1a1a", bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
