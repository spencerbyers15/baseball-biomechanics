"""Probe distribution of bat orientation in Manzardo's pre-osc frames where
hands are already on the bat. We want to know: do clearly-non-stance moments
show up as obvious outliers in simple bat geometry signals?

For each frame we compute:
  - bat_head_y minus bat_handle_y: how much bat tip is above grip (ft)
  - vertical angle of the bat (arcsin((head_y - handle_y) / bat_length))
  - bat tilt back-vs-forward (head_x - handle_x; sign tells which way the bat
    points relative to catcher direction)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.storage import JOINT_COLS
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

MANZARDO = 700932
JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
HAND_RT_BID, HAND_LT_BID = 28, 67
HAND_TO_HANDLE_MAX_FT = 1.0


def main():
    out = "data/oscillation_report/pre_pitch_preparatory_movement/manzardo_bat_orientation.png"
    joint_cols_select = ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)

    head_minus_handle_y = []
    bat_pitch_deg = []
    head_minus_handle_x = []

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
        for play_id, pid, t_rel in rows:
            prr = load_clean_batter_actor_frames(
                conn, pid, t_rel - 5, t_rel + 2.5, joint_cols_select)
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
            pre_lo, pre_hi = ev.windup_onset_t - 5.0, ev.windup_onset_t

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
                if HAND_RT_BID not in wp or HAND_LT_BID not in wp: continue
                if len(bat_times) == 0: continue
                bi = int(np.argmin(np.abs(bat_times - t)))
                if abs(bat_times[bi] - t) > 0.07: continue
                head = np.array(bat_records[bi][1])
                handle = np.array(bat_records[bi][2])
                # hands-on-bat gate
                if (np.linalg.norm(np.array(wp[HAND_RT_BID]) - handle) > HAND_TO_HANDLE_MAX_FT or
                    np.linalg.norm(np.array(wp[HAND_LT_BID]) - handle) > HAND_TO_HANDLE_MAX_FT):
                    continue
                v = head - handle
                blen = np.linalg.norm(v)
                if blen < 1e-3: continue
                head_minus_handle_y.append(v[1])
                head_minus_handle_x.append(v[0])
                bat_pitch_deg.append(np.degrees(np.arcsin(v[1] / blen)))
        conn.close()

    head_minus_handle_y = np.array(head_minus_handle_y)
    head_minus_handle_x = np.array(head_minus_handle_x)
    bat_pitch_deg = np.array(bat_pitch_deg)
    n = len(bat_pitch_deg)
    print(f"n_frames={n}  (pre-osc + hands-on-bat-gated)")
    print(f"head_y - handle_y (ft):  median={np.median(head_minus_handle_y):.2f}  "
          f"p10={np.percentile(head_minus_handle_y,10):.2f}  "
          f"p90={np.percentile(head_minus_handle_y,90):.2f}  "
          f"p99={np.percentile(head_minus_handle_y,99):.2f}  "
          f"min={head_minus_handle_y.min():.2f}  max={head_minus_handle_y.max():.2f}")
    print(f"bat pitch (deg):          median={np.median(bat_pitch_deg):.0f}  "
          f"p10={np.percentile(bat_pitch_deg,10):.0f}  "
          f"p90={np.percentile(bat_pitch_deg,90):.0f}  "
          f"p99={np.percentile(bat_pitch_deg,99):.0f}  "
          f"min={bat_pitch_deg.min():.0f}  max={bat_pitch_deg.max():.0f}")
    print(f"head_x - handle_x (ft):  median={np.median(head_minus_handle_x):.2f}  "
          f"p10={np.percentile(head_minus_handle_x,10):.2f}  "
          f"p90={np.percentile(head_minus_handle_x,90):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a1a")
    for ax, data, title in [
        (axes[0], head_minus_handle_y, "head_y − handle_y (ft) — bat tip vs grip"),
        (axes[1], bat_pitch_deg, "bat vertical pitch (degrees) — 90 = bat straight up"),
    ]:
        ax.set_facecolor("#0a0a0a")
        ax.hist(data, bins=80, color="#42a5f5", edgecolor="white", lw=0.3)
        for q, color in [(10, "#fbc02d"), (90, "#fbc02d"), (50, "#ff9800")]:
            v = np.percentile(data, q)
            ax.axvline(v, color=color, lw=1.0, ls="--", label=f"p{q}={v:.1f}")
        ax.legend(facecolor="#1a1a1a", edgecolor="#666", labelcolor="white", fontsize=8)
        ax.set_xlabel(title, color="white")
        ax.set_ylabel("frames", color="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#666")
    fig.suptitle("Manzardo bat orientation distribution (hands-on-bat-gated pre-osc frames)",
                 color="white", fontsize=11)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, facecolor="#1a1a1a", bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
