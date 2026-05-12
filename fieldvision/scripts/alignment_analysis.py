"""Q2 + Q3: per-batter alignment between batter swing_trigger and pitcher events.

Q2: phase at batter swing_trigger (WHERE in the waggle was the batter when he started his stroke?)
Q3: latency = batter_swing_trigger − pitcher_event_time for each of the 4 pitcher events,
    distribution per batter. Low std = consistent reaction timing.

Also produces:
  - aggregate cross-batter scatter: median latency vs phase-lock R at windup_onset
    (do well-locked batters react faster/more consistently?)
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
from fieldvision.batter_kinematics import detect_batter_events
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.storage import JOINT_COLS
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames, assess_pitch_quality)

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]

# Use the existing PRE_OSC window + Hilbert/PCA phase from the census pipeline
PRE_OSC_SECONDS = 5.0
SAMPLE_HZ = 30.0
N_PRE_OSC_FRAMES = int(PRE_OSC_SECONDS * SAMPLE_HZ)


def load_pitch(conn, batter_id, play_id, pitcher_id, release_t):
    joint_cols_select = ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)
    p_rows_raw = load_clean_batter_actor_frames(
        conn, pitcher_id, release_t - 5, release_t + 0.5, joint_cols_select)
    p_rows = [(r[0],) + r[2:] for r in p_rows_raw]
    if len(p_rows) < 30: return None
    p_frames = []
    for r in p_rows:
        wp = {}
        for i, jbid in enumerate(JOINT_BIDS):
            x, y, z = r[1+i*3], r[2+i*3], r[3+i*3]
            if x is not None: wp[jbid] = (x, y, z)
        p_frames.append((r[0], wp))
    pev = detect_pitcher_events(p_frames, release_t, search_back=4.0)
    if pev.windup_onset_t is None: return None

    b_rows_raw = load_clean_batter_actor_frames(
        conn, batter_id, pev.windup_onset_t - PRE_OSC_SECONDS, release_t + 0.8, joint_cols_select)
    b_rows = [(r[0],) + r[2:] for r in b_rows_raw]
    if len(b_rows) < 80: return None
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
        (release_t - 2, release_t + 0.8)).fetchall()
    bat_frames_raw = [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bat_rows]
    bat_frames = filter_bat_frames(bat_frames_raw)
    quality = assess_pitch_quality(b_frames, bat_frames)
    if not quality["is_clean"]:
        return None
    bev = detect_batter_events(b_frames, bat_frames, release_t)
    return {
        "release_t": release_t,
        "windup_onset_t": pev.windup_onset_t,
        "knee_high_t": pev.knee_high_t,
        "foot_landing_t": pev.foot_landing_t,
        "swing_trigger_t": bev.swing_trigger_t if bev.has_swing else None,
        "has_swing": bev.has_swing,
        "b_frames": b_frames,
    }


def compute_waggle_phase_at(b_frames, win_lo, win_hi, target_t):
    """Build PCA on waggle window then return PC1×PC2 phase at target_t.
    Returns (phase_angle_rad, ok)."""
    # Restrict to pre-osc window for PCA fit
    pre = [(t, wp) for (t, wp) in b_frames if win_lo <= t <= win_hi]
    if len(pre) < 50: return np.nan, False
    n_dim = len(JOINT_BIDS) * 3
    arr = np.full((len(pre), n_dim), np.nan)
    for i, (_, wp) in enumerate(pre):
        for j, bid in enumerate(JOINT_BIDS):
            if bid in wp: arr[i, j*3:j*3+3] = wp[bid]
    valid = ~np.any(np.isnan(arr), axis=1)
    if valid.sum() < 50: return np.nan, False
    centered = arr[valid] - arr[valid].mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan, False
    pc = Vt[:2]
    mean_pose = arr[valid].mean(axis=0)
    # Find frame nearest target_t in the FULL batter-frame array (not just pre)
    full_times = np.array([t for t, _ in b_frames])
    idx = int(np.argmin(np.abs(full_times - target_t)))
    wp = b_frames[idx][1]
    vec = np.full(n_dim, np.nan)
    for j, bid in enumerate(JOINT_BIDS):
        if bid in wp: vec[j*3:j*3+3] = wp[bid]
    if np.any(np.isnan(vec)): return np.nan, False
    proj = (vec - mean_pose) @ pc.T
    return float(np.arctan2(proj[1], proj[0])), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR", "data"))
    ap.add_argument("--out-dir", default="data/oscillation_report/pre_pitch_preparatory_movement")
    ap.add_argument("--min-swings", type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Build batter pitch list across games, keyed by (mlb_id, side) so switch
    # hitters' L and R at-bats are analyzed separately.
    by_batter = defaultdict(list)
    for game_pk in list_games(data_dir):
        try:
            conn = open_game(game_pk, data_dir)
        except Exception: continue
        for r in conn.execute(
            "SELECT batter_id, play_id, pitcher_id, start_time_unix, "
            "result_call, pitch_type, batter_side FROM pitch_label "
            "WHERE batter_id IS NOT NULL AND start_time_unix IS NOT NULL"
        ).fetchall():
            side = r[6] if r[6] in ("L", "R") else "?"
            by_batter[(r[0], side)].append({"game_pk": game_pk, "play_id": r[1], "pitcher_id": r[2],
                                    "release_t": r[3], "result_call": r[4], "pitch_type": r[5]})
        conn.close()

    # Analyze each batter with >= min_swings detectable swings
    out_rows = []
    import urllib.request, json
    name_cache = {}
    def get_name(bid):
        if bid in name_cache: return name_cache[bid]
        try:
            req = urllib.request.Request(f"https://statsapi.mlb.com/api/v1/people/{bid}",
                                         headers={"User-Agent": "FV/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                name_cache[bid] = json.loads(r.read())["people"][0]["fullName"]
        except Exception:
            name_cache[bid] = f"player_{bid}"
        return name_cache[bid]

    candidates = sorted(by_batter.items(), key=lambda x: -len(x[1]))
    print(f"{len(candidates)} batter (id, side) candidates total")
    for key, pitches in candidates:
        bid, side = key
        if len(pitches) < args.min_swings * 2: continue  # heuristic: half might be swings
        # Open conns
        conns = {}
        per_pitch = []
        for p in pitches:
            if p["game_pk"] not in conns:
                conns[p["game_pk"]] = open_game(p["game_pk"], data_dir)
            d = load_pitch(conns[p["game_pk"]], bid, p["play_id"], p["pitcher_id"], p["release_t"])
            if d is None: continue
            d.update(result_call=p["result_call"], pitch_type=p["pitch_type"])
            per_pitch.append(d)
        for c in conns.values(): c.close()
        swings = [p for p in per_pitch if p["has_swing"]]
        if len(swings) < args.min_swings: continue
        # Check if this player appears on both sides — if so, suffix side to name
        n_sides_for_this_id = sum(1 for (b, s), _ in candidates if b == bid)
        base = get_name(bid)
        name = f"{base} ({side})" if n_sides_for_this_id > 1 else base
        # Compute alignment latencies (skip None events)
        def lat(ev_key):
            return np.array([s["swing_trigger_t"] - s[ev_key] for s in swings
                             if s.get(ev_key) is not None])
        L_windup = lat("windup_onset_t")
        L_knee = lat("knee_high_t")
        L_foot = lat("foot_landing_t")
        L_release = lat("release_t")
        if len(L_release) == 0: continue
        def med_std(arr):
            return (float(np.median(arr)), float(np.std(arr))) if len(arr) > 0 else (np.nan, np.nan)
        med_L_windup, std_L_windup = med_std(L_windup)
        med_L_knee, std_L_knee = med_std(L_knee)
        med_L_foot, std_L_foot = med_std(L_foot)
        med_L_release, std_L_release = med_std(L_release)
        # Phase at swing_trigger using waggle PCA per pitch
        phases_swing_trigger = []
        for s in swings:
            phase, ok = compute_waggle_phase_at(
                s["b_frames"],
                win_lo=s["windup_onset_t"] - PRE_OSC_SECONDS,
                win_hi=s["windup_onset_t"],
                target_t=s["swing_trigger_t"])
            if ok: phases_swing_trigger.append(phase)
        phases_swing_trigger = np.array(phases_swing_trigger)
        R_swing_trigger = (float(np.abs(np.mean(np.exp(1j * phases_swing_trigger))))
                           if len(phases_swing_trigger) > 1 else np.nan)
        out_rows.append({
            "batter_id": f"{bid}_{side}", "name": name, "n_swings": len(swings),
            "batter_side": side,
            "median_L_windup": med_L_windup,
            "median_L_knee_high": med_L_knee,
            "median_L_foot_land": med_L_foot,
            "median_L_release": med_L_release,
            "std_L_windup": std_L_windup,
            "std_L_knee_high": std_L_knee,
            "std_L_foot_land": std_L_foot,
            "std_L_release": std_L_release,
            "R_phase_at_swing_trigger": R_swing_trigger,
            "n_phase_at_trigger": len(phases_swing_trigger),
        })
        print(f"  {name:25s} n_sw={len(swings):2d}  "
              f"med_L_release={med_L_release:+.2f}s±{std_L_release:.2f}  "
              f"med_L_windup={med_L_windup:+.2f}s  "
              f"phase_at_trig R={R_swing_trigger:.2f} (n={len(phases_swing_trigger)})")

    if not out_rows:
        print("no batters with enough swings"); return

    # Save CSV
    csv_path = out_dir / "alignment_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows: w.writerow(r)
    print(f"\nwrote {csv_path}")

    # Plot: latency distribution per batter (boxplot per event)
    # Filter to batters with all 4 events to keep plot clean
    plotted = [r for r in out_rows if not np.isnan(r["median_L_windup"])
               and not np.isnan(r["median_L_knee_high"])
               and not np.isnan(r["median_L_foot_land"])
               and not np.isnan(r["median_L_release"])]
    plotted.sort(key=lambda r: r["median_L_release"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#1a1a1a")
    key_to_full = {"windup": "windup", "knee_high": "knee_high",
                   "foot_land": "foot_land", "release": "release"}
    for ax_i, key in enumerate(["windup", "knee_high", "foot_land", "release"]):
        ax = axes.flat[ax_i]
        meds = [r[f"median_L_{key}"] for r in plotted]
        stds = [r[f"std_L_{key}"] for r in plotted]
        names = [r["name"] for r in plotted]
        ns = [r["n_swings"] for r in plotted]
        y = np.arange(len(plotted))
        ax.barh(y, meds, xerr=stds, color="#4caf50", alpha=0.85, capsize=2,
                error_kw=dict(ecolor="#aaa", lw=0.6))
        ax.set_yticks(y); ax.set_yticklabels([f"{n} (n={ns[i]})" for i, n in enumerate(names)],
                                              color="white", fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="white", lw=1, alpha=0.5)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        ax.set_xlabel(f"median (swing_trigger − pitcher_{key}_t) seconds  ±std", color="white", fontsize=9)
        for s in ax.spines.values(): s.set_color("#666")
        ax.set_title(f"Alignment to pitcher {key}", color="white", fontsize=10)
    fig.suptitle("Batter swing_trigger timing relative to pitcher delivery events\n"
                 "positive = batter triggers AFTER the pitcher event; negative = before",
                 color="white", fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(out_dir / "alignment_latencies.png", dpi=110, facecolor="#1a1a1a")
    plt.close(fig)
    print(f"saved {out_dir/'alignment_latencies.png'}")


if __name__ == "__main__":
    main()
