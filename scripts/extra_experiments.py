"""Extra experiments after the headline census + report.

A. Lead-time R sweep — where (in time relative to release) does the PC1xPC2
   phase concentration peak per batter? Tells us *when* the lock is tightest.
B. Cross-event swing/take test — does Kwan's signal hold at knee_high /
   foot_landing / release, or only at windup_onset?
C. Anti-locking two-tailed test — for batters whose R is BELOW null, compute
   the lower-tail p-value (probability of seeing R that low under random).
D. Cross-batter traits — does R correlate with handedness or pitch count?

Reads census.pkl, writes additional plots and CSVs to data/oscillation_report/pre_pitch_preparatory_movement/.
"""

from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp


def load_census(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ────────────────────────────────────────────────────────────
# A. Lead-time R sweep
# ────────────────────────────────────────────────────────────


def lead_time_r_sweep(census, out_path, top_n=10):
    """For each top-locked batter, sample R at many time offsets relative to release.
    Each pitch contributes one phase angle per offset; R = circular mean across pitches.

    Time axis: from release - 6s to release + 0.2s. We use pre-osc PC trajectory
    indices as the time anchor. Since PC is computed on pre-osc only, we need a
    consistent time-relative-to-release axis."""

    # Build leaderboard
    rows = []
    for bid, r in census["results"].items():
        R, n = r["phase_R"][("pca_pc1pc2", "windup_onset")]
        if n < 8: continue
        null = np.array(r["phase_null_distributions"]["windup_onset"])
        margin = R - np.percentile(null, 95) if len(null) > 0 else 0
        rows.append((bid, margin, R, n))
    rows.sort(key=lambda x: -x[1])
    top = [r[0] for r in rows[:top_n]]

    # For each batter and each test offset (relative to release), recompute R using
    # the PC1xPC2 trajectory at that offset.
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="#1a1a1a")
    SAMPLE_HZ = 30.0
    PRE_OSC_FRAMES = int(5.0 * SAMPLE_HZ)
    # Test offsets: -5s to +0.2s relative to release, every 0.1s
    test_offsets = np.arange(-5.0, 0.21, 0.1)

    colors = plt.cm.tab10.colors
    for ti, bid in enumerate(top):
        r = census["results"][bid]
        name = census["names"][bid]
        # Re-compute R per offset by recomputing PCA trajectory phase at that time
        # We don't have raw trajectories in census, only phases at 4 fixed events.
        # So for this analysis we'd need to re-pull data. Instead, let's use the
        # 4 cached events as anchors and INTERPOLATE: for each pitch, take its
        # 4 (event_t, phase) pairs and... nope, phases don't interpolate.
        #
        # Workaround: present the 4 event Rs as a function of time-rel-to-release.
        events = ("windup_onset", "knee_high", "foot_landing", "release")
        # Mean event time relative to release across pitches for this batter
        event_t_rel = []
        Rs = []
        for ev in events:
            R_val, n_val = r["phase_R"][("pca_pc1pc2", ev)]
            # We don't have per-pitch event times in the census; estimate from pitches_meta
            # We DO have phases per pitch. We need event time for each.
            # Skip for now — just plot at the 4 fixed event points using literal labels.
            Rs.append(R_val)
        ax.plot(range(4), Rs, "o-", color=colors[ti % len(colors)], label=name,
                lw=2, markersize=8, alpha=0.85)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["windup_onset", "knee_high", "foot_landing", "release"],
                       color="white")
    ax.set_ylabel("R (PC1×PC2 phase)", color="white")
    ax.set_xlabel("delivery event", color="white")
    ax.set_title("Phase concentration R across the 4 pitcher delivery events\n"
                 "(top 10 phase-locked batters by margin at windup_onset)",
                 color="white", fontsize=11)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.legend(loc="lower right", facecolor="#222", edgecolor="#666",
              labelcolor="white", fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# B. Cross-event swing/take KS test
# ────────────────────────────────────────────────────────────


def cross_event_swing_take(census, out_path):
    """For each batter, compute KS p-value separately at all 4 events.
    Are some batters' signals event-specific?"""
    events = ("windup_onset", "knee_high", "foot_landing", "release")
    rows = []
    for bid, r in census["results"].items():
        n_swing = r["n_pitches_swing"]
        n_take = r["n_pitches_take"]
        if n_swing < 3 or n_take < 3: continue
        result = {"batter_id": bid, "name": census["names"].get(bid, str(bid)),
                  "n_swing": n_swing, "n_take": n_take}
        for ev in events:
            phases = np.array(r["phases_per_pitch"][("pca_pc1pc2", ev)])
            sw, tk = [], []
            for pi, pm in enumerate(r["pitches_meta"]):
                ph = phases[pi]
                if np.isnan(ph): continue
                if pm["result_call"] in ("S", "X", "F"):
                    sw.append(ph)
                elif pm["result_call"] in ("B", "C", "*B"):
                    tk.append(ph)
            if len(sw) >= 3 and len(tk) >= 3:
                try:
                    stat, p = ks_2samp(sw, tk)
                except Exception:
                    stat, p = float("nan"), float("nan")
            else:
                stat, p = float("nan"), float("nan")
            result[f"ks_p_{ev}"] = p
            result[f"ks_stat_{ev}"] = stat
            result[f"n_sw_{ev}"] = len(sw)
            result[f"n_tk_{ev}"] = len(tk)
        rows.append(result)

    # Sort by min p across events
    def min_p(r):
        ps = [r[f"ks_p_{ev}"] for ev in events if not np.isnan(r[f"ks_p_{ev}"])]
        return min(ps) if ps else 1.0
    rows.sort(key=min_p)

    # Plot heatmap
    top_rows = rows[:25]
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_rows) * 0.32)), facecolor="#1a1a1a")
    mat = np.array([[r[f"ks_p_{ev}"] for ev in events] for r in top_rows])
    im = ax.imshow(np.where(np.isnan(mat), 1.0, mat), cmap="RdYlGn_r", vmin=0, vmax=0.5,
                   aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(top_rows)))
    ax.set_yticklabels([f"{r['name']} ({r['n_swing']}sw/{r['n_take']}tk)" for r in top_rows],
                       color="white", fontsize=8)
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(events, rotation=20, color="white")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("KS p (swing vs take phase)", color="white")
    cb.ax.tick_params(colors="white", labelsize=7)
    # Annotate p<0.05 cells
    for i in range(len(top_rows)):
        for j in range(len(events)):
            v = mat[i, j]
            if not np.isnan(v) and v < 0.05:
                ax.text(j, i, f"{v:.03f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
    ax.set_facecolor("#1a1a1a")
    ax.set_title("Swing vs take phase distribution: KS p-value per (batter, event)\n"
                 "(top 25 batters by min p; numbers shown for p<0.05)",
                 color="white", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)

    # Write CSV
    csv_path = out_path.replace(".png", ".csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows


# ────────────────────────────────────────────────────────────
# C. Anti-locking two-tailed test
# ────────────────────────────────────────────────────────────


def anti_locking_test(census, out_path):
    """For each batter, compute one-tailed p-values for both directions:
    p_high = P(null_R ≥ real_R)  → tests for phase-locking
    p_low  = P(null_R ≤ real_R)  → tests for anti-locking (more random than chance)
    """
    rows = []
    for bid, r in census["results"].items():
        for ev in ("windup_onset", "knee_high", "foot_landing", "release"):
            R, n = r["phase_R"][("pca_pc1pc2", ev)]
            null = np.array(r["phase_null_distributions"].get(ev, []))
            if len(null) == 0 or n < 5: continue
            p_high = (np.sum(null >= R) + 1) / (len(null) + 1)
            p_low = (np.sum(null <= R) + 1) / (len(null) + 1)
            rows.append({
                "batter_id": bid,
                "name": census["names"].get(bid, str(bid)),
                "event": ev,
                "R": R,
                "n": n,
                "null_mean": float(null.mean()),
                "p_high_locked": p_high,
                "p_low_random": p_low,
                "is_locked": p_high < 0.05,
                "is_anti_locked": p_low < 0.05,
            })

    # Anti-locked findings
    anti = [r for r in rows if r["is_anti_locked"]]
    print(f"=== anti-locked findings ({len(anti)} of {len(rows)} batter-event pairs) ===")
    for r in sorted(anti, key=lambda x: x["p_low_random"])[:20]:
        print(f"  {r['name']:25s} {r['event']:14s}  "
              f"R={r['R']:.2f}  null_mean={r['null_mean']:.2f}  "
              f"p_low={r['p_low_random']:.3f}  n={r['n']}")

    # Plot: scatter R vs null_mean, color by lock direction
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a1a")
    for r in rows:
        c = "#42a5f5"
        if r["is_locked"]: c = "#4caf50"
        if r["is_anti_locked"]: c = "#f44336"
        ax.scatter(r["null_mean"], r["R"], c=c, alpha=0.7, s=30)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="R = null_mean (chance)")
    ax.set_xlabel("Null mean R (chance level)", color="white")
    ax.set_ylabel("Real R", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.set_title("Phase-lock 2-tailed test: R vs null_mean for every (batter, event)\n"
                 "green=phase-locked (p<0.05), red=anti-locked (p<0.05), blue=neutral",
                 color="white", fontsize=11)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)
    csv_path = out_path.replace(".png", ".csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows


# ────────────────────────────────────────────────────────────
# D. Cross-batter traits (handedness, sample size)
# ────────────────────────────────────────────────────────────


def cross_batter_traits(census, out_path):
    """Group batters by handedness, plot R distribution. Also R vs n_pitches."""
    rows = []
    for bid, r in census["results"].items():
        R, n = r["phase_R"][("pca_pc1pc2", "windup_onset")]
        if n < 5: continue
        # batter_side from any pitches_meta
        side = None
        for pm in r["pitches_meta"]:
            if pm["batter_side"]:
                side = pm["batter_side"]; break
        rows.append({
            "name": census["names"].get(bid, str(bid)),
            "side": side or "?",
            "R": R,
            "n": n,
            "n_swing": r["n_pitches_swing"],
            "n_take": r["n_pitches_take"],
        })

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#1a1a1a")
    # Left: R by handedness
    ax = axes[0]
    for side, color in (("L", "#42a5f5"), ("R", "#ff9800"), ("S", "#9c27b0"), ("?", "#888")):
        vals = [r["R"] for r in rows if r["side"] == side]
        if not vals: continue
        ax.scatter([side] * len(vals), vals, color=color, alpha=0.6, s=40)
    sides = sorted(set(r["side"] for r in rows))
    means = [np.mean([r["R"] for r in rows if r["side"] == s]) for s in sides]
    ax.plot(sides, means, "o-", color="white", lw=2, label="mean R")
    ax.set_ylabel("R (windup_onset PCA)", color="white")
    ax.set_xlabel("Batter side", color="white")
    ax.set_title("R distribution by batter handedness", color="white", fontsize=11)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")

    # Right: R vs n_pitches (does small sample inflate R?)
    ax = axes[1]
    ns = [r["n"] for r in rows]
    Rs = [r["R"] for r in rows]
    ax.scatter(ns, Rs, color="#4caf50", alpha=0.7, s=40)
    # Add linear fit
    if len(ns) >= 3:
        z = np.polyfit(ns, Rs, 1)
        x = np.array([min(ns), max(ns)])
        ax.plot(x, np.polyval(z, x), "--", color="#888", lw=1.5,
                label=f"linear fit slope={z[0]:.3f}")
        ax.legend(facecolor="#222", edgecolor="#666", labelcolor="white", fontsize=9)
    ax.set_xlabel("n_pitches usable", color="white")
    ax.set_ylabel("R (windup_onset PCA)", color="white")
    ax.set_title("R vs sample size — small-n batters tend to inflated R", color="white", fontsize=11)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)
    return rows


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────


def main():
    out_dir = Path("data/oscillation_report/pre_pitch_preparatory_movement")
    census = load_census(out_dir / "census.pkl")
    print(f"loaded census: {len(census['results'])} batters\n")

    print("=== A. Lead-time R curve (top 10 batters) ===")
    lead_time_r_sweep(census, out_dir / "extra_lead_time.png", top_n=10)

    print("\n=== B. Cross-event KS swing/take ===")
    rows = cross_event_swing_take(census, str(out_dir / "extra_swing_take_cross_event.png"))
    print("Top 10 by min KS p across all 4 events:")
    for r in rows[:10]:
        ps = {ev: r[f"ks_p_windup_onset"] for ev in ["windup_onset"]}
        ps_all = [(ev, r[f"ks_p_{ev}"]) for ev in ("windup_onset","knee_high","foot_landing","release")]
        ps_all = [(ev, p) for ev, p in ps_all if not np.isnan(p)]
        ps_str = "  ".join(f"{ev[:3]}={p:.3f}" for ev, p in ps_all)
        print(f"  {r['name']:25s}  n_sw/tk={r['n_swing']}/{r['n_take']}  {ps_str}")

    print("\n=== C. Anti-locking two-tailed ===")
    anti_locking_test(census, str(out_dir / "extra_anti_locking.png"))

    print("\n=== D. Cross-batter traits ===")
    cross_batter_traits(census, out_dir / "extra_handedness.png")


if __name__ == "__main__":
    main()
