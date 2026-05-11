"""Build the overnight oscillation report from census.pkl.

Generates:
    - leaderboard table (markdown + CSV)
    - leaderboard plots (R, R-margin, n-pitches, freqs)
    - per-batter fingerprint plots (top 10 + bottom 10)
    - PC mode-shape interpretation
    - pitcher-effect analysis (does locking depend on which pitcher?)
    - pitch-type discrimination (does phase differ by FF vs breaking?)
    - outcome correlation (does phase differ by swing vs take?)
    - final markdown report

Usage:
    python scripts/build_pre_pitch_preparatory_movement.py --census data/oscillation_report/pre_pitch_preparatory_movement/census.pkl
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import circmean, ks_2samp, kruskal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.storage import JOINT_COLS

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
JOINT_NAMES = {bid: name for bid, name in JOINT_COLS}


def percentile(arr, p):
    return float(np.percentile(arr, p)) if len(arr) > 0 else float("nan")


def p_value(real_R, null_arr):
    if len(null_arr) == 0: return float("nan")
    return (np.sum(np.array(null_arr) >= real_R) + 1) / (len(null_arr) + 1)


# ────────────────────────────────────────────────────────────
# Leaderboard
# ────────────────────────────────────────────────────────────


def build_leaderboard(census, event="windup_onset", method="jpca"):
    rows = []
    for bid, r in census["results"].items():
        R, n = r["phase_R"][(method, event)]
        if n < 5: continue
        null = r["phase_null_distributions"].get(event, [])
        n95 = percentile(null, 95)
        margin = R - n95
        p = p_value(R, null)
        rows.append({
            "batter_id": bid,
            "name": census["names"].get(bid, str(bid)),
            "R": R,
            "n_phase": n,
            "n_pitches_usable": r["n_pitches_usable"],
            "n_pitches_total": r["n_pitches_total"],
            "null95": n95,
            "margin": margin,
            "p_value": p,
            "n_swing": r["n_pitches_swing"],
            "n_take": r["n_pitches_take"],
        })
    rows.sort(key=lambda x: -x["margin"])
    return rows


def write_leaderboard_csv(rows, out_path):
    if not rows: return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_leaderboard(rows, out_path, event_label="windup_onset"):
    """Horizontal bar chart of R-vs-null95 for all batters, sorted by margin."""
    fig, ax = plt.subplots(figsize=(11, max(8, len(rows) * 0.22)), facecolor="#1a1a1a")
    names = [r["name"] for r in rows]
    Rs = [r["R"] for r in rows]
    n95s = [r["null95"] for r in rows]
    pvs = [r["p_value"] for r in rows]
    y = np.arange(len(rows))

    # Plot null95 as gray bars (background)
    ax.barh(y, n95s, color="#444", alpha=0.6, label="null 95th percentile (chance level)", height=0.7)
    # Plot real R as colored bars (red if significant, else cyan)
    colors = ["#f44336" if p < 0.05 else "#42a5f5" for p in pvs]
    ax.barh(y, Rs, color=colors, alpha=0.85, height=0.55, label="real R")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name']} (n={r['n_phase']:2d})" for r in rows], color="white", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Phase concentration R   (0=random, 1=perfectly locked)", color="white")
    ax.set_title(f"Batter pre-pitch oscillation phase-lock leaderboard at {event_label}\n"
                 f"red = significant (p<0.05), blue = not significant   (gray bar = null 95%ile)",
                 color="white", fontsize=11)
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#666")
    ax.legend(loc="lower right", facecolor="#222", edgecolor="#666", labelcolor="white", fontsize=9)
    ax.axvline(1.0, color="#666", lw=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Per-event leaderboard heatmap (4 events × all batters)
# ────────────────────────────────────────────────────────────


def plot_event_heatmap(census, out_path, method="jpca"):
    events = ["windup_onset", "knee_high", "foot_landing", "release"]
    bids = sorted(census["results"].keys(),
                  key=lambda b: -(census["results"][b]["phase_R"][(method, "windup_onset")][0] or 0))
    n_b = len(bids)
    mat = np.full((n_b, len(events)), np.nan)
    margins = np.full((n_b, len(events)), np.nan)
    for i, bid in enumerate(bids):
        r = census["results"][bid]
        for j, ev in enumerate(events):
            R, n = r["phase_R"][(method, ev)]
            mat[i, j] = R
            null = np.array(r["phase_null_distributions"].get(ev, []))
            if len(null) > 0:
                margins[i, j] = R - np.percentile(null, 95)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(8, n_b * 0.22)), facecolor="#1a1a1a")
    im1 = ax1.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax1.set_xticks(range(len(events)))
    ax1.set_xticklabels(events, rotation=20, color="white", fontsize=9)
    ax1.set_yticks(range(n_b))
    ax1.set_yticklabels([census["names"].get(b, str(b)) for b in bids], color="white", fontsize=7)
    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.set_label("R (phase concentration)", color="white")
    cb1.ax.tick_params(colors="white", labelsize=7)
    ax1.set_facecolor("#1a1a1a"); ax1.set_title("R per batter × event", color="white")

    im2 = ax2.imshow(margins, aspect="auto", cmap="RdBu", vmin=-0.4, vmax=0.4, interpolation="nearest")
    ax2.set_xticks(range(len(events)))
    ax2.set_xticklabels(events, rotation=20, color="white", fontsize=9)
    ax2.set_yticks([])
    cb2 = fig.colorbar(im2, ax=ax2)
    cb2.set_label("Margin: R − null95   (red = real lock, blue = below chance)", color="white")
    cb2.ax.tick_params(colors="white", labelsize=7)
    ax2.set_facecolor("#1a1a1a"); ax2.set_title("Margin (real − chance)", color="white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Per-batter fingerprint
# ────────────────────────────────────────────────────────────


def plot_per_batter_fingerprint(bid, r, name, out_path):
    """4-panel: joint-stddev heat (per-pitch × joint), per-pitch dominant freqs,
    rose plots at 4 events, R-vs-null bar."""
    fig = plt.figure(figsize=(15, 10), facecolor="#1a1a1a")
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.45,
                          left=0.06, right=0.97, top=0.92, bottom=0.06)

    # Top-left: per-joint mean stddev (sorted)
    ax1 = fig.add_subplot(gs[0, :2])
    ranked = sorted(zip(JOINT_BIDS, r["joint_stds_mean"]), key=lambda x: -x[1])
    names_j = [JOINT_NAMES[b] for b, _ in ranked[:15]]
    vals = [v for _, v in ranked[:15]]
    ax1.barh(range(len(names_j)), vals, color="#4caf50", alpha=0.85)
    ax1.set_yticks(range(len(names_j)))
    ax1.set_yticklabels(names_j, color="white", fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("3D stddev (ft) — pre-osc window mean", color="white", fontsize=8)
    ax1.set_facecolor("#1a1a1a"); ax1.tick_params(colors="white", labelsize=8)
    for s in ax1.spines.values(): s.set_color("#666")
    ax1.set_title("Top 15 most-active joints", color="white", fontsize=10)

    # Top-right: per-pitch dominant frequencies (3 signals)
    ax2 = fig.add_subplot(gs[0, 2:])
    feats = [("pelvis_x", r["pelvis_x_freqs"], "#42a5f5"),
             ("hand_rt_y", r["hand_rt_y_freqs"], "#ff9800"),
             ("bat_axis_angle", r["bat_angle_freqs"], "#d4a04c")]
    for label, vals, color in feats:
        v = vals[~np.isnan(vals)]
        if len(v) == 0: continue
        ax2.scatter(np.full(len(v), label), v, color=color, alpha=0.6, s=20)
        ax2.plot([label, label], [v.mean(), v.mean()], color="white", lw=2)
    ax2.set_ylabel("Dominant freq (Hz)", color="white"); ax2.set_ylim(0, 4)
    ax2.set_facecolor("#1a1a1a"); ax2.tick_params(colors="white", labelsize=8)
    for s in ax2.spines.values(): s.set_color("#666")
    ax2.set_title("Per-pitch dominant frequency by signal", color="white", fontsize=10)

    # Middle row: rose plots at each event
    events = [("windup_onset", "#4caf50"), ("knee_high", "#ff9800"),
              ("foot_landing", "#e91e63"), ("release", "#9c27b0")]
    for ei, (ev, color) in enumerate(events):
        ax = fig.add_subplot(gs[1, ei], projection="polar")
        phases = np.array(r["phases_per_pitch"][("jpca", ev)])
        phases = phases[~np.isnan(phases)]
        if len(phases) > 0:
            bins = np.linspace(-np.pi, np.pi, 13)
            counts, _ = np.histogram(phases, bins=bins)
            widths = np.diff(bins)
            centers = bins[:-1] + widths / 2
            ax.bar(centers, counts, width=widths, bottom=0, color=color, alpha=0.75,
                   edgecolor="white", lw=0.4)
            R_val, n_val = r["phase_R"][("jpca", ev)]
            null = np.array(r["phase_null_distributions"].get(ev, []))
            n95 = percentile(null, 95)
            p = p_value(R_val, null)
            sig = "*" if p < 0.05 else ""
            ax.set_title(f"{ev}\nR={R_val:.2f} {sig} n={n_val} (95%null={n95:.2f})",
                         color="white", fontsize=8, pad=18)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white", labelsize=6)

    # Bottom-left: R vs null95 across all 4 events × 3 methods
    ax_r = fig.add_subplot(gs[2, :2])
    methods = ["pelvis_x", "bat_axis_angle", "pca_pc1pc2", "jpca"]
    method_colors = ["#42a5f5", "#d4a04c", "#4caf50", "#e91e63"]
    x = np.arange(4)
    width = 0.20
    for mi, m in enumerate(methods):
        Rs = []
        for ev, _ in events:
            rec = r["phase_R"].get((m, ev), (np.nan, 0))
            Rs.append(rec[0])
        ax_r.bar(x + (mi - 1.5) * width, Rs, width=width * 0.9,
                 color=method_colors[mi], label=m, alpha=0.85)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([e for e, _ in events], color="white", fontsize=8)
    ax_r.set_ylabel("R", color="white")
    ax_r.set_ylim(0, 1.05)
    ax_r.legend(loc="upper right", fontsize=7, facecolor="#222",
                edgecolor="#666", labelcolor="white")
    ax_r.set_facecolor("#1a1a1a"); ax_r.tick_params(colors="white", labelsize=7)
    for s in ax_r.spines.values(): s.set_color("#666")
    ax_r.set_title("R by phase coordinate × event", color="white", fontsize=10)

    # Bottom-right: PCA variance explained
    ax_pc = fig.add_subplot(gs[2, 2:])
    ve = r["pca_var_explained"][:8] * 100
    ax_pc.bar(range(1, len(ve) + 1), ve, color="#9c27b0", alpha=0.85)
    ax_pc.set_xlabel("PC#", color="white"); ax_pc.set_ylabel("% variance", color="white")
    ax_pc.set_facecolor("#1a1a1a"); ax_pc.tick_params(colors="white")
    for s in ax_pc.spines.values(): s.set_color("#666")
    cum = np.cumsum(ve)
    ax_pc.set_title(f"PCA variance (top 3 = {cum[2]:.0f}%)", color="white", fontsize=10)

    fig.suptitle(f"{name}  (id={bid})  |  pitches: {r['n_pitches_usable']} usable / {r['n_pitches_total']} total  |  "
                 f"swing/take = {r['n_pitches_swing']}/{r['n_pitches_take']}",
                 color="white", fontsize=12, y=0.98)
    fig.savefig(out_path, dpi=100, facecolor="#1a1a1a")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# PC mode-shape interpretation
# ────────────────────────────────────────────────────────────


def plot_pc_mode_shapes(census, out_path, n_top_batters=6):
    """For top-locked batters, show what physical motion PC1, PC2 correspond to.
    For each batter, plot loading magnitude per joint (3D stddev of component vector)."""
    leaderboard = build_leaderboard(census)
    top = [r["batter_id"] for r in leaderboard[:n_top_batters]]
    fig, axes = plt.subplots(n_top_batters, 3, figsize=(14, 3 * n_top_batters), facecolor="#1a1a1a")
    if n_top_batters == 1: axes = [axes]
    for ri, bid in enumerate(top):
        r = census["results"][bid]
        name = census["names"][bid]
        comps = r["pca_components"][:3]  # (3, 66) — 60 joint dims + 6 bat dims
        n_joints = len(JOINT_BIDS)
        labels = [JOINT_NAMES[b] for b in JOINT_BIDS] + ["bat_head", "bat_handle"]
        for ci in range(3):
            comp = comps[ci]
            joint_part = comp[:n_joints * 3].reshape(n_joints, 3)
            bat_part = comp[n_joints * 3:].reshape(2, 3) if len(comp) > n_joints * 3 else np.zeros((2, 3))
            mags = np.concatenate([np.linalg.norm(joint_part, axis=1),
                                   np.linalg.norm(bat_part, axis=1)])
            ax = axes[ri][ci] if n_top_batters > 1 else axes[ci]
            colors = ["#42a5f5"] * n_joints + ["#d4a04c"] * 2
            ax.bar(range(len(mags)),
                   mags,
                   color=[["#42a5f5", "#ff9800", "#4caf50"][ci]] * n_joints +
                         ["#d4a04c"] * 2,
                   alpha=0.85)
            ax.set_xticks(range(len(mags)))
            ax.set_xticklabels(labels, rotation=70, fontsize=6, color="white")
            ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white", labelsize=6)
            for s in ax.spines.values(): s.set_color("#666")
            top_j = sorted(zip(labels, mags), key=lambda x: -x[1])[:3]
            top_str = ", ".join(b for b, _ in top_j)
            ax.set_title(f"PC{ci+1} ({r['pca_var_explained'][ci]*100:.0f}%) — top: {top_str}",
                         color="white", fontsize=8)
        # row label
        if n_top_batters > 1:
            axes[ri][0].set_ylabel(f"{name}", color="white", fontsize=10, fontweight="bold")
    fig.suptitle("PC mode shapes — joint-loading magnitude per principal component\n"
                 "(rows = top phase-locked batters; columns = PC1/2/3)",
                 color="white", fontsize=12, y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, facecolor="#1a1a1a")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Pitcher-effect: does R differ when batter faces different pitchers?
# ────────────────────────────────────────────────────────────


def pitcher_effect_analysis(census, out_path, min_per_pitcher=3):
    """For each batter who faced multiple pitchers ≥3 pitches each, compute R per pitcher.
    Test if some pitchers induce more locking than others."""
    rows = []
    for bid, r in census["results"].items():
        # Group pitches by pitcher
        by_pitcher = defaultdict(list)
        for i, pm in enumerate(r["pitches_meta"]):
            by_pitcher[pm["pitcher_id"]].append(i)
        for pitcher_id, idxs in by_pitcher.items():
            if len(idxs) < min_per_pitcher: continue
            phases = np.array(r["phases_per_pitch"][("jpca", "windup_onset")])[idxs]
            phases = phases[~np.isnan(phases)]
            if len(phases) < min_per_pitcher: continue
            R = float(np.abs(np.mean(np.exp(1j * phases))))
            rows.append({
                "batter_id": bid,
                "batter_name": census["names"].get(bid, str(bid)),
                "pitcher_id": pitcher_id,
                "n_pitches": len(idxs),
                "R": R,
            })
    if not rows:
        return None

    # Plot: scatter of R vs pitcher_id grouped by batter (so we can see if same batter has different R per pitcher)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#1a1a1a")
    by_batter = defaultdict(list)
    for r in rows:
        by_batter[r["batter_name"]].append((r["pitcher_id"], r["R"], r["n_pitches"]))
    multi_pitcher_batters = [(b, p) for b, p in by_batter.items() if len(p) >= 2]
    multi_pitcher_batters.sort(key=lambda x: -np.std([r for _, r, _ in x[1]]))
    for i, (name, ps) in enumerate(multi_pitcher_batters[:15]):
        ys = [r for _, r, _ in ps]
        ax.scatter([i] * len(ys), ys, s=[n * 8 for _, _, n in ps], alpha=0.7,
                   color="#42a5f5")
        ax.plot([i, i], [min(ys), max(ys)], color="#666", lw=1)
    ax.set_xticks(range(len(multi_pitcher_batters[:15])))
    ax.set_xticklabels([b for b, _ in multi_pitcher_batters[:15]], rotation=45, ha="right",
                       color="white", fontsize=9)
    ax.set_ylabel("R (PCA phase at windup_onset, per pitcher)", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_title("Within-batter variation across pitchers (top 15 by R-spread)\n"
                 "Each point = one (batter, pitcher) pair, ≥3 pitches; size ∝ n_pitches",
                 color="white", fontsize=11)
    for s in ax.spines.values(): s.set_color("#666")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)
    return rows


# ────────────────────────────────────────────────────────────
# Pitch-type discrimination: phase at windup_onset by pitch_type
# ────────────────────────────────────────────────────────────


def pitch_type_discrimination(census, out_path):
    """For each batter, are phases at windup_onset different across pitch types?
    Pitch type is decided AT release, but if the batter reads pitcher's body language
    during the windup, his oscillation phase might already correlate with the upcoming pitch type."""
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), facecolor="#1a1a1a",
                             subplot_kw=dict(projection="polar"))
    leaderboard = build_leaderboard(census)
    top15 = [r["batter_id"] for r in leaderboard[:15]]
    for i, bid in enumerate(top15):
        ax = axes.flat[i]
        r = census["results"][bid]
        name = census["names"][bid]
        # Group phases by pitch_type
        by_type = defaultdict(list)
        phases = r["phases_per_pitch"][("jpca", "windup_onset")]
        for pi, pm in enumerate(r["pitches_meta"]):
            ph = phases[pi]
            if np.isnan(ph): continue
            by_type[pm["pitch_type"] or "?"].append(ph)
        # Plot up to 4 types as separate colored layers
        types_sorted = sorted(by_type.items(), key=lambda x: -len(x[1]))[:4]
        colors = ["#42a5f5", "#ff9800", "#e91e63", "#4caf50"]
        for ti, (ptype, ph_list) in enumerate(types_sorted):
            if len(ph_list) < 2: continue
            arr = np.array(ph_list)
            R_val = float(np.abs(np.mean(np.exp(1j * arr))))
            ax.scatter(arr, np.full(len(arr), 1.0 + ti * 0.15),
                       s=18, color=colors[ti], alpha=0.7, label=f"{ptype} n={len(arr)} R={R_val:.2f}")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white", labelsize=6)
        ax.set_yticks([])
        ax.set_title(name, color="white", fontsize=8, pad=14)
        ax.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, -0.25),
                  facecolor="#222", edgecolor="#666", labelcolor="white")
    fig.suptitle("Phase at windup_onset, colored by upcoming pitch type — top 15 phase-locked batters\n"
                 "If a batter reads pitch type from the windup, phases for different types should cluster differently",
                 color="white", fontsize=11, y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, facecolor="#1a1a1a")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Outcome correlation: phase × swing/take
# ────────────────────────────────────────────────────────────


def swing_take_phase_test(census, out_path):
    """Aggregate test: does the phase distribution at windup_onset differ between
    pitches that the batter swings at vs takes? Use Watson U² (or KS as approx)."""
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), facecolor="#1a1a1a",
                             subplot_kw=dict(projection="polar"))
    leaderboard = build_leaderboard(census)
    top15 = [r["batter_id"] for r in leaderboard[:15]]
    rows = []
    for i, bid in enumerate(top15):
        ax = axes.flat[i]
        r = census["results"][bid]
        name = census["names"][bid]
        phases = r["phases_per_pitch"][("jpca", "windup_onset")]
        swing_phases = []
        take_phases = []
        for pi, pm in enumerate(r["pitches_meta"]):
            ph = phases[pi]
            if np.isnan(ph): continue
            if pm["result_call"] in ("S", "X", "F"):
                swing_phases.append(ph)
            elif pm["result_call"] in ("B", "C", "*B"):
                take_phases.append(ph)
        for arr, color, lbl in [(swing_phases, "#f44336", "swing"),
                                 (take_phases, "#42a5f5", "take")]:
            if len(arr) >= 2:
                ax.scatter(arr, np.full(len(arr), 1.0),
                           s=22, color=color, alpha=0.7, label=f"{lbl} n={len(arr)}")
        # Compute KS statistic on circular as a proxy
        if len(swing_phases) >= 3 and len(take_phases) >= 3:
            try:
                stat, p = ks_2samp(swing_phases, take_phases)
            except Exception:
                stat, p = float("nan"), float("nan")
        else:
            stat, p = float("nan"), float("nan")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white", labelsize=6)
        ax.set_yticks([])
        ax.set_title(f"{name}\nKS p={p:.2f}" if not np.isnan(p) else name,
                     color="white", fontsize=8, pad=14)
        ax.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, -0.25),
                  facecolor="#222", edgecolor="#666", labelcolor="white")
        rows.append({"batter_id": bid, "name": name,
                     "n_swing": len(swing_phases), "n_take": len(take_phases),
                     "ks_stat": stat, "ks_p": p})
    fig.suptitle("Phase at windup_onset for swings (red) vs takes (blue) — top 15 phase-locked batters\n"
                 "If different distributions → batter's pre-pitch state predicts swing/take decision",
                 color="white", fontsize=11, y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, facecolor="#1a1a1a")
    plt.close(fig)
    return rows


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="data/oscillation_report/pre_pitch_preparatory_movement/census.pkl")
    ap.add_argument("--out-dir", default="data/oscillation_report/pre_pitch_preparatory_movement")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.census, "rb") as f:
        census = pickle.load(f)
    print(f"loaded census with {len(census['results'])} batters")

    # 1. Leaderboard for windup_onset
    print("\n=== building leaderboards ===")
    for ev in ("windup_onset", "knee_high", "foot_landing", "release"):
        rows = build_leaderboard(census, event=ev)
        write_leaderboard_csv(rows, out_dir / f"leaderboard_{ev}.csv")
        plot_leaderboard(rows, out_dir / f"leaderboard_{ev}.png", event_label=ev)
        print(f"  {ev}: {len(rows)} batters; significant (p<0.05): "
              f"{sum(1 for r in rows if r['p_value'] < 0.05)}")

    # 2. Per-event heatmap
    print("\n=== event heatmap ===")
    plot_event_heatmap(census, out_dir / "event_heatmap.png")

    # 3. Per-batter fingerprints (top 10 + bottom 10 + Julio for sanity)
    print("\n=== per-batter fingerprints ===")
    leaderboard = build_leaderboard(census)
    top10_bids = [r["batter_id"] for r in leaderboard[:10]]
    bottom10_bids = [r["batter_id"] for r in leaderboard[-10:]]
    # Julio Rodríguez (RHB) — switch hitters not applicable; lookup by name
    julio_keys = [k for k, n in census["names"].items() if "Julio Rodríguez" in n]
    for bid in set(top10_bids + bottom10_bids + julio_keys):
        if bid not in census["results"]: continue
        r = census["results"][bid]
        name = census["names"][bid]
        # safe filename
        safe = "".join(c for c in name.replace(" ", "_") if c.isalnum() or c in "._-")[:30]
        plot_per_batter_fingerprint(bid, r, name, out_dir / f"fingerprint_{safe}_{bid}.png")
    print(f"  rendered fingerprints for top 10 + bottom 10 + Julio")

    # 4. PC mode shapes
    print("\n=== PC mode-shape interpretation ===")
    plot_pc_mode_shapes(census, out_dir / "pc_mode_shapes.png", n_top_batters=6)

    # 5. Pitcher effect
    print("\n=== pitcher effect ===")
    pe_rows = pitcher_effect_analysis(census, out_dir / "pitcher_effect.png")

    # 6. Pitch-type discrimination
    print("\n=== pitch-type discrimination ===")
    pitch_type_discrimination(census, out_dir / "pitch_type_discrimination.png")

    # 7. Swing/take phase test
    print("\n=== swing/take phase test ===")
    st_rows = swing_take_phase_test(census, out_dir / "swing_take_phase.png")
    for r in sorted(st_rows, key=lambda x: x["ks_p"] if not np.isnan(x["ks_p"]) else 1)[:5]:
        print(f"  {r['name']:25s}  swing/take = {r['n_swing']}/{r['n_take']}  KS p={r['ks_p']:.3f}")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
