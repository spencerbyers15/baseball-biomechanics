"""Phase-distribution-by-outcome analysis.

The user's hypothesis: a batter's pre-pitch fidget pattern biases his swing
outcome. So phase at windup_onset (a low-dim summary of his pre-pitch
postural state) should differ between:
    take (B/C)    — laid off
    in-play (X)   — good swing, made contact
    foul (F)      — partial contact (late adjust or fooled-with-a-piece)
    whiff (S)     — fooled, missed entirely

For each batter with enough pitches:
  1. Compute pre-pitch waggle PC1×PC2 phase at windup_onset (already cached in census)
  2. Bucket by result_call
  3. Pairwise KS tests on the 4 most interesting pairs:
        TAKE vs WHIFF     — "lay off vs fooled" — body state when fooled
        X    vs WHIFF     — "good contact vs fooled" — body state when connected
        X    vs FOUL      — solid contact vs partial contact
        SWING vs TAKE     — original signal
  4. Plot PC1×PC2 scatter per batter, colored by outcome, with phase markers

Outputs to data/oscillation_report/pre_pitch_preparatory_movement/outcome_split/
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp


OUTCOME_GROUPS = {
    "take":   ("B", "C", "*B"),       # laid off
    "inplay": ("X",),                  # connected, ball put in play
    "foul":   ("F",),                  # got a piece
    "whiff":  ("S",),                  # missed entirely
}
# "swing" is a synthetic pseudo-class combining all three swing-outcomes.
SWING_OUTCOMES = ("inplay", "foul", "whiff")
OUTCOME_COLORS = {
    "take":   "#42a5f5",  # blue
    "swing":  "#9c27b0",  # purple — committed
    "inplay": "#4caf50",  # green — connected
    "foul":   "#ff9800",  # orange — partial
    "whiff":  "#f44336",  # red — fooled
}

# Pairwise tests to run.
# Level 1: take vs swing (committed or not). Level 2: the 3 swing outcomes
# pairwise against each other (what kind of swing did they make).
PAIRS = [
    ("take",   "swing"),   # laid off vs committed (THE primary decision)
    ("inplay", "foul"),    # solid contact vs partial — quality of contact
    ("inplay", "whiff"),   # good swing vs fooled — connected vs missed
    ("foul",   "whiff"),   # partial contact vs total miss — got-a-piece vs nothing
]


def bucket(result_call):
    """Return the FINE-grained class (take/inplay/foul/whiff)."""
    for name, codes in OUTCOME_GROUPS.items():
        if result_call in codes: return name
    return None


def expand_buckets(by_outcome_fine):
    """Given a dict of fine-class -> list, also add the 'swing' pseudo-class
    that pools inplay+foul+whiff."""
    pooled = []
    for k in SWING_OUTCOMES:
        pooled.extend(by_outcome_fine.get(k, []))
    by_outcome_fine["swing"] = pooled
    return by_outcome_fine


def analyze(census, out_dir, min_per_class=3, method="pca_pc1pc2"):
    """Returns list of per-batter dicts with KS p-values for each pair.

    method: which phase signal to test — "pca_pc1pc2" (Hilbert phase of PC1
    plane) or "jpca" (angle within jPCA dominant rotation plane).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bid, r in census["results"].items():
        if (method, "windup_onset") not in r["phases_per_pitch"]: continue
        phases_we = np.array(r["phases_per_pitch"][(method, "windup_onset")])
        phases_kh = np.array(r["phases_per_pitch"][(method, "knee_high")])

        by_outcome_we = defaultdict(list)
        by_outcome_kh = defaultdict(list)
        for pi, pm in enumerate(r["pitches_meta"]):
            o = bucket(pm["result_call"])
            if o is None: continue
            if not np.isnan(phases_we[pi]):
                by_outcome_we[o].append(phases_we[pi])
            if not np.isnan(phases_kh[pi]):
                by_outcome_kh[o].append(phases_kh[pi])
        # Add pooled 'swing' pseudo-class to both
        expand_buckets(by_outcome_we)
        expand_buckets(by_outcome_kh)

        result = {
            "batter_id": bid,
            "name": census["names"].get(bid, str(bid)),
            "n_take": len(by_outcome_we["take"]),
            "n_swing": len(by_outcome_we["swing"]),
            "n_inplay": len(by_outcome_we["inplay"]),
            "n_foul": len(by_outcome_we["foul"]),
            "n_whiff": len(by_outcome_we["whiff"]),
        }
        for a, b in PAIRS:
            for evname, by_outcome in [("we", by_outcome_we), ("kh", by_outcome_kh)]:
                arr_a = by_outcome[a]; arr_b = by_outcome[b]
                if len(arr_a) >= min_per_class and len(arr_b) >= min_per_class:
                    try:
                        stat, p = ks_2samp(arr_a, arr_b)
                    except Exception:
                        stat, p = float("nan"), float("nan")
                else:
                    stat, p = float("nan"), float("nan")
                result[f"ks_{evname}_{a}_vs_{b}_p"] = float(p) if not np.isnan(p) else ""
                result[f"ks_{evname}_{a}_vs_{b}_n"] = f"{len(arr_a)}/{len(arr_b)}"
        rows.append(result)

    return rows


def plot_top_batter_phase_scatter(census, batter_ids, out_path, method="pca_pc1pc2"):
    """For each top batter, plot phase scatter at windup_onset colored by outcome.

    We don't have raw projection coordinates per pitch in census (only phase
    angle). Project the phase angle onto the unit circle and tag with outcome
    marker. Multiple concentric rings = multiple events.
    """
    n_b = len(batter_ids)
    if n_b == 0: return
    cols = 3
    rows = (n_b + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5),
                             subplot_kw=dict(projection="polar"),
                             facecolor="#1a1a1a")
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    axes = np.array(axes).reshape(rows, cols)
    for ai, bid in enumerate(batter_ids):
        ax = axes.flat[ai]
        r = census["results"][bid]
        name = census["names"][bid]
        phases = np.array(r["phases_per_pitch"][(method, "windup_onset")])

        # Layout: TAKE on inner ring; the 3 swing outcomes on outer rings.
        # This visually preserves "take vs swing" as inner-vs-outer separation,
        # and "swing-outcome vs swing-outcome" as different outer rings.
        for outcome in ("take", "inplay", "foul", "whiff"):
            arr = []
            for pi, pm in enumerate(r["pitches_meta"]):
                if bucket(pm["result_call"]) == outcome and not np.isnan(phases[pi]):
                    arr.append(phases[pi])
            if not arr: continue
            arr = np.array(arr)
            radius = {"take": 0.45, "inplay": 0.85, "foul": 1.05, "whiff": 1.25}[outcome]
            ax.scatter(arr, np.full(len(arr), radius),
                       c=OUTCOME_COLORS[outcome], s=35, alpha=0.7,
                       label=f"{outcome} n={len(arr)}", edgecolor="white", lw=0.4)
        # Dividing ring at r=0.65 between take and swings
        ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 0.65),
                color="#666", lw=0.5, alpha=0.5)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white", labelsize=6)
        ax.set_yticks([])
        ax.set_title(name, color="white", fontsize=9, pad=12)
        ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.3),
                  facecolor="#222", edgecolor="#666", labelcolor="white", ncol=2)
    for ai in range(len(batter_ids), rows * cols):
        axes.flat[ai].axis("off")
    fig.suptitle("Phase at windup_onset colored by outcome\n"
                 "INNER ring: take (blue, laid off)   |   OUTER rings: 3 swing outcomes — "
                 "inplay (green, connected), foul (orange, partial), whiff (red, missed)",
                 color="white", fontsize=11, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def plot_pairwise_p_heatmap(rows, out_path):
    """Heatmap: batters (rows) × pairwise tests (cols) of KS p-values."""
    # Filter to batters with at least one valid test
    pair_keys_we = [f"ks_we_{a}_vs_{b}_p" for a, b in PAIRS]
    pair_keys_kh = [f"ks_kh_{a}_vs_{b}_p" for a, b in PAIRS]
    valid = []
    for r in rows:
        has_any = any(r[k] != "" for k in pair_keys_we + pair_keys_kh)
        if has_any: valid.append(r)
    if not valid: return
    # Sort by min p across tests
    def min_p(r):
        ps = [float(r[k]) for k in pair_keys_we + pair_keys_kh if r[k] != ""]
        return min(ps) if ps else 1.0
    valid.sort(key=min_p)
    valid = valid[:25]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, len(valid) * 0.3)), facecolor="#1a1a1a")
    for ax_i, (event_label, pair_keys) in enumerate([("windup_onset", pair_keys_we),
                                                       ("knee_high", pair_keys_kh)]):
        ax = axes[ax_i]
        mat = np.full((len(valid), len(pair_keys)), np.nan)
        for i, r in enumerate(valid):
            for j, k in enumerate(pair_keys):
                v = r[k]
                if v != "": mat[i, j] = float(v)
        im = ax.imshow(np.where(np.isnan(mat), 1.0, mat),
                       aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5,
                       interpolation="nearest")
        ax.set_yticks(range(len(valid)))
        ax.set_yticklabels([f"{r['name']} (take/inp/F/whi={r['n_take']}/{r['n_inplay']}/{r['n_foul']}/{r['n_whiff']})"
                            for r in valid], color="white", fontsize=7)
        ax.set_xticks(range(len(pair_keys)))
        ax.set_xticklabels([f"{a}\nvs\n{b}" for (a, b) in PAIRS], color="white", fontsize=8)
        # Annotate p < 0.05
        for i in range(len(valid)):
            for j in range(len(pair_keys)):
                v = mat[i, j]
                if not np.isnan(v) and v < 0.05:
                    ax.text(j, i, f"{v:.03f}", ha="center", va="center",
                            color="white", fontsize=7, fontweight="bold")
        ax.set_facecolor("#1a1a1a")
        ax.set_title(f"KS p — phase at {event_label}", color="white", fontsize=11)
    cbar = fig.colorbar(im, ax=axes.tolist())
    cbar.set_label("KS p-value", color="white")
    cbar.ax.tick_params(colors="white", labelsize=7)
    fig.suptitle("Pairwise phase-distribution KS tests by outcome class (top 25 by min p)\n"
                 "values shown for p<0.05 only",
                 color="white", fontsize=11, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="data/oscillation_report/pre_pitch_preparatory_movement/census.pkl")
    ap.add_argument("--out-dir", default="data/oscillation_report/pre_pitch_preparatory_movement")
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--method", default="pca_pc1pc2",
                    choices=["pca_pc1pc2", "jpca", "bat_axis_angle", "pelvis_x"],
                    help="Phase signal to test")
    args = ap.parse_args()

    with open(args.census, "rb") as f:
        census = pickle.load(f)
    print(f"loaded census: {len(census['results'])} batters  ·  method={args.method}")

    rows = analyze(census, Path(args.out_dir), min_per_class=args.min_per_class,
                   method=args.method)
    # Save CSV
    method_suffix = f"_{args.method}"
    csv_path = Path(args.out_dir) / f"outcome_phase_pairwise{method_suffix}.csv"
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {csv_path}")

    def print_pair_leaderboard(pair_a, pair_b, evt="we", evt_label="windup_onset", top=12):
        key = f"ks_{evt}_{pair_a}_vs_{pair_b}_p"
        hits = [r for r in rows if r.get(key, "") != ""]
        if not hits:
            print(f"\n=== KS {pair_a} vs {pair_b} @ {evt_label} === (no batters with sufficient n)")
            return
        hits.sort(key=lambda x: float(x[key]))
        print(f"\n=== KS {pair_a} vs {pair_b} @ {evt_label} (sorted by p) ===")
        for r in hits[:top]:
            p = float(r[key])
            sig = " ★" if p < 0.05 else ""
            n_a = r[f"n_{pair_a}"]; n_b = r[f"n_{pair_b}"]
            print(f"  {r['name']:30s}  {pair_a}={n_a:>2d}  {pair_b}={n_b:>2d}  p={p:.3f}{sig}")

    for evt_key, evt_label in (("we", "windup_onset"), ("kh", "knee_high")):
        print(f"\n────────── {evt_label} ──────────")
        for a, b in PAIRS:
            print_pair_leaderboard(a, b, evt=evt_key, evt_label=evt_label, top=10)

    # Plot heatmap
    plot_pairwise_p_heatmap(rows, Path(args.out_dir) / f"outcome_phase_pairwise{method_suffix}.png")
    print(f"wrote {args.out_dir}/outcome_phase_pairwise{method_suffix}.png")

    # Per-batter scatter for top 12 by min p
    def min_p(r):
        ps = [float(r[k]) for k in (f"ks_we_{a}_vs_{b}_p" for a, b in PAIRS) if r[k] != ""]
        return min(ps) if ps else 1.0
    top = sorted(rows, key=min_p)[:12]
    plot_top_batter_phase_scatter(census, [r["batter_id"] for r in top],
                                   Path(args.out_dir) / f"outcome_phase_scatter_top12{method_suffix}.png",
                                   method=args.method)
    print(f"wrote {args.out_dir}/outcome_phase_scatter_top12{method_suffix}.png")


if __name__ == "__main__":
    main()
