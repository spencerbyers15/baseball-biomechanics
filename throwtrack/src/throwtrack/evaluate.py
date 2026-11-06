"""Score detected throws against ground truth.

The same code scores simulated sessions and, once labels exist, the club's
held-out videos. Ground-truth rows: {"thrower": id, "time_s": float,
"target": id | "wall", optional "distance_ft", "mph", "ball_type"}.
"""

from __future__ import annotations

import numpy as np

from .schema import ThrowEvent


def match(throws: list[ThrowEvent], truth: list[dict], tol_s: float = 0.25) -> list[tuple[int, int]]:
    """One-to-one (throw_idx, truth_idx) matches: same thrower, closest in time within tol_s."""
    cands = []
    for ti, t in enumerate(throws):
        for gi, g in enumerate(truth):
            if g["thrower"] == t.track_id:
                dt = abs(t.time_s - g["time_s"])
                if dt <= tol_s:
                    cands.append((dt, ti, gi))
    cands.sort()
    used_t, used_g, out = set(), set(), []
    for _, ti, gi in cands:
        if ti not in used_t and gi not in used_g:
            used_t.add(ti)
            used_g.add(gi)
            out.append((ti, gi))
    return out


def evaluate(throws: list[ThrowEvent], truth: list[dict], tol_s: float = 0.25) -> dict:
    pairs = match(throws, truth, tol_s)
    tp = len(pairs)
    res = {
        "n_truth": len(truth),
        "n_detected": len(throws),
        "precision": tp / len(throws) if throws else float("nan"),
        "recall": tp / len(truth) if truth else float("nan"),
    }
    timing = [throws[ti].time_s - truth[gi]["time_s"] for ti, gi in pairs]
    res["timing_sd_ms"] = 1000 * float(np.std(timing)) if timing else float("nan")

    target_ok = []
    for ti, gi in pairs:
        t, g = throws[ti], truth[gi]
        if g["target"] == "wall":
            target_ok.append(t.target_type == "wall")
        else:
            target_ok.append(t.target_type == "partner" and t.receiver_id == g["target"])
    res["target_accuracy"] = float(np.mean(target_ok)) if target_ok else float("nan")

    def mae(key_true, attr):
        errs = [abs(getattr(throws[ti], attr) - truth[gi][key_true]) for ti, gi in pairs
                if truth[gi].get(key_true) is not None and getattr(throws[ti], attr) is not None]
        return (float(np.mean(errs)) if errs else float("nan")), len(errs)

    res["distance_mae_ft"], res["n_distance"] = mae("distance_ft", "distance_ft")
    res["velo_mae_mph"], res["n_velo"] = mae("mph", "velo_mph")
    bt = [throws[ti].ball_type == truth[gi]["ball_type"] for ti, gi in pairs
          if truth[gi].get("ball_type") is not None and throws[ti].ball_type is not None]
    res["ball_type_accuracy"] = float(np.mean(bt)) if bt else float("nan")
    inside = [throws[ti].velo_lo_mph <= truth[gi]["mph"] <= throws[ti].velo_hi_mph
              for ti, gi in pairs if throws[ti].velo_lo_mph is not None and truth[gi].get("mph") is not None]
    if inside:
        res["velo_interval_coverage"] = float(np.mean(inside))
    return res
