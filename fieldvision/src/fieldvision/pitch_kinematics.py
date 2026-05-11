"""Derive pitcher delivery sub-events from skeletal trajectories.

The wire-format event labels (PITCHER_FIRST_MOVEMENT, BALL_WAS_RELEASED) are
unreliable for fine-grained kinematic timing — verified empirically that they
mark the wrong things for this dataset. Statsapi's pitch start_time is
trustworthy as physical release, but the windup sub-events (motion onset,
leg lift apex, foot landing) have to be derived from joints.

Public functions:
    detect_pitcher_events(pitcher_frames, release_time, search_back=4.0)
        → dict with windup_onset_t, knee_high_t, foot_landing_t plus diagnostics

A `pitcher_frames` is a list of (time_unix, world_pos_dict) tuples sorted by
time, where world_pos_dict maps bone_id -> (x, y, z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Joint ids
PELVIS = 0
KNEE_LT, KNEE_RT = 11, 3
FOOT_LT, FOOT_RT = 12, 4
HAND_LT, HAND_RT = 67, 28
SHOULDER_LT, SHOULDER_RT = 65, 26


@dataclass
class PitcherEvents:
    """Kinematically-derived pitcher delivery sub-events. All times are unix seconds.

    Conceptual order during a normal delivery:
        windup_onset  →  knee_high  →  foot_landing  →  release_t (given)
    Each may be None if not detectable.
    """
    windup_onset_t: Optional[float]
    knee_high_t: Optional[float]
    knee_high_height: Optional[float]
    foot_landing_t: Optional[float]
    release_t: float                    # passed in (statsapi start_time)
    front_leg_side: Optional[str]       # "LT" or "RT" (the lifted leg)


def _xy_arrays(frames, bone_id):
    """Extract (times, y, x, z) arrays for one joint over a frame sequence.
    Returns (times, ys, xs, zs) ndarrays; entries with missing joint are NaN."""
    n = len(frames)
    ts = np.empty(n)
    ys = np.full(n, np.nan)
    xs = np.full(n, np.nan)
    zs = np.full(n, np.nan)
    for i, (t, wp) in enumerate(frames):
        ts[i] = t
        if bone_id in wp:
            xs[i], ys[i], zs[i] = wp[bone_id]
    return ts, ys, xs, zs


def _smooth(arr, window=5):
    """Simple centered moving average; NaNs propagate."""
    if window <= 1:
        return arr
    pad = window // 2
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo, hi = max(0, i - pad), min(len(arr), i + pad + 1)
        seg = arr[lo:hi]
        out[i] = np.nanmean(seg)
    return out


def detect_pitcher_events(
    pitcher_frames,
    release_t: float,
    search_back: float = 4.0,
) -> PitcherEvents:
    """Detect (windup_onset, knee_high, foot_landing) for one pitch.

    `release_t` is the trusted release time (statsapi start_time_unix).
    We look back `search_back` seconds before release to find the events.

    Strategy:
        1. Pick the pitcher's frames within [release_t - search_back, release_t].
        2. Determine front leg by which knee rises higher in this window.
        3. knee_high_t  = arg-max of front-knee y (must rise >= 0.8 ft above baseline).
        4. foot_landing_t = first frame after knee_high where front foot is near
           ground (< 1.5 ft) and roughly stationary (|vy| < 1.0 ft/s).
        5. windup_onset_t = first sustained motion. Compute throwing-hand
           y-position rolling std over a 0.4s window; baseline = first 0.7s
           of the search window. Onset = first window where std > 4x baseline
           (with a minimum absolute threshold).
    """
    # Subset to search window
    win_lo = release_t - search_back
    sub = [(t, wp) for (t, wp) in pitcher_frames if win_lo <= t <= release_t]
    if len(sub) < 30:
        return PitcherEvents(None, None, None, None, release_t, None)

    # Extract trajectories
    ts, knee_lt_y, _, _ = _xy_arrays(sub, KNEE_LT)
    _, knee_rt_y, _, _ = _xy_arrays(sub, KNEE_RT)
    _, foot_lt_y, _, _ = _xy_arrays(sub, FOOT_LT)
    _, foot_rt_y, _, _ = _xy_arrays(sub, FOOT_RT)
    _, hand_lt_y, _, _ = _xy_arrays(sub, HAND_LT)
    _, hand_rt_y, _, _ = _xy_arrays(sub, HAND_RT)
    _, _, pelv_x, pelv_z = _xy_arrays(sub, PELVIS)

    knee_lt_s = _smooth(knee_lt_y, 3)
    knee_rt_s = _smooth(knee_rt_y, 3)

    # Determine front leg: whichever knee shows higher peak above baseline
    base_knee_lt = np.nanmedian(knee_lt_s[: max(1, len(knee_lt_s) // 4)])
    base_knee_rt = np.nanmedian(knee_rt_s[: max(1, len(knee_rt_s) // 4)])
    lift_lt = np.nanmax(knee_lt_s) - base_knee_lt
    lift_rt = np.nanmax(knee_rt_s) - base_knee_rt
    if lift_lt < 0.8 and lift_rt < 0.8:
        # No clear leg lift detected — skip the kinematic events
        front = None
        knee_high_t = None
        knee_high_h = None
        foot_landing_t = None
    else:
        if lift_lt >= lift_rt:
            front = "LT"
            knee_y = knee_lt_s
            foot_y = _smooth(foot_lt_y, 3)
        else:
            front = "RT"
            knee_y = knee_rt_s
            foot_y = _smooth(foot_rt_y, 3)
        kh_idx = int(np.nanargmax(knee_y))
        knee_high_t = float(ts[kh_idx])
        knee_high_h = float(knee_y[kh_idx])

        # Foot landing: after knee_high, find first frame where foot near ground & slow
        foot_landing_t = None
        for i in range(kh_idx, len(ts) - 1):
            if not np.isnan(foot_y[i]) and foot_y[i] < 1.5:
                # check vy
                dt = ts[i + 1] - ts[i]
                vy = abs(foot_y[i + 1] - foot_y[i]) / max(dt, 1e-6)
                if vy < 1.5:
                    foot_landing_t = float(ts[i])
                    break

    # Windup onset: detect from throwing-hand y-trajectory (use whichever hand
    # has more variance over the window — that's likely the throwing hand)
    var_lt = np.nanstd(hand_lt_y) if not np.all(np.isnan(hand_lt_y)) else 0.0
    var_rt = np.nanstd(hand_rt_y) if not np.all(np.isnan(hand_rt_y)) else 0.0
    throwing_hand_y = hand_rt_y if var_rt >= var_lt else hand_lt_y
    throwing_hand_y = _smooth(throwing_hand_y, 3)

    # Rolling std over ~0.4s windows (12 frames @ 30fps)
    win_n = 12
    rolling_std = np.full_like(throwing_hand_y, np.nan)
    for i in range(len(throwing_hand_y)):
        lo = max(0, i - win_n // 2)
        hi = min(len(throwing_hand_y), i + win_n // 2 + 1)
        rolling_std[i] = np.nanstd(throwing_hand_y[lo:hi])

    # Baseline: the first 0.7s
    baseline_idx = int(0.7 / max(np.median(np.diff(ts)), 1e-6))
    baseline_idx = max(8, min(baseline_idx, len(rolling_std) // 3))
    baseline_std = np.nanmedian(rolling_std[:baseline_idx])
    baseline_std = max(baseline_std, 0.02)  # floor so we don't trigger on pure noise

    # Onset: first frame where rolling_std > 4x baseline (and min 0.10 ft absolute).
    # Constrained to a physically plausible window before knee_high
    # ([knee_high - 2.0s, knee_high - 0.2s]); if knee_high isn't available
    # we fall back to a window before release.
    onset_threshold = max(4 * baseline_std, 0.10)
    if knee_high_t is not None:
        onset_window_lo = knee_high_t - 2.0
        onset_window_hi = knee_high_t - 0.2
    else:
        onset_window_lo = release_t - 2.5
        onset_window_hi = release_t - 0.5
    onset_t = None
    for i in range(baseline_idx, len(rolling_std)):
        if ts[i] < onset_window_lo: continue
        if ts[i] > onset_window_hi: break
        if rolling_std[i] > onset_threshold:
            sustained = sum(1 for j in range(i, min(i + 6, len(rolling_std)))
                            if rolling_std[j] > onset_threshold)
            if sustained >= 4:
                onset_t = float(ts[i])
                break
    # Fallback if no qualifying onset found inside the window
    if onset_t is None and knee_high_t is not None:
        onset_t = float(knee_high_t - 1.2)

    return PitcherEvents(
        windup_onset_t=onset_t,
        knee_high_t=knee_high_t,
        knee_high_height=knee_high_h,
        foot_landing_t=foot_landing_t,
        release_t=release_t,
        front_leg_side=front,
    )


__all__ = ["PitcherEvents", "detect_pitcher_events"]
