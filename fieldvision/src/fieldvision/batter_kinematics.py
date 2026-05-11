"""Derive batter swing sub-events from skeletal + bat trajectories.

Public functions:
    detect_batter_events(batter_frames, bat_frames, release_t)
        Returns BatterEvents with swing_trigger_t, peak_bat_speed_t,
        load_onset_t (optional), stride_land_t (optional).

A "swing" is detected by bat-head 3D speed crossing a threshold (waggle is
typically <10 ft/s, swings ramp to 50-80+ ft/s within ~150ms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BatterEvents:
    swing_trigger_t: Optional[float]   # bat-head speed crosses SWING_THRESHOLD
    peak_bat_speed_t: Optional[float]  # max bat-head speed (≈ contact / near-contact)
    peak_bat_speed_value: Optional[float]
    load_onset_t: Optional[float]      # first sustained pelvis rotation back (heuristic)
    stride_land_t: Optional[float]     # front foot replants (heuristic, may be None)
    front_leg_side: Optional[str]      # "LT" or "RT" (the lifted/stride leg)
    has_swing: bool


PELVIS = 0
KNEE_LT, KNEE_RT = 11, 3
FOOT_LT, FOOT_RT = 12, 4

# Heuristic thresholds (ft/s)
WAGGLE_SPEED_CEILING = 12.0
SWING_TRIGGER_SPEED = 25.0


def _speed_series(positions, times):
    """3D speed (ft/s) between consecutive frames."""
    if len(positions) < 2:
        return np.array([]), np.array([])
    dts = np.diff(times)
    dts = np.where(dts == 0, 1e-6, dts)
    diffs = np.diff(positions, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / dts
    ts_mid = (times[1:] + times[:-1]) / 2
    return ts_mid, speeds


def detect_batter_events(batter_frames, bat_frames, release_t,
                         search_back=2.0, search_forward=0.6):
    """Detect batter swing events.

    batter_frames: list of (t, joint_dict)
    bat_frames:    list of (t, head_xyz_tuple, handle_xyz_tuple)  (head is what we track)
    release_t:     statsapi physical release time

    Returns BatterEvents with all detected events (or None for any that can't be found).
    """
    win_lo = release_t - search_back
    win_hi = release_t + search_forward

    # Bat head trajectory
    bat_in = [(t, head) for (t, head, _handle) in bat_frames if win_lo <= t <= win_hi]
    if len(bat_in) < 8:
        return BatterEvents(None, None, None, None, None, None, has_swing=False)

    bat_t = np.array([b[0] for b in bat_in])
    bat_pos = np.array([b[1] for b in bat_in])
    valid = ~np.any(np.isnan(bat_pos), axis=1)
    if valid.sum() < 8:
        return BatterEvents(None, None, None, None, None, None, has_swing=False)
    bat_t = bat_t[valid]; bat_pos = bat_pos[valid]

    ts_mid, speeds = _speed_series(bat_pos, bat_t)
    if len(speeds) == 0:
        return BatterEvents(None, None, None, None, None, None, has_swing=False)

    # smooth speed over ~3 frames
    if len(speeds) >= 3:
        speeds_s = np.convolve(speeds, np.ones(3)/3, mode="same")
    else:
        speeds_s = speeds

    peak_idx = int(np.argmax(speeds_s))
    peak_speed = float(speeds_s[peak_idx])

    # Has-swing test: did speed exceed SWING_TRIGGER_SPEED at some point?
    above = speeds_s >= SWING_TRIGGER_SPEED
    if not np.any(above):
        # Just a take/check-swing. No real swing motion.
        return BatterEvents(None, None, None, None, None, None, has_swing=False)

    # swing_trigger = first frame in run that culminates in the peak where speed first
    # crosses WAGGLE_SPEED_CEILING upward (start of acceleration)
    trigger_idx = 0
    for i in range(peak_idx, -1, -1):
        if speeds_s[i] < WAGGLE_SPEED_CEILING:
            trigger_idx = i + 1
            break
    trigger_idx = max(0, min(trigger_idx, len(ts_mid) - 1))
    swing_trigger_t = float(ts_mid[trigger_idx])
    peak_bat_speed_t = float(ts_mid[peak_idx])

    # Batter front-leg detection (similar to pitcher): which knee lifts higher
    # near swing_trigger?
    front_leg = None
    knee_lt_y_seq = []
    knee_rt_y_seq = []
    foot_lt_y_seq = []
    foot_rt_y_seq = []
    rel_ts = []
    for (t, wp) in batter_frames:
        if t < swing_trigger_t - 0.5 or t > swing_trigger_t + 0.5:
            continue
        rel_ts.append(t)
        knee_lt_y_seq.append(wp.get(KNEE_LT, (None, None, None))[1])
        knee_rt_y_seq.append(wp.get(KNEE_RT, (None, None, None))[1])
        foot_lt_y_seq.append(wp.get(FOOT_LT, (None, None, None))[1])
        foot_rt_y_seq.append(wp.get(FOOT_RT, (None, None, None))[1])
    rel_ts = np.array(rel_ts)
    knee_lt_y_seq = np.array([np.nan if v is None else v for v in knee_lt_y_seq])
    knee_rt_y_seq = np.array([np.nan if v is None else v for v in knee_rt_y_seq])
    foot_lt_y_seq = np.array([np.nan if v is None else v for v in foot_lt_y_seq])
    foot_rt_y_seq = np.array([np.nan if v is None else v for v in foot_rt_y_seq])

    stride_land_t = None
    if len(rel_ts) >= 10:
        # baseline knee_y = mean for first 0.3s of window
        early_mask = rel_ts < (rel_ts[0] + 0.3)
        bl_lt = np.nanmean(knee_lt_y_seq[early_mask])
        bl_rt = np.nanmean(knee_rt_y_seq[early_mask])
        lift_lt = np.nanmax(knee_lt_y_seq) - bl_lt if not np.isnan(bl_lt) else 0
        lift_rt = np.nanmax(knee_rt_y_seq) - bl_rt if not np.isnan(bl_rt) else 0
        if max(lift_lt, lift_rt) > 0.4:
            front_leg = "LT" if lift_lt > lift_rt else "RT"
            foot_y = foot_lt_y_seq if front_leg == "LT" else foot_rt_y_seq
            # stride land: first time foot_y < 1.2 ft AFTER swing_trigger
            for i in range(len(rel_ts)):
                if rel_ts[i] > swing_trigger_t and not np.isnan(foot_y[i]) and foot_y[i] < 1.5:
                    stride_land_t = float(rel_ts[i])
                    break

    return BatterEvents(
        swing_trigger_t=swing_trigger_t,
        peak_bat_speed_t=peak_bat_speed_t,
        peak_bat_speed_value=peak_speed,
        load_onset_t=None,  # deferred
        stride_land_t=stride_land_t,
        front_leg_side=front_leg,
        has_swing=True,
    )


def interpolate_trajectory(times, positions, target_times):
    """Linearly interpolate (n, 3) trajectory to target_times grid. Returns (m, 3)."""
    out = np.full((len(target_times), positions.shape[1]), np.nan)
    valid = ~np.any(np.isnan(positions), axis=1)
    if valid.sum() < 2:
        return out
    t = times[valid]
    for k in range(positions.shape[1]):
        out[:, k] = np.interp(target_times, t, positions[valid, k],
                              left=np.nan, right=np.nan)
    return out


def _point_to_segment_distance(P, A, B):
    """3D distance from point P to line segment from A to B."""
    AB = B - A
    L2 = np.dot(AB, AB)
    if L2 < 1e-9:
        return np.linalg.norm(P - A), A
    t = np.clip(np.dot(P - A, AB) / L2, 0.0, 1.0)
    closest = A + t * AB
    return np.linalg.norm(P - closest), closest


def ball_bat_min_distance(ball_frames, bat_frames, t_start, t_end, n_samples=400):
    """Time-interpolated minimum 3D distance between ball and the BAT AXIS
    (line segment from handle to head). Contact happens along the bat axis, not
    just at the tip, so distance-to-axis is the right measure.

    ball_frames: list of (t, ball_xyz)
    bat_frames:  list of (t, head_xyz, handle_xyz)
    Returns (min_distance_ft, t_of_min, ball_at_min, closest_point_on_bat) or
    (None, ...) if not computable.
    """
    ball_in = [(t, pos) for (t, pos) in ball_frames if t_start <= t <= t_end]
    bat_in = [(t, head, handle) for (t, head, handle) in bat_frames if t_start <= t <= t_end]
    if len(ball_in) < 2 or len(bat_in) < 2:
        return None, None, None, None
    ball_t = np.array([b[0] for b in ball_in])
    ball_p = np.array([b[1] for b in ball_in])
    bat_t = np.array([b[0] for b in bat_in])
    bat_head = np.array([b[1] for b in bat_in])
    bat_handle = np.array([b[2] for b in bat_in])

    t_lo = max(ball_t.min(), bat_t.min())
    t_hi = min(ball_t.max(), bat_t.max())
    if t_hi <= t_lo:
        return None, None, None, None
    grid = np.linspace(t_lo, t_hi, n_samples)
    bi = interpolate_trajectory(ball_t, ball_p, grid)
    head_i = interpolate_trajectory(bat_t, bat_head, grid)
    handle_i = interpolate_trajectory(bat_t, bat_handle, grid)
    valid = ~(np.any(np.isnan(bi), axis=1) | np.any(np.isnan(head_i), axis=1) | np.any(np.isnan(handle_i), axis=1))
    if valid.sum() == 0:
        return None, None, None, None
    min_d = None
    min_idx = None
    min_closest = None
    valid_grid = grid[valid]
    for k, j in enumerate(np.where(valid)[0]):
        d, closest = _point_to_segment_distance(bi[j], handle_i[j], head_i[j])
        if min_d is None or d < min_d:
            min_d = float(d); min_idx = k; min_closest = closest
    return (min_d, float(valid_grid[min_idx]), bi[np.where(valid)[0][min_idx]], min_closest)
