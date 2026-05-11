"""Filters for impossible postures and impossible bat positions.

Two common data-integrity issues in the wire data:

1. **mlb_player_id collision**: the same MLB player ID gets assigned to two
   different actor_uids within the same game/frame range (e.g., catcher
   and batter share an id, or a previous-AB batter's id lingers). The
   result is that a SELECT WHERE mlb_player_id=X returns rows for two
   different physical people. Symptoms: pelvis_y collapses (one row at
   3 ft = standing, one at 1 ft = crouching), or pelvis_x/z teleports
   (catcher behind plate vs batter in box).

2. **Bat-frame stray entries**: `bat_frame` is populated whenever a bat
   is inferred, but it can pick up: bat on the ground between pitches
   (bat_y ~ 0), bat being passed to bat boy, or bat held by on-deck
   batter (far from active batter). These are not the active swinging
   bat for the current pitch.

These functions filter both data sources to keep only valid frames.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


# Bounds in stadium feet
PELVIS_Y_STANDING_LO = 2.4   # below this is crouching (catcher) or sitting
PELVIS_Y_STANDING_HI = 5.0
HEAD_Y_STANDING_LO = 4.0     # below this is crouched
BAT_LENGTH_LO = 2.5          # 30 in
BAT_LENGTH_HI = 3.2          # 38 in
BAT_HEAD_Y_MIN = 1.5         # below this is on the ground / not held
BAT_HANDLE_TO_BATTER_HAND_MAX = 3.5  # bat handle should be near batter's hands


def filter_batter_frames_by_actor_uid(rows, batter_id):
    """When the SQL query returns multiple actor_frame rows per timestamp for
    the same mlb_player_id, pick the actor_uid most likely to be the real
    standing batter.

    Heuristic: rank candidates by mean pelvis_y + head_y over the window;
    pick whichever has the higher score (taller).

    `rows` is a list of (time_unix, actor_uid, pelvis_x, pelvis_y, pelvis_z,
    head_y, ...other_joints...). Returns the filtered rows (one row per
    timestamp, the "best" actor).

    If only one actor_uid is present, returns rows unchanged.
    """
    # Group by actor_uid
    from collections import defaultdict
    by_uid = defaultdict(list)
    for r in rows:
        by_uid[r[1]].append(r)
    if len(by_uid) <= 1:
        return rows

    # Score each uid by mean pelvis_y + head_y (taller = real batter)
    def score(uid_rows):
        py = [r[3] for r in uid_rows if r[3] is not None]
        hy = [r[5] for r in uid_rows if r[5] is not None]
        if not py or not hy: return -1
        return float(np.mean(py) + np.mean(hy))

    scores = {uid: score(rs) for uid, rs in by_uid.items()}
    best_uid = max(scores, key=scores.get)
    return by_uid[best_uid]


def is_valid_standing_pose(pelvis_y, head_y):
    """True if this is a plausible standing-batter pose."""
    if pelvis_y is None or head_y is None: return False
    if not (PELVIS_Y_STANDING_LO <= pelvis_y <= PELVIS_Y_STANDING_HI):
        return False
    if head_y < HEAD_Y_STANDING_LO:
        return False
    if head_y < pelvis_y + 1.0:  # head must be at least 1 ft above pelvis
        return False
    return True


def filter_bat_frames(bat_records, batter_hand_positions=None):
    """Per-frame PHYSICAL sanity filter for bat data only.

    Drops frames where:
      - bat length is outside [2.5, 3.2] ft (the recorded "bat" doesn't have
        the right physical dimensions — corrupted record)
      - bat head_y < 1.5 ft (bat clearly on the ground / not held by anyone)

    Does NOT filter based on proximity to batter's hands — that's a pitch-level
    quality check (see assess_pitch_quality) that decides whether to DROP the
    whole pitch from analysis, rather than just censoring frames. Per-frame
    filtering would hide bad data from videos; per-pitch filtering is honest.

    `batter_hand_positions` is accepted for backwards compatibility but ignored.
    """
    out = []
    for r in bat_records:
        t, head, handle = r
        if head is None or handle is None: continue
        if any(v is None for v in head) or any(v is None for v in handle):
            continue
        L = float(np.linalg.norm(np.array(head) - np.array(handle)))
        if not (BAT_LENGTH_LO <= L <= BAT_LENGTH_HI):
            continue
        if head[1] < BAT_HEAD_Y_MIN:
            continue
        out.append(r)
    return out


# ────────────────────────────────────────────────────────────
# Pitch-level data-quality assessment
# ────────────────────────────────────────────────────────────


PITCH_QUALITY_MIN_BAT_IN_HANDS = 0.80   # at least 80% of bat frames within reach of batter
PITCH_QUALITY_MIN_BODY_STANDING = 0.90  # at least 90% of body frames in standing posture
PITCH_QUALITY_MIN_BAT_FRAMES = 30        # need at least this many bat frames to assess at all


def assess_pitch_quality(batter_frames, bat_frames):
    """Per-pitch decision: is the data clean enough to keep this pitch in analysis?

    Returns dict with diagnostic fractions + `is_clean` boolean.

    batter_frames: list of (t, joint_dict)
    bat_frames:    list of (t, head_xyz, handle_xyz) — already physically-sane (passed
                   filter_bat_frames)

    Logic:
        - bat_in_hands_fraction = fraction of bat frames where bat handle is within
          BAT_HANDLE_TO_BATTER_HAND_MAX of either hand. Low fraction = the recorded
          bat is not actually in this batter's hands (probably mislabeled).
        - body_standing_fraction = fraction of batter frames where pelvis_y is in
          [PELVIS_Y_STANDING_LO, PELVIS_Y_STANDING_HI] AND head_y >= HEAD_Y_STANDING_LO.
          Low fraction = the "batter" data is actually catcher/umpire data.

    is_clean is True iff both fractions exceed their thresholds AND there are enough
    bat frames to make the assessment.
    """
    diag = {"n_bat_frames": len(bat_frames),
            "n_body_frames": len(batter_frames),
            "bat_in_hands_fraction": None,
            "body_standing_fraction": None,
            "is_clean": False,
            "reason": ""}

    if len(bat_frames) < PITCH_QUALITY_MIN_BAT_FRAMES:
        diag["reason"] = f"too few bat frames ({len(bat_frames)} < {PITCH_QUALITY_MIN_BAT_FRAMES})"
        return diag
    if len(batter_frames) < 30:
        diag["reason"] = f"too few body frames ({len(batter_frames)} < 30)"
        return diag

    # bat_in_hands: build a time index of hand positions
    hand_pos = {}
    for entry in batter_frames:
        t = entry[0]
        wp = entry[1] if isinstance(entry[1], dict) else None
        if wp is None: continue
        hand_pos[t] = (wp.get(67), wp.get(28))  # hand_lt, hand_rt
    if not hand_pos:
        diag["reason"] = "no hand positions in body frames"
        return diag
    hand_ts = sorted(hand_pos.keys())

    n_in_hand = 0
    n_with_hand_ref = 0
    for (t, head, handle) in bat_frames:
        i = np.searchsorted(hand_ts, t)
        i = max(0, min(i, len(hand_ts) - 1))
        t_near = hand_ts[i]
        if abs(t_near - t) > 0.1: continue
        hand_lt, hand_rt = hand_pos[t_near]
        if hand_lt is None and hand_rt is None: continue
        n_with_hand_ref += 1
        handle_arr = np.array(handle, dtype=float)
        d_lt = (np.linalg.norm(handle_arr - np.array(hand_lt, dtype=float))
                if hand_lt is not None else 99)
        d_rt = (np.linalg.norm(handle_arr - np.array(hand_rt, dtype=float))
                if hand_rt is not None else 99)
        if min(d_lt, d_rt) <= BAT_HANDLE_TO_BATTER_HAND_MAX:
            n_in_hand += 1
    bat_in_hands_fraction = n_in_hand / n_with_hand_ref if n_with_hand_ref > 0 else 0.0
    diag["bat_in_hands_fraction"] = bat_in_hands_fraction

    # body_standing: check pelvis_y + head_y per frame
    n_standing = 0; n_body = 0
    for entry in batter_frames:
        wp = entry[1] if isinstance(entry[1], dict) else None
        if wp is None: continue
        pelvis = wp.get(0)
        head = wp.get(21)
        if pelvis is None or head is None: continue
        py = pelvis[1] if hasattr(pelvis, "__len__") else None
        hy = head[1] if hasattr(head, "__len__") else None
        if py is None or hy is None: continue
        n_body += 1
        if is_valid_standing_pose(py, hy):
            n_standing += 1
    body_standing_fraction = n_standing / n_body if n_body > 0 else 0.0
    diag["body_standing_fraction"] = body_standing_fraction

    # Final decision
    if bat_in_hands_fraction < PITCH_QUALITY_MIN_BAT_IN_HANDS:
        diag["reason"] = (f"bat_in_hands={bat_in_hands_fraction:.0%} "
                          f"< {PITCH_QUALITY_MIN_BAT_IN_HANDS:.0%} — "
                          f"recorded bat isn't in this batter's hands")
        return diag
    if body_standing_fraction < PITCH_QUALITY_MIN_BODY_STANDING:
        diag["reason"] = (f"body_standing={body_standing_fraction:.0%} "
                          f"< {PITCH_QUALITY_MIN_BODY_STANDING:.0%} — "
                          f"body data is probably catcher/umpire mislabeled")
        return diag
    diag["is_clean"] = True
    return diag


def load_clean_batter_actor_frames(conn, batter_id, t_lo, t_hi, joint_cols_select):
    """SELECT actor_frame rows for batter, then resolve actor_uid collisions
    in favor of the standing-batter actor. Returns rows in same shape but
    with at most one row per timestamp.

    `joint_cols_select` is the SQL projection string for actor_frame.
    """
    # We need actor_uid in the projection. Splice it into the SELECT.
    proj = "time_unix, actor_uid, " + joint_cols_select
    rows = conn.execute(
        f"SELECT {proj} FROM actor_frame "
        "WHERE mlb_player_id=? AND time_unix BETWEEN ? AND ? "
        "ORDER BY time_unix",
        (batter_id, t_lo, t_hi),
    ).fetchall()
    if not rows:
        return []
    # Group by time, resolve collisions
    from collections import defaultdict
    by_time = defaultdict(list)
    for r in rows:
        by_time[r[0]].append(r)
    # Per timestamp, pick the row whose pelvis_y is in the "standing" range,
    # or the higher pelvis_y if neither matches.
    out = []
    for t in sorted(by_time.keys()):
        candidates = by_time[t]
        if len(candidates) == 1:
            out.append(candidates[0]); continue
        # Score each: prefer pelvis_y in [2.4, 5.0] AND head_y > 4.0
        # The first 3 fields are time, actor_uid; then joint_cols start with the
        # given projection. We need to know where pelvis_x/y/z and head_y are.
        # Convention: caller's joint_cols_select MUST start with pelvis_x, pelvis_y, pelvis_z.
        # And include head_y at column 14 (offset based on JOINT_COLS standard).
        # Simplest: pick the candidate with the highest pelvis_y.
        # (pelvis_y is column 3 in projection: time=0, actor_uid=1, pelvis_x=2, pelvis_y=3)
        def py(r):
            v = r[3]
            return v if v is not None else -1
        out.append(max(candidates, key=py))
    return out
