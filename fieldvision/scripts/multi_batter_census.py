"""Multi-batter pre-pitch oscillation census across all captured games.

For every batter with >= MIN_PITCHES total pitches across the data corpus, run
the oscillation pipeline (joint stats + FFT + PCA + phase at 4 events) and
cache results to a pickle so subsequent ranking/visualization runs are fast.

Usage:
    python scripts/multi_batter_census.py [--min-pitches 10] [--output data/oscillation_report/pre_pitch_preparatory_movement/census.pkl]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.storage import JOINT_COLS
from fieldvision.parquet_readers import list_games, open_game
from fieldvision.pitch_kinematics import detect_pitcher_events
from fieldvision.validate_frames import (load_clean_batter_actor_frames,
                                          filter_bat_frames,
                                          assess_pitch_quality)

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
JOINT_NAMES = {bid: name for bid, name in JOINT_COLS}

PRE_OSC_SECONDS = 5.0
SAMPLE_HZ = 30.0
N_PRE_OSC_FRAMES = int(PRE_OSC_SECONDS * SAMPLE_HZ)


# ────────────────────────────────────────────────────────────
# Cross-game batter discovery
# ────────────────────────────────────────────────────────────


def get_batter_pitches_across_games(data_dir: Path):
    """Return dict[batter_id] -> list of pitch dicts across all games."""
    by_batter = defaultdict(list)
    for game_pk in list_games(data_dir):
        try:
            conn = open_game(game_pk, data_dir)
            for r in conn.execute(
                "SELECT batter_id, play_id, pitcher_id, start_time_unix, "
                "pitch_type, result_call, batter_side, pitcher_throws, "
                "start_speed FROM pitch_label "
                "WHERE batter_id IS NOT NULL AND start_time_unix IS NOT NULL"
            ).fetchall():
                side = r[6] if r[6] in ("L", "R", "S") else "?"
                # Treat switch hitters' L and R at-bats as separate "batter entries"
                # since stance/waggle pattern is completely different per side.
                key = (r[0], side)
                by_batter[key].append({
                    "game_pk": game_pk,
                    "play_id": r[1],
                    "pitcher_id": r[2],
                    "release_t": r[3],
                    "pitch_type": r[4],
                    "result_call": r[5],
                    "batter_side": side,
                    "pitcher_throws": r[7],
                    "start_speed": r[8],
                })
            conn.close()
        except Exception as e:
            print(f"  WARN: skipping game {game_pk}: {e}")
    return dict(by_batter)


# Player-name lookup (cached)
_NAME_CACHE = {}
def get_player_name(player_id):
    if player_id in _NAME_CACHE:
        return _NAME_CACHE[player_id]
    if player_id is None or player_id < 0:
        _NAME_CACHE[player_id] = f"player_{player_id}"
        return _NAME_CACHE[player_id]
    try:
        req = urllib.request.Request(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}",
            headers={"User-Agent": "FV-census/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            d = json.loads(resp.read())
        name = d["people"][0]["fullName"]
    except Exception:
        name = f"player_{player_id}"
    _NAME_CACHE[player_id] = name
    return name


# ────────────────────────────────────────────────────────────
# Per-pitch data loading + analysis
# ────────────────────────────────────────────────────────────


def load_pitch_data(conn, batter_id, play_id, pitcher_id, release_t):
    joint_cols_select = ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)
    p_rows_raw = load_clean_batter_actor_frames(
        conn, pitcher_id, release_t - 5, release_t + 0.5, joint_cols_select)
    p_rows = [(r[0],) + r[2:] for r in p_rows_raw]
    if len(p_rows) < 30:
        return None
    p_frames = []
    for r in p_rows:
        wp = {}
        for i, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1 + i*3], r[2 + i*3], r[3 + i*3]
            if x is not None: wp[bid] = (x, y, z)
        p_frames.append((r[0], wp))
    ev = detect_pitcher_events(p_frames, release_t, search_back=4.0)
    if ev.windup_onset_t is None or ev.knee_high_t is None or ev.foot_landing_t is None:
        return None

    onset_t = ev.windup_onset_t
    win_lo = onset_t - PRE_OSC_SECONDS
    win_hi = release_t + 0.2

    b_rows_raw = load_clean_batter_actor_frames(conn, batter_id, win_lo, win_hi, joint_cols_select)
    b_rows = [(r[0],) + r[2:] for r in b_rows_raw]
    if len(b_rows) < 100:
        return None

    times = np.array([r[0] for r in b_rows])
    joint_mat = np.full((len(b_rows), len(JOINT_BIDS), 3), np.nan)
    for i, r in enumerate(b_rows):
        for j, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1 + j*3], r[2 + j*3], r[3 + j*3]
            if x is not None: joint_mat[i, j] = (x, y, z)

    bat_rows = conn.execute(
        "SELECT time_unix, head_x, head_y, head_z, handle_x, handle_y, handle_z "
        "FROM bat_frame WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (win_lo, win_hi),
    ).fetchall()

    # Per-pitch data-quality check: drop the whole pitch if bat isn't in batter's
    # hands enough OR body data is wrong (catcher/umpire mislabeled as batter).
    bat_records_for_qa = filter_bat_frames(
        [(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6])) for r in bat_rows]
    )
    body_for_qa = []
    for r in b_rows:
        wp = {}
        for j, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1 + j*3], r[2 + j*3], r[3 + j*3]
            if x is not None: wp[bid] = (x, y, z)
        body_for_qa.append((r[0], wp))
    quality = assess_pitch_quality(body_for_qa, bat_records_for_qa)
    if not quality["is_clean"]:
        return None

    bat_times = np.array([r[0] for r in bat_rows]) if bat_rows else np.array([])
    bat_head = np.array([(r[1], r[2], r[3]) for r in bat_rows]) if bat_rows else np.empty((0, 3))
    bat_handle = np.array([(r[4], r[5], r[6]) for r in bat_rows]) if bat_rows else np.empty((0, 3))

    t_grid = np.arange(win_lo, win_hi, 1.0 / SAMPLE_HZ)
    n_grid = len(t_grid)

    def resample(times_in, vals_in):
        if len(times_in) == 0:
            return np.full((n_grid, vals_in.shape[1]), np.nan)
        out = np.empty((n_grid, vals_in.shape[1]))
        for k in range(vals_in.shape[1]):
            v = vals_in[:, k]
            mask = ~np.isnan(v)
            if mask.sum() < 2:
                out[:, k] = np.nan
                continue
            out[:, k] = np.interp(t_grid, times_in[mask], v[mask])
        return out

    joint_flat = joint_mat.reshape(len(b_rows), -1)
    joint_resampled = resample(times, joint_flat)
    bat_head_r = resample(bat_times, bat_head) if len(bat_times) else np.full((n_grid, 3), np.nan)
    bat_handle_r = resample(bat_times, bat_handle) if len(bat_times) else np.full((n_grid, 3), np.nan)

    def t_to_idx(t):
        if t is None: return None
        return int(np.clip((t - win_lo) * SAMPLE_HZ, 0, n_grid - 1))

    return {
        "play_id": play_id,
        "release_t": release_t,
        "joint_mat": joint_resampled,
        "bat_head": bat_head_r,
        "bat_handle": bat_handle_r,
        "windup_onset_t": onset_t,
        "knee_high_t": ev.knee_high_t,
        "foot_landing_t": ev.foot_landing_t,
        "front_leg_side": ev.front_leg_side,
        "idx_windup_onset": t_to_idx(onset_t),
        "idx_knee_high": t_to_idx(ev.knee_high_t),
        "idx_foot_landing": t_to_idx(ev.foot_landing_t),
        "idx_release": t_to_idx(release_t),
    }


# ────────────────────────────────────────────────────────────
# Phase computation (numpy-only Hilbert via FFT to avoid scipy hot reload)
# ────────────────────────────────────────────────────────────


def _hilbert_phase(signal):
    """Compute analytic-signal phase via FFT (Hilbert transform)."""
    n = len(signal)
    X = np.fft.fft(signal)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = np.fft.ifft(X * h)
    return np.angle(analytic)


def _bandpass_fft(signal, sample_hz, low_hz, high_hz):
    """Simple FFT-based band-pass filter."""
    n = len(signal)
    X = np.fft.rfft(signal)
    f = np.fft.rfftfreq(n, 1 / sample_hz)
    mask = (f >= low_hz) & (f <= high_hz)
    X_filt = np.where(mask, X, 0)
    return np.fft.irfft(X_filt, n)


def phase_at_event(signal, sample_hz, idx, dom_freq=None, bandwidth=0.6):
    """Filter narrowly around dom_freq, Hilbert phase at idx."""
    pre = signal[:N_PRE_OSC_FRAMES]
    if np.any(np.isnan(pre)):
        return np.nan
    pre_centered = pre - pre.mean()
    if dom_freq is None:
        windowed = pre_centered * np.hanning(len(pre_centered))
        X = np.abs(np.fft.rfft(windowed))
        f = np.fft.rfftfreq(len(windowed), 1 / sample_hz)
        m = (f > 0.2) & (f < 4.0)
        if m.sum() == 0: return np.nan
        dom_freq = float(f[m][np.argmax(X[m])])
    full = signal.copy()
    nans = np.isnan(full)
    if nans.all(): return np.nan
    full = full - np.nanmean(full[:N_PRE_OSC_FRAMES])
    if nans.any():
        ix = np.arange(len(full))
        full[nans] = np.interp(ix[nans], ix[~nans], full[~nans])
    low = max(0.05, dom_freq - bandwidth / 2)
    high = min(sample_hz / 2 - 0.5, dom_freq + bandwidth / 2)
    if low >= high: return np.nan
    filtered = _bandpass_fft(full, sample_hz, low, high)
    phase = _hilbert_phase(filtered)
    return float(phase[min(idx, len(phase) - 1)])


def pca_phase_at_event(traj, idx):
    if traj is None: return np.nan
    if idx >= len(traj): idx = len(traj) - 1
    pc1, pc2 = traj[idx, 0], traj[idx, 1]
    if np.isnan(pc1) or np.isnan(pc2): return np.nan
    return float(np.arctan2(pc2, pc1))


# ────────────────────────────────────────────────────────────
# Per-batter pipeline
# ────────────────────────────────────────────────────────────


def analyze_batter(batter_id, pitches_meta, verbose=False):
    """Returns a per-batter analysis dict, or None if too few usable pitches."""
    pitches = []
    db_conns = {}
    for pm in pitches_meta:
        if pm["game_pk"] not in db_conns:
            db_conns[pm["game_pk"]] = open_game(pm["game_pk"])
        conn = db_conns[pm["game_pk"]]
        d = load_pitch_data(conn, batter_id, pm["play_id"], pm["pitcher_id"], pm["release_t"])
        if d is None: continue
        d.update(pitch_type=pm["pitch_type"], result_call=pm["result_call"],
                 batter_side=pm["batter_side"], pitcher_throws=pm["pitcher_throws"],
                 pitcher_id=pm["pitcher_id"], game_pk=pm["game_pk"],
                 start_speed=pm["start_speed"])
        pitches.append(d)
    for c in db_conns.values():
        c.close()

    n_usable = len(pitches)
    if n_usable < 5:
        return None

    n_dim = len(JOINT_BIDS) * 3

    # === Joint movement stats (per-pitch × per-joint stddev magnitude) ===
    joint_stds = np.zeros((n_usable, len(JOINT_BIDS)))
    for pi, p in enumerate(pitches):
        seg = p["joint_mat"][:N_PRE_OSC_FRAMES].reshape(N_PRE_OSC_FRAMES, len(JOINT_BIDS), 3)
        for ji in range(len(JOINT_BIDS)):
            joint_stds[pi, ji] = np.linalg.norm(np.nanstd(seg[:, ji, :], axis=0))

    # === Per-pitch dominant frequencies (use just hand_rt_y as the "waggle" signal) ===
    hand_rt_idx = (JOINT_BIDS.index(28) * 3) + 1  # hand_rt y
    pelvis_x_idx = 0
    bat_freqs = []
    hand_freqs = []
    pelvis_freqs = []
    for p in pitches:
        for sig_idx, freq_list in [(pelvis_x_idx, pelvis_freqs), (hand_rt_idx, hand_freqs)]:
            sig = p["joint_mat"][:N_PRE_OSC_FRAMES, sig_idx]
            if np.any(np.isnan(sig)):
                freq_list.append(np.nan); continue
            sig_c = (sig - sig.mean()) * np.hanning(len(sig))
            X = np.abs(np.fft.rfft(sig_c))
            f = np.fft.rfftfreq(len(sig_c), 1 / SAMPLE_HZ)
            m = (f > 0.2) & (f < 4.0)
            if m.sum() == 0:
                freq_list.append(np.nan)
            else:
                freq_list.append(float(f[m][np.argmax(X[m])]))
        # bat_axis_angle (head-handle) Z-Y plane
        head, handle = p["bat_head"][:N_PRE_OSC_FRAMES], p["bat_handle"][:N_PRE_OSC_FRAMES]
        if np.any(np.isnan(head)) or np.any(np.isnan(handle)):
            bat_freqs.append(np.nan); continue
        axis = head - handle
        bang = np.arctan2(axis[:, 1], axis[:, 2])
        bang_c = (bang - bang.mean()) * np.hanning(len(bang))
        X = np.abs(np.fft.rfft(bang_c))
        f = np.fft.rfftfreq(len(bang_c), 1 / SAMPLE_HZ)
        m = (f > 0.2) & (f < 4.0)
        bat_freqs.append(float(f[m][np.argmax(X[m])]) if m.sum() > 0 else np.nan)

    # === PCA on stacked pre-osc frames ===
    # Posture vector = 60 joint dims (20 joints × xyz) + 6 bat dims (head xyz + handle xyz) = 66.
    # The bat IS part of the batter's preparatory posture — must be in the feature vector
    # so that waggle of the bat oscillates within the same PC basis as body motion.
    #
    # Two-pass fit to remove pre-stance setup movement:
    #   pass 1: fit PCs on all pre-osc frames (setup + stance mixed)
    #   pass 2: per pitch, find density centroid in PC1×PC2 and drop frames more
    #           than 3·MAD from it (the setup/walking-to-the-box frames are
    #           outliers in the projection), then refit PCs on remaining frames.
    # This keeps the basis focused on in-stance waggle dynamics.
    HAND_RT_IDX = JOINT_BIDS.index(28)  # hand_rt
    HAND_LT_IDX = JOINT_BIDS.index(67)  # hand_lt
    HAND_TO_HANDLE_MAX_FT = 1.0  # both wrists within 1 ft of bat handle (just above grip-on-bat p90)

    def posture_mat(p, n=None):
        """66-dim posture per frame with PER-FRAME BODY-CENTROID NORMALIZATION
        and HANDS-ON-BAT validity gating.

        Body centroid = mean of the 20 joint positions in that frame. Subtracting
        it from every joint AND from bat_head / bat_handle makes the posture
        translation-invariant: where the batter stands in the box (depth/lateral
        position) drops out, so PCA / clustering capture pure intrinsic shape
        variation rather than artifactual position offsets.

        Frames where either wrist is further than HAND_TO_HANDLE_MAX_FT from the
        bat handle are NaN'd — that filters out plate-tapping, bat-pointing,
        helmet-fixing and any other moment when both hands aren't gripping the
        bat. Those aren't preparatory stance movement, so we don't want them in
        the PCA basis or the phase analysis."""
        joint_seq = p["joint_mat"]   # (n_frames, 60)
        bat_head = p["bat_head"]     # (n_frames, 3)
        bat_handle = p["bat_handle"] # (n_frames, 3)
        n_frames = len(joint_seq)
        joints_3d = joint_seq.reshape(n_frames, 20, 3)
        body_centroid = np.nanmean(joints_3d, axis=1, keepdims=True)
        joints_norm = (joints_3d - body_centroid).reshape(n_frames, 60)
        bc_flat = body_centroid[:, 0, :]
        bat_head_norm = bat_head - bc_flat
        bat_handle_norm = bat_handle - bc_flat
        seg = np.concatenate([joints_norm, bat_head_norm, bat_handle_norm], axis=1)

        # Hands-on-bat gating
        hand_rt = joints_3d[:, HAND_RT_IDX]
        hand_lt = joints_3d[:, HAND_LT_IDX]
        with np.errstate(invalid="ignore"):
            d_rt = np.linalg.norm(hand_rt - bat_handle, axis=1)
            d_lt = np.linalg.norm(hand_lt - bat_handle, axis=1)
        hands_off = (d_rt > HAND_TO_HANDLE_MAX_FT) | (d_lt > HAND_TO_HANDLE_MAX_FT)
        seg[hands_off] = np.nan

        return seg if n is None else seg[:n]

    def fit_pcs(pitches, stance_masks=None):
        blocks = []
        for pi, p in enumerate(pitches):
            seg = posture_mat(p, N_PRE_OSC_FRAMES)
            valid = ~np.any(np.isnan(seg), axis=1)
            if stance_masks is not None: valid &= stance_masks[pi]
            if valid.sum() < 30: continue
            seg_c = seg[valid] - np.nanmean(seg[valid], axis=0)
            blocks.append(seg_c)
        if not blocks: return None, None
        X = np.vstack(blocks)
        try: U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError: return None, None
        return Vt[:5], (S ** 2) / np.sum(S ** 2)

    # Pass 1: initial fit (setup + stance mixed)
    components, var_explained = fit_pcs(pitches)
    if components is None:
        if verbose: print(f"    SVD failed (pass 1)")
        return None

    # Per-pitch stance mask via TEMPLATE-MATCHING in centroid-normalized
    # posture space. The template for each pitch is the MEDIAN of all its
    # pre-osc frames (median is robust — sporadic divergent frames don't
    # pull it). A frame is "in stance" iff its cosine similarity to that
    # pitch's median template is above STANCE_SIM_THRESHOLD.
    #
    # This is batter-agnostic: no per-batter clustering, no per-batter
    # threshold tuning. A frame either looks like that pitch's typical
    # stance pose or it doesn't.
    STANCE_SIM_THRESHOLD = 0.95

    valid_per_pitch = []
    stance_masks = []
    for pi, p in enumerate(pitches):
        seg = posture_mat(p, N_PRE_OSC_FRAMES)
        valid = ~np.any(np.isnan(seg), axis=1)
        valid_per_pitch.append(valid)
        m = np.zeros(N_PRE_OSC_FRAMES, dtype=bool)
        if valid.sum() < 10:
            stance_masks.append(m); continue
        template = np.median(seg[valid], axis=0)
        t_norm = np.linalg.norm(template) + 1e-9
        v = seg[valid]
        v_norm = np.linalg.norm(v, axis=1) + 1e-9
        sims = (v @ template) / (v_norm * t_norm)
        keep_valid = sims > STANCE_SIM_THRESHOLD
        m[np.where(valid)[0]] = keep_valid
        stance_masks.append(m)
    stance_starts = [0] * len(pitches)  # legacy field; not used by new filter
    n_stance = sum(m.sum() for m in stance_masks)
    n_total = sum(((~np.any(np.isnan(posture_mat(p, N_PRE_OSC_FRAMES)), axis=1)).sum()
                   for p in pitches))
    stance_frac = n_stance / max(n_total, 1)
    if verbose: print(f"    stance filter (sim>{STANCE_SIM_THRESHOLD}): kept "
                      f"{n_stance}/{n_total} ({stance_frac*100:.0f}%) pre-osc frames")

    # Pass 2: refit on stance frames only
    components, var_explained = fit_pcs(pitches, stance_masks=stance_masks)
    if components is None:
        if verbose: print(f"    SVD failed (pass 2)")
        return None

    # Trajectories: per-pitch mean_pose computed from stance frames only,
    # then project full posture sequence (including setup + post-windup motion)
    # onto the stance-fit PCs.
    pca_trajs = [None] * n_usable
    per_pitch_mean_pose = [None] * n_usable
    for pi, p in enumerate(pitches):
        seg = posture_mat(p)
        pre = seg[:N_PRE_OSC_FRAMES]
        sm = stance_masks[pi]
        if sm.sum() < 30:
            pre_valid = ~np.any(np.isnan(pre), axis=1)
        else:
            pre_valid = sm
        if pre_valid.sum() < 30: continue
        mean_pose = np.nanmean(pre[pre_valid], axis=0)
        per_pitch_mean_pose[pi] = mean_pose
        centered = seg - mean_pose
        valid = ~np.any(np.isnan(centered), axis=1)
        traj = np.full((seg.shape[0], 5), np.nan)
        traj[valid] = centered[valid] @ components.T
        pca_trajs[pi] = traj

    # === jPCA: rotational dynamics decomposition on stance-filtered PC trajectories ===
    # Fit jPCA across all of this batter's stance-filtered pre-osc PC trajectories.
    # The dominant rotation plane gives a true oscillation axis, so the phase
    # angle within it is more meaningful than Hilbert-of-PC1 (which assumes
    # PC1 happens to be one axis of the oscillation, which it often isn't).
    from fieldvision.jpca import fit_jpca, jpca_phase
    jpca_result = None
    jpca_x_mean = None
    jpca_plane = None
    try:
        stance_trajs = []
        for pi, p in enumerate(pitches):
            if pca_trajs[pi] is None: continue
            traj = pca_trajs[pi][:N_PRE_OSC_FRAMES]  # only pre-osc
            sm = stance_masks[pi] if pi < len(stance_masks) else None
            if sm is not None and sm.sum() >= 10:
                traj = traj[sm]
            valid = ~np.any(np.isnan(traj), axis=1)
            if valid.sum() < 10: continue
            stance_trajs.append(traj[valid])
        if len(stance_trajs) >= 3:
            jpca_result = fit_jpca(stance_trajs, dt=1.0 / SAMPLE_HZ)
            jpca_x_mean = jpca_result["X_mean"]
            if jpca_result["rotation_planes"]:
                jpca_plane = jpca_result["rotation_planes"][0]
    except Exception as e:
        if verbose: print(f"    jPCA failed: {e}")

    # === Phase at each pitcher event, four coordinates ===
    methods_events = {}
    for method in ("pelvis_x", "bat_axis_angle", "pca_pc1pc2", "jpca"):
        for evname in ("windup_onset", "knee_high", "foot_landing", "release"):
            methods_events[(method, evname)] = []
    for pi, p in enumerate(pitches):
        for evname in ("windup_onset", "knee_high", "foot_landing", "release"):
            idx = p[f"idx_{evname}"]
            if idx is None:
                methods_events[("pelvis_x", evname)].append(np.nan)
                methods_events[("bat_axis_angle", evname)].append(np.nan)
                methods_events[("pca_pc1pc2", evname)].append(np.nan)
                methods_events[("jpca", evname)].append(np.nan)
                continue
            sig = p["joint_mat"][:, pelvis_x_idx]
            methods_events[("pelvis_x", evname)].append(phase_at_event(sig, SAMPLE_HZ, idx))
            head, handle = p["bat_head"], p["bat_handle"]
            valid = ~(np.any(np.isnan(head), axis=1) | np.any(np.isnan(handle), axis=1))
            if valid.sum() < N_PRE_OSC_FRAMES * 0.5:
                methods_events[("bat_axis_angle", evname)].append(np.nan)
            else:
                axis = head - handle
                bang = np.arctan2(axis[:, 1], axis[:, 2])
                methods_events[("bat_axis_angle", evname)].append(phase_at_event(bang, SAMPLE_HZ, idx))
            methods_events[("pca_pc1pc2", evname)].append(pca_phase_at_event(pca_trajs[pi], idx))
            # jPCA phase: project the event-frame's PC vector onto rotation plane
            if jpca_plane is not None and pca_trajs[pi] is not None:
                v = pca_trajs[pi][idx]
                if np.any(np.isnan(v)):
                    methods_events[("jpca", evname)].append(np.nan)
                else:
                    methods_events[("jpca", evname)].append(
                        float(jpca_phase(v[None, :], jpca_plane, jpca_x_mean)[0]))
            else:
                methods_events[("jpca", evname)].append(np.nan)

    def circ_R(phases):
        ph = np.array([p for p in phases if not np.isnan(p)])
        if len(ph) == 0: return np.nan, 0
        return float(np.abs(np.mean(np.exp(1j * ph)))), len(ph)

    # Permutation null test for PCA-based phase at each event
    rng = np.random.default_rng(42)
    n_perm = 500
    null_R = {}
    for evname in ("windup_onset", "knee_high", "foot_landing", "release"):
        # Real: PC1×PC2 angle at the actual event index per pitch
        real_phases = methods_events[("pca_pc1pc2", evname)]
        # Null: pick random valid index in each pitch's pca traj
        nulls = []
        for _ in range(n_perm):
            null_phases = []
            for pi, p in enumerate(pitches):
                t = pca_trajs[pi]
                if t is None: continue
                # pick random index in the same time window (between windup_onset and idx_release)
                lo = p["idx_windup_onset"] or 0
                hi = p["idx_release"] or len(t) - 1
                if hi <= lo + 1: continue
                rand_idx = rng.integers(lo, hi)
                pc1, pc2 = t[rand_idx, 0], t[rand_idx, 1]
                if not (np.isnan(pc1) or np.isnan(pc2)):
                    null_phases.append(np.arctan2(pc2, pc1))
            if null_phases:
                nulls.append(np.abs(np.mean(np.exp(1j * np.array(null_phases)))))
        null_R[evname] = nulls

    out = {
        "batter_id": batter_id,
        "n_pitches_total": len(pitches_meta),
        "n_pitches_usable": n_usable,
        "joint_stds": joint_stds,
        "joint_stds_mean": joint_stds.mean(axis=0),
        "joint_stds_per_pitch": joint_stds.sum(axis=1),  # whole-body movement per pitch
        "pelvis_x_freqs": np.array(pelvis_freqs),
        "hand_rt_y_freqs": np.array(hand_freqs),
        "bat_angle_freqs": np.array(bat_freqs),
        "pca_var_explained": var_explained[:10],
        "pca_components": components,  # for mode-shape interpretation
        "stance_frac": float(stance_frac),
        "jpca": ({
            "M": jpca_result["M"],
            "eigenvalues": jpca_result["eigenvalues"],
            "rotation_planes": jpca_result["rotation_planes"],
            "x_mean": jpca_result["X_mean"],
        } if jpca_result is not None else None),
        "per_pitch_mean_pose": {  # play_id -> stance-frames mean pose (66-d)
            p["play_id"]: per_pitch_mean_pose[pi]
            for pi, p in enumerate(pitches) if per_pitch_mean_pose[pi] is not None
        },
        "per_pitch_stance_start_idx": {  # play_id -> first in-stance frame index (in pre-osc)
            p["play_id"]: int(stance_starts[pi])
            for pi, p in enumerate(pitches)
        },
        "phase_R": {  # (method, event) -> R value + n
            k: circ_R(v) for k, v in methods_events.items()
        },
        "phase_null_distributions": null_R,
        "n_pitches_swing": sum(1 for p in pitches if p["result_call"] in ("S", "X", "F")),
        "n_pitches_take": sum(1 for p in pitches if p["result_call"] in ("B", "C", "*B")),
        "pitches_meta": [{
            "play_id": p["play_id"], "pitch_type": p["pitch_type"],
            "result_call": p["result_call"], "pitcher_id": p["pitcher_id"],
            "game_pk": p["game_pk"], "start_speed": p["start_speed"],
            "batter_side": p["batter_side"], "pitcher_throws": p["pitcher_throws"],
        } for p in pitches],
        "phases_per_pitch": {
            k: list(v) for k, v in methods_events.items()
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR", "data"))
    ap.add_argument("--min-pitches", type=int, default=10)
    ap.add_argument("--output", type=str,
                    default="data/oscillation_report/pre_pitch_preparatory_movement/census.pkl")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit-batters", type=int, default=0,
                    help="for debugging, only process N batters")
    args = ap.parse_args()

    print("=== discovering batters across games ===")
    by_batter = get_batter_pitches_across_games(Path(args.data_dir))
    counts = sorted([(bid, len(pl)) for bid, pl in by_batter.items()],
                    key=lambda x: -x[1])
    candidates = [(bid, n) for bid, n in counts if n >= args.min_pitches]
    print(f"  {len(by_batter)} unique batters total")
    print(f"  {len(candidates)} batters with >= {args.min_pitches} pitches")
    print(f"  total pitches across candidates: {sum(n for _, n in candidates)}")

    if args.limit_batters:
        candidates = candidates[:args.limit_batters]

    results = {}
    name_lookup = {}
    side_lookup = {}
    t0 = time.time()
    for ci, (key, n) in enumerate(candidates):
        # key is (mlb_player_id, side)
        bid, side = key
        base_name = get_player_name(bid)
        # Per-side display name. Check if this player appears with both sides in the corpus
        is_switch = sum(1 for (b, s), _ in counts if b == bid) > 1
        display_name = f"{base_name} ({side})" if is_switch else base_name
        # Use composite string key for results / names (pickle-able and stable)
        composite_key = f"{bid}_{side}"
        name_lookup[composite_key] = display_name
        side_lookup[composite_key] = side
        print(f"\n[{ci+1}/{len(candidates)}] {display_name} ({bid}_{side})  n_pitches={n}")
        try:
            r = analyze_batter(bid, by_batter[key], verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        if r is None:
            print(f"  -> too few usable pitches")
            continue
        r["mlb_id"] = bid
        r["batter_side"] = side
        r["is_switch_hitter"] = is_switch
        results[composite_key] = r
        # Quick R summary
        for ev in ("windup_onset", "knee_high"):
            R, nn = r["phase_R"][("pca_pc1pc2", ev)]
            null = r["phase_null_distributions"][ev]
            null_arr = np.array(null) if null else np.array([])
            p_val = (np.sum(null_arr >= R) + 1) / (len(null_arr) + 1) if len(null_arr) > 0 else np.nan
            print(f"  PCA phase @ {ev:14s}  R={R:.2f} (n={nn})  null95={np.percentile(null_arr, 95) if len(null_arr) else np.nan:.2f}  p={p_val:.3f}")
        elapsed = time.time() - t0
        eta = elapsed / (ci + 1) * (len(candidates) - ci - 1)
        print(f"  elapsed {elapsed:.0f}s, eta {eta:.0f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "names": name_lookup,
                     "sides": side_lookup,
                     "all_batter_counts": {f"{b}_{s}": n for ((b, s), n) in counts}}, f)
    print(f"\n=== wrote census: {out_path}  ({len(results)} batters) ===")


if __name__ == "__main__":
    main()
