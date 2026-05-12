"""Comprehensive pre-pitch oscillation analysis for ONE batter.

Designed as an exploratory dossier: leaves no joint, no axis, no obvious
spectral or postural feature un-touched. Outputs several PNG figures plus
a CSV of per-pitch features and a console summary.

Sections produced:
    1. Per-joint movement summary    (heatmap pitch × joint, variance + range)
    2. Per-joint dominant FFT freq   (which joints oscillate, at what Hz)
    3. Bat behavior                  (handle/head trajectories, bat axis angle FFT)
    4. PCA postural state space      (variance explained + PC1×PC2 trajectories)
    5. Phase-portrait                (PC1 vs dPC1/dt for one example pitch)
    6. Phase at pitcher events       (at windup_onset, knee_high, foot_landing, release)
                                       three phase coordinates: pelvis-based, bat-angle-based, PC1×PC2-based
    7. Phase × outcome               (rose histograms split swing/take)
    8. Per-pitch feature CSV         (so you can correlate any feature with anything later)

Usage:
    python scripts/analyze_batter_oscillation.py --game 823141 --batter-id 677594
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
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import circmean, circstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.parquet_readers import open_game
from fieldvision.skeleton import SKELETON_CONNECTIONS
from fieldvision.storage import JOINT_COLS
from fieldvision.pitch_kinematics import detect_pitcher_events

JOINT_BIDS = [bid for bid, _ in JOINT_COLS]
JOINT_NAMES = {bid: name for bid, name in JOINT_COLS}

PRE_OSC_SECONDS = 5.0
SAMPLE_HZ = 30.0
N_PRE_OSC_FRAMES = int(PRE_OSC_SECONDS * SAMPLE_HZ)


# ────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────


def load_pitch_data(conn, batter_id, play_id, pitcher_id, release_t):
    """Returns dict with all data for one pitch, or None if not usable."""
    select_cols = "time_unix, " + ", ".join(f"{n}_x, {n}_y, {n}_z" for _, n in JOINT_COLS)

    # Pitcher frames for kinematic event detection (need 5s back from release)
    p_rows = conn.execute(
        f"SELECT {select_cols} FROM actor_frame WHERE mlb_player_id=? "
        "AND time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (pitcher_id, release_t - 5, release_t + 0.5),
    ).fetchall()
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
    win_hi = release_t + 0.2  # extend through release for phase analysis at later events

    # Batter frames over extended window
    b_rows = conn.execute(
        f"SELECT {select_cols} FROM actor_frame WHERE mlb_player_id=? "
        "AND time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (batter_id, win_lo, win_hi),
    ).fetchall()
    if len(b_rows) < 100:
        return None

    times = np.array([r[0] for r in b_rows])
    # Build (n_frames, 60) joint matrix; missing values → NaN
    joint_mat = np.full((len(b_rows), len(JOINT_BIDS), 3), np.nan)
    for i, r in enumerate(b_rows):
        for j, bid in enumerate(JOINT_BIDS):
            x, y, z = r[1 + j*3], r[2 + j*3], r[3 + j*3]
            if x is not None:
                joint_mat[i, j] = (x, y, z)

    # Bat trajectory
    bat_rows = conn.execute(
        "SELECT time_unix, head_x, head_y, head_z, handle_x, handle_y, handle_z "
        "FROM bat_frame WHERE time_unix BETWEEN ? AND ? ORDER BY time_unix",
        (win_lo, win_hi),
    ).fetchall()
    bat_times = np.array([r[0] for r in bat_rows]) if bat_rows else np.array([])
    bat_head = np.array([(r[1], r[2], r[3]) for r in bat_rows]) if bat_rows else np.empty((0, 3))
    bat_handle = np.array([(r[4], r[5], r[6]) for r in bat_rows]) if bat_rows else np.empty((0, 3))

    # Resample to uniform 30 Hz grid over [win_lo, win_hi]
    t_grid = np.arange(win_lo, win_hi, 1.0 / SAMPLE_HZ)
    n_grid = len(t_grid)

    def resample(times_in, vals_in):
        """Linear-interpolate vals_in (n_in, k) at t_grid timestamps; NaN → mean if needed."""
        if len(times_in) == 0:
            return np.full((n_grid, vals_in.shape[1] if vals_in.ndim > 1 else 1), np.nan)
        if vals_in.ndim == 1:
            return np.interp(t_grid, times_in, vals_in)
        out = np.empty((n_grid, vals_in.shape[1]))
        for k in range(vals_in.shape[1]):
            v = vals_in[:, k]
            if np.all(np.isnan(v)):
                out[:, k] = np.nan
                continue
            # Drop NaNs
            mask = ~np.isnan(v)
            out[:, k] = np.interp(t_grid, times_in[mask], v[mask])
        return out

    joint_flat = joint_mat.reshape(len(b_rows), -1)  # (n, 60)
    joint_resampled = resample(times, joint_flat)    # (n_grid, 60)

    bat_head_resampled = resample(bat_times, bat_head) if len(bat_times) else np.full((n_grid, 3), np.nan)
    bat_handle_resampled = resample(bat_times, bat_handle) if len(bat_times) else np.full((n_grid, 3), np.nan)

    # Find indices for each event in the resampled grid
    def t_to_idx(t):
        if t is None: return None
        return int(np.clip((t - win_lo) * SAMPLE_HZ, 0, n_grid - 1))

    return {
        "play_id": play_id,
        "release_t": release_t,
        "win_lo": win_lo,
        "win_hi": win_hi,
        "t_grid": t_grid,
        "joint_mat": joint_resampled,                # (n_grid, 60)
        "bat_head": bat_head_resampled,              # (n_grid, 3)
        "bat_handle": bat_handle_resampled,
        "windup_onset_t": onset_t,
        "knee_high_t": ev.knee_high_t,
        "foot_landing_t": ev.foot_landing_t,
        "front_leg_side": ev.front_leg_side,
        "idx_windup_onset": t_to_idx(onset_t),
        "idx_knee_high": t_to_idx(ev.knee_high_t),
        "idx_foot_landing": t_to_idx(ev.foot_landing_t),
        "idx_release": t_to_idx(release_t),
    }


def load_all_pitches(game_pk, data_dir, batter_id):
    conn = open_game(game_pk, data_dir)
    rows = conn.execute(
        "SELECT play_id, pitch_type, result_call, pitcher_id, start_time_unix "
        "FROM pitch_label WHERE batter_id=? AND start_time_unix IS NOT NULL ORDER BY start_time_unix",
        (batter_id,),
    ).fetchall()
    pitches = []
    for play_id, ptype, call, pitcher_id, t_release in rows:
        d = load_pitch_data(conn, batter_id, play_id, pitcher_id, t_release)
        if d is None: continue
        d.update(pitch_type=ptype, result_call=call, pitcher_id=pitcher_id, batter_id=batter_id)
        pitches.append(d)
    conn.close()
    return pitches


# ────────────────────────────────────────────────────────────
# Analyses
# ────────────────────────────────────────────────────────────


def joint_movement_stats(pitches):
    """For each pitch × each joint, compute the per-axis stddev over the
    pre-osc window (frames 0..N_PRE_OSC_FRAMES). Returns (n_pitches, n_joints) array of stddev magnitude."""
    n_p = len(pitches)
    n_j = len(JOINT_BIDS)
    stds = np.zeros((n_p, n_j))
    for pi, p in enumerate(pitches):
        # use first N_PRE_OSC_FRAMES (the pure pre-osc period)
        seg = p["joint_mat"][: N_PRE_OSC_FRAMES].reshape(N_PRE_OSC_FRAMES, n_j, 3)
        for ji in range(n_j):
            stds[pi, ji] = np.linalg.norm(np.nanstd(seg[:, ji, :], axis=0))
    return stds


def joint_dominant_frequency(pitches, target_axis="all"):
    """Per pitch × per joint coordinate (60 dims), compute dominant freq via FFT
    over the pre-osc window. Returns (n_pitches, 60) freq array + (n_pitches, 60) power."""
    n_p = len(pitches)
    n_dim = 60
    freqs = np.full((n_p, n_dim), np.nan)
    powers = np.full((n_p, n_dim), np.nan)
    for pi, p in enumerate(pitches):
        seg = p["joint_mat"][: N_PRE_OSC_FRAMES]  # (N, 60)
        for di in range(n_dim):
            x = seg[:, di]
            if np.any(np.isnan(x)):
                continue
            # detrend + window
            x = x - x.mean()
            x = x * np.hanning(len(x))
            X = np.abs(np.fft.rfft(x))
            f = np.fft.rfftfreq(len(x), 1 / SAMPLE_HZ)
            # ignore DC + frequencies outside 0.2–4 Hz (typical batter waggle range)
            mask = (f > 0.2) & (f < 4.0)
            if mask.sum() == 0: continue
            best_idx = np.argmax(X[mask])
            freqs[pi, di] = f[mask][best_idx]
            powers[pi, di] = X[mask][best_idx]
    return freqs, powers


def bat_features(pitches):
    """Compute per-pitch bat metrics over the pre-osc window."""
    out = []
    for p in pitches:
        seg = p["bat_handle"][: N_PRE_OSC_FRAMES]
        head = p["bat_head"][: N_PRE_OSC_FRAMES]
        handle_speed = np.linalg.norm(np.diff(seg, axis=0), axis=1) * SAMPLE_HZ
        head_speed = np.linalg.norm(np.diff(head, axis=0), axis=1) * SAMPLE_HZ
        bat_axis = head - seg
        bat_len = np.linalg.norm(bat_axis, axis=1)
        # bat angle in side view (Z, Y plane): atan2(Y_diff, Z_diff)
        bat_angle = np.arctan2(bat_axis[:, 1], bat_axis[:, 2])
        out.append({
            "handle_speed_mean": float(np.nanmean(handle_speed)),
            "handle_speed_max": float(np.nanmax(handle_speed)),
            "head_speed_mean": float(np.nanmean(head_speed)),
            "head_speed_max": float(np.nanmax(head_speed)),
            "bat_len_mean": float(np.nanmean(bat_len)),
            "bat_angle_var": float(np.nanvar(bat_angle)),
            "bat_angle_range": float(np.nanmax(bat_angle) - np.nanmin(bat_angle)),
        })
    return out


def fit_pca_postural(pitches):
    """Stack all pitches' pre-osc windows. Per-pitch mean-center the joint vector
    (so we're modeling MOTION around each pitch's pose, not absolute pose differences).
    Then SVD → PC1..PC5."""
    blocks = []
    for p in pitches:
        seg = p["joint_mat"][: N_PRE_OSC_FRAMES]  # (N, 60)
        valid = ~np.any(np.isnan(seg), axis=1)
        if valid.sum() < N_PRE_OSC_FRAMES * 0.8:
            continue
        seg_centered = seg[valid] - np.nanmean(seg[valid], axis=0)
        blocks.append(seg_centered)
    if not blocks:
        return None
    X = np.vstack(blocks)  # (total_frames, 60)
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_explained = S ** 2 / np.sum(S ** 2)
    # Project EVERY pitch's full window (incl. post-onset) into PC space
    components = Vt[:5]  # (5, 60)
    pca_trajs = []
    for p in pitches:
        seg = p["joint_mat"]  # (n_grid, 60)
        valid = ~np.any(np.isnan(seg), axis=1)
        # Center using the same per-pitch pre-osc mean as in the SVD prep
        pre = seg[: N_PRE_OSC_FRAMES]
        pre_valid = ~np.any(np.isnan(pre), axis=1)
        if pre_valid.sum() < N_PRE_OSC_FRAMES * 0.5:
            pca_trajs.append(None)
            continue
        mean_pose = np.nanmean(pre[pre_valid], axis=0)
        centered = seg - mean_pose
        traj = np.full((seg.shape[0], 5), np.nan)
        traj[valid] = centered[valid] @ components.T
        pca_trajs.append(traj)
    return {
        "var_explained": var_explained,
        "components": Vt,
        "pca_trajs": pca_trajs,
    }


def phase_at_index(signal, sample_hz, idx, dom_freq=None, bandwidth=0.6):
    """Filter signal narrowly around dom_freq, Hilbert-transform, return phase at idx (radians).
    If dom_freq is None, find it from FFT of pre-osc window."""
    sig = signal[: N_PRE_OSC_FRAMES]
    sig = sig - np.nanmean(sig)
    if np.any(np.isnan(sig)):
        return np.nan
    if dom_freq is None:
        sig_w = sig * np.hanning(len(sig))
        X = np.abs(np.fft.rfft(sig_w))
        f = np.fft.rfftfreq(len(sig_w), 1 / sample_hz)
        mask = (f > 0.2) & (f < 4.0)
        if mask.sum() == 0: return np.nan
        dom_freq = f[mask][np.argmax(X[mask])]
    # Filter on the WHOLE signal (so we have phase at later indices too)
    full_sig = signal - np.nanmean(signal[: N_PRE_OSC_FRAMES])
    nyq = sample_hz / 2
    low = max(0.05, (dom_freq - bandwidth / 2) / nyq)
    high = min(0.95, (dom_freq + bandwidth / 2) / nyq)
    if low >= high:
        return np.nan
    b, a = butter(2, [low, high], btype="band")
    if np.any(np.isnan(full_sig)):
        # Fill NaN with linear interp
        nans = np.isnan(full_sig)
        if nans.all(): return np.nan
        ix = np.arange(len(full_sig))
        full_sig = full_sig.copy()
        full_sig[nans] = np.interp(ix[nans], ix[~nans], full_sig[~nans])
    try:
        filtered = filtfilt(b, a, full_sig)
    except Exception:
        return np.nan
    analytic = hilbert(filtered)
    if idx >= len(analytic): idx = len(analytic) - 1
    return float(np.angle(analytic[idx]))


def pca_phase_at_index(traj, idx):
    """Use PC1×PC2 plane to compute phase = atan2(PC2, PC1) at idx."""
    if traj is None: return np.nan
    if idx >= len(traj): idx = len(traj) - 1
    pc1, pc2 = traj[idx, 0], traj[idx, 1]
    if np.isnan(pc1) or np.isnan(pc2): return np.nan
    return float(np.arctan2(pc2, pc1))


# ────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────


def plot_joint_movement_heatmap(stds, pitches, out_path):
    n_p, n_j = stds.shape
    fig, ax = plt.subplots(figsize=(12, max(4, n_p * 0.32)), facecolor="#1a1a1a")
    im = ax.imshow(stds, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(n_p))
    ax.set_yticklabels([f"#{i+1} {p['pitch_type'] or '?'}/{p['result_call'] or '?'}" for i, p in enumerate(pitches)],
                       color="white", fontsize=8)
    ax.set_xticks(range(n_j))
    ax.set_xticklabels([JOINT_NAMES[bid] for bid in JOINT_BIDS], rotation=60, ha="right",
                       color="white", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.set_label("3D stddev (ft)", color="white", fontsize=8)
    ax.set_facecolor("#1a1a1a")
    ax.set_title(f"Per-joint movement (3D stddev) over 5s pre-pitch window — {n_p} pitches",
                 color="white", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def plot_dominant_frequency(freqs, powers, out_path):
    """Bar chart: per-joint mean dominant frequency (averaged across pitches), with error bars."""
    n_dim = freqs.shape[1]
    # Pick the y-axis (vertical) component for each joint, since vertical bobbing is the
    # most likely periodic signal. Index 1 of every 3-tuple.
    axis_freqs = freqs[:, 1::3]  # (n_pitches, 20 joints)
    axis_powers = powers[:, 1::3]
    mean_f = np.nanmean(axis_freqs, axis=0)
    std_f = np.nanstd(axis_freqs, axis=0)
    mean_p = np.nanmean(axis_powers, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), facecolor="#1a1a1a")
    names = [JOINT_NAMES[bid] for bid in JOINT_BIDS]
    x = np.arange(len(names))
    ax1.bar(x, mean_f, yerr=std_f, color="#4caf50", alpha=0.85, capsize=3,
            error_kw=dict(ecolor="#aaa", lw=0.8))
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=55, ha="right", color="white", fontsize=8)
    ax1.set_facecolor("#1a1a1a")
    ax1.tick_params(colors="white")
    ax1.set_ylabel("Dominant frequency Y-axis (Hz)", color="white")
    ax1.set_title("Per-joint dominant oscillation frequency in 5s pre-pitch window (Y-axis only)",
                  color="white", fontsize=11)
    ax1.set_ylim(0, 4)
    for s in ax1.spines.values(): s.set_color("#666")

    ax2.bar(x, mean_p, color="#ff9800", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=55, ha="right", color="white", fontsize=8)
    ax2.set_facecolor("#1a1a1a")
    ax2.tick_params(colors="white")
    ax2.set_ylabel("Mean spectral power at dominant freq", color="white")
    for s in ax2.spines.values(): s.set_color("#666")

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def plot_bat_features(pitches, bat_feats, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="#1a1a1a")
    metrics = [
        ("handle_speed_mean", "Bat handle mean speed (ft/s)"),
        ("head_speed_mean", "Bat head (tip) mean speed (ft/s)"),
        ("head_speed_max", "Bat head peak speed (ft/s)"),
        ("bat_angle_var", "Bat angle variance (rad²)"),
        ("bat_angle_range", "Bat angle range (rad)"),
        ("bat_len_mean", "Bat length (ft)"),
    ]
    for ax, (key, label) in zip(axes.flat, metrics):
        vals = np.array([f[key] for f in bat_feats])
        labels = [f"#{i+1}" for i in range(len(vals))]
        # color by swing/take
        colors = ["#f44336" if pitches[i]["result_call"] in ("S","X","F") else "#42a5f5"
                  for i in range(len(vals))]
        ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_title(label, color="white", fontsize=9)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white", labelsize=7)
        for s in ax.spines.values(): s.set_color("#444")
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=6)
    fig.suptitle("Per-pitch bat metrics (red=swing, blue=take)", color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def plot_pca_state_space(pitches, pca_result, out_path):
    fig = plt.figure(figsize=(15, 10), facecolor="#1a1a1a")
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # Variance explained
    ax_var = fig.add_subplot(gs[0, 0])
    ve = pca_result["var_explained"][:10] * 100
    ax_var.bar(range(1, len(ve) + 1), ve, color="#4caf50")
    ax_var.set_xlabel("Principal component", color="white")
    ax_var.set_ylabel("Variance explained (%)", color="white")
    ax_var.set_facecolor("#1a1a1a")
    ax_var.tick_params(colors="white")
    ax_var.set_title("PCA variance explained", color="white", fontsize=10)
    for s in ax_var.spines.values(): s.set_color("#666")

    # PC1×PC2 trajectory for one example pitch
    ax_traj = fig.add_subplot(gs[0, 1:])
    example_idx = 0
    for i, traj in enumerate(pca_result["pca_trajs"]):
        if traj is not None:
            example_idx = i; break
    traj = pca_result["pca_trajs"][example_idx]
    p_data = pitches[example_idx]
    pre_traj = traj[: N_PRE_OSC_FRAMES]
    valid = ~np.any(np.isnan(pre_traj), axis=1)
    sc = ax_traj.scatter(pre_traj[valid, 0], pre_traj[valid, 1],
                         c=np.arange(N_PRE_OSC_FRAMES)[valid], cmap="viridis", s=14, alpha=0.85)
    ax_traj.plot(pre_traj[valid, 0], pre_traj[valid, 1], "-", color="#888", lw=0.8, alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax_traj)
    cbar.set_label("frame # (older → newer)", color="white", fontsize=8)
    cbar.ax.tick_params(colors="white", labelsize=7)
    # Mark event points (only windup_onset is in pre-osc window)
    if p_data["idx_windup_onset"] is not None and p_data["idx_windup_onset"] < N_PRE_OSC_FRAMES:
        idx = p_data["idx_windup_onset"]
        if not np.isnan(pre_traj[idx]).any():
            ax_traj.scatter([pre_traj[idx, 0]], [pre_traj[idx, 1]],
                            s=120, c="#ff9800", marker="*", label="WINDUP_ONSET", edgecolor="white", lw=1.5)
    ax_traj.set_xlabel("PC1", color="white"); ax_traj.set_ylabel("PC2", color="white")
    ax_traj.set_facecolor("#1a1a1a"); ax_traj.tick_params(colors="white")
    ax_traj.set_title(f"Postural state-space trajectory for pitch #{example_idx+1} ({p_data['pitch_type']}/{p_data['result_call']})",
                      color="white", fontsize=10)
    ax_traj.legend(loc="best", fontsize=8)
    for s in ax_traj.spines.values(): s.set_color("#666")
    ax_traj.set_aspect("equal", "datalim")

    # All-pitches trajectories overlay
    ax_all = fig.add_subplot(gs[1, :])
    for i, (traj, p) in enumerate(zip(pca_result["pca_trajs"], pitches)):
        if traj is None: continue
        pre = traj[: N_PRE_OSC_FRAMES]
        valid = ~np.any(np.isnan(pre), axis=1)
        color = "#f44336" if p["result_call"] in ("S","X","F") else "#42a5f5"
        ax_all.plot(pre[valid, 0], pre[valid, 1], "-", color=color, lw=0.8, alpha=0.5)
        if valid.any():
            ax_all.scatter([pre[valid, 0][0]], [pre[valid, 1][0]],
                           s=20, c=color, alpha=0.7, edgecolor="white", lw=0.4)
    ax_all.set_xlabel("PC1", color="white"); ax_all.set_ylabel("PC2", color="white")
    ax_all.set_facecolor("#1a1a1a"); ax_all.tick_params(colors="white")
    ax_all.set_title("All pitches' postural trajectories overlaid (red = swing, blue = take)",
                     color="white", fontsize=10)
    for s in ax_all.spines.values(): s.set_color("#666")
    ax_all.set_aspect("equal", "datalim")

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def rose_plot(ax, phases, n_bins=12, color="#4caf50", title="", show_R=True):
    """Plot rose histogram of phases (radians, in [-pi, pi])."""
    phases = phases[~np.isnan(phases)]
    if len(phases) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color="white", transform=ax.transAxes)
        ax.set_title(title, color="white", fontsize=9)
        return
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(phases, bins=bins)
    widths = np.diff(bins)
    centers = bins[:-1] + widths / 2
    ax.bar(centers, counts, width=widths, bottom=0, color=color, alpha=0.75, edgecolor="white", lw=0.5)
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white", labelsize=7)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    if show_R:
        # Mean resultant length R: how concentrated the distribution is (0=uniform, 1=all same angle)
        R = np.abs(np.mean(np.exp(1j * phases)))
        mean_ang = circmean(phases, high=np.pi, low=-np.pi)
        title = f"{title}\nn={len(phases)}, R={R:.2f}, μ={np.degrees(mean_ang):.0f}°"
    ax.set_title(title, color="white", fontsize=9, pad=12)


def plot_phase_at_events(pitches, pca_result, out_path):
    """For each of 4 pitcher events, compute batter phase via 3 different signals,
    and plot rose histograms."""
    events = ["windup_onset", "knee_high", "foot_landing", "release"]
    # Three phase coordinates: pelvis_x (Hilbert), bat_axis_angle (Hilbert), PC1xPC2 angle
    phase_methods = ["pelvis_x", "bat_axis_angle", "pca_pc1pc2"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11), facecolor="#1a1a1a",
                             subplot_kw=dict(projection="polar"))

    for mi, method in enumerate(phase_methods):
        for ei, evname in enumerate(events):
            phases = []
            for pi, p in enumerate(pitches):
                idx = p[f"idx_{evname}"]
                if idx is None: phases.append(np.nan); continue
                if method == "pelvis_x":
                    sig = p["joint_mat"][:, 0]  # pelvis x = first dim
                    phases.append(phase_at_index(sig, SAMPLE_HZ, idx))
                elif method == "bat_axis_angle":
                    head = p["bat_head"]; handle = p["bat_handle"]
                    valid = ~(np.any(np.isnan(head), axis=1) | np.any(np.isnan(handle), axis=1))
                    if valid.sum() < N_PRE_OSC_FRAMES * 0.5:
                        phases.append(np.nan); continue
                    axis = head - handle
                    bat_angle = np.arctan2(axis[:, 1], axis[:, 2])
                    phases.append(phase_at_index(bat_angle, SAMPLE_HZ, idx))
                elif method == "pca_pc1pc2":
                    traj = pca_result["pca_trajs"][pi]
                    phases.append(pca_phase_at_index(traj, idx))
            phases = np.array(phases)
            color = ["#4caf50", "#ff9800", "#e91e63", "#9c27b0"][ei]
            rose_plot(axes[mi, ei], phases, n_bins=12, color=color,
                      title=f"{method}\n{evname}")

    fig.suptitle(f"Batter oscillation PHASE at each pitcher delivery event\n"
                 f"3 phase coordinates × 4 events. R near 1 = phase-locked, R near 0 = phase-random",
                 color="white", fontsize=12, y=0.99)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#1a1a1a")
    plt.close(fig)


def write_feature_csv(pitches, bat_feats, joint_stds, out_path):
    fields = ["pitch_idx", "play_id", "pitch_type", "result_call", "is_swing",
              "windup_onset_t_rel", "knee_high_t_rel", "foot_landing_t_rel",
              "front_leg_side"]
    fields += [f"std_{JOINT_NAMES[bid]}" for bid in JOINT_BIDS]
    fields += list(bat_feats[0].keys()) if bat_feats else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, p in enumerate(pitches):
            row = {
                "pitch_idx": i + 1,
                "play_id": p["play_id"],
                "pitch_type": p["pitch_type"],
                "result_call": p["result_call"],
                "is_swing": int(p["result_call"] in ("S","X","F")),
                "windup_onset_t_rel": p["windup_onset_t"] - p["release_t"],
                "knee_high_t_rel": p["knee_high_t"] - p["release_t"] if p["knee_high_t"] else "",
                "foot_landing_t_rel": p["foot_landing_t"] - p["release_t"] if p["foot_landing_t"] else "",
                "front_leg_side": p["front_leg_side"] or "",
            }
            for ji, bid in enumerate(JOINT_BIDS):
                row[f"std_{JOINT_NAMES[bid]}"] = float(joint_stds[i, ji])
            row.update(bat_feats[i])
            w.writerow(row)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=823141)
    ap.add_argument("--batter-id", type=int, required=True)
    ap.add_argument("--data-dir", default=os.environ.get("FV_DATA_DIR", "data"))
    ap.add_argument("--out-prefix", type=str, default=None,
                    help="prefix for output files (default: data/oscillation_<batter_id>)")
    args = ap.parse_args()

    pitches = load_all_pitches(args.game, Path(args.data_dir), args.batter_id)
    print(f"loaded {len(pitches)} usable pitches for batter {args.batter_id}")
    if not pitches:
        raise SystemExit("no pitches")

    prefix = args.out_prefix or f"data/oscillation_{args.batter_id}"

    print("\n=== 1. joint movement summary (3D stddev per pitch × per joint, pre-osc window) ===")
    joint_stds = joint_movement_stats(pitches)
    plot_joint_movement_heatmap(joint_stds, pitches, f"{prefix}_1_joint_heatmap.png")
    # Per-joint mean across pitches (rank movement)
    mean_per_joint = joint_stds.mean(axis=0)
    rank = np.argsort(-mean_per_joint)
    print("  Top 10 most-active joints (mean 3D stddev across pitches):")
    for r in rank[:10]:
        print(f"    {JOINT_NAMES[JOINT_BIDS[r]]:14s}  mean_std={mean_per_joint[r]:.3f}ft  "
              f"per-pitch range=[{joint_stds[:,r].min():.3f}, {joint_stds[:,r].max():.3f}]")

    print("\n=== 2. dominant frequency per joint coordinate (FFT 0.2-4 Hz) ===")
    freqs, powers = joint_dominant_frequency(pitches)
    plot_dominant_frequency(freqs, powers, f"{prefix}_2_freqs.png")
    # Print summary for key joints (Y-axis = vertical bobbing)
    print("  Y-axis dominant frequency, mean across pitches:")
    for ji, bid in enumerate(JOINT_BIDS):
        if JOINT_NAMES[bid] in ("pelvis", "torso_b", "head", "hand_lt", "hand_rt", "hipmaster"):
            f_y = freqs[:, ji * 3 + 1]
            print(f"    {JOINT_NAMES[bid]:12s}  Y-freq mean={np.nanmean(f_y):.2f}Hz  std={np.nanstd(f_y):.2f}")

    print("\n=== 3. bat features ===")
    bat_feats = bat_features(pitches)
    plot_bat_features(pitches, bat_feats, f"{prefix}_3_bat.png")
    head_speeds = [b["head_speed_mean"] for b in bat_feats]
    print(f"  bat_head_speed_mean across pitches: median={np.median(head_speeds):.2f}ft/s "
          f"min={min(head_speeds):.2f} max={max(head_speeds):.2f}")
    angle_vars = [b["bat_angle_var"] for b in bat_feats]
    print(f"  bat_angle_var across pitches: median={np.median(angle_vars):.4f} "
          f"min={min(angle_vars):.4f} max={max(angle_vars):.4f}")

    print("\n=== 4. PCA postural state-space ===")
    pca_result = fit_pca_postural(pitches)
    if pca_result is None:
        print("  PCA failed (not enough valid frames)")
    else:
        plot_pca_state_space(pitches, pca_result, f"{prefix}_4_pca.png")
        print(f"  variance explained by first 5 PCs: " +
              ", ".join(f"{v*100:.1f}%" for v in pca_result["var_explained"][:5]))
        cumulative = np.cumsum(pca_result["var_explained"][:10])
        print(f"  cumulative through PC10: {cumulative[-1]*100:.1f}%")

    print("\n=== 5/6/7. phase analysis at each pitcher event ===")
    if pca_result is not None:
        plot_phase_at_events(pitches, pca_result, f"{prefix}_5_phases.png")
        # Print circular concentration R for each (method, event) pair
        events = ["windup_onset", "knee_high", "foot_landing", "release"]
        methods = ["pelvis_x", "bat_axis_angle", "pca_pc1pc2"]
        print(f"  {'method':18s}  " + "  ".join(f"{e[:14]:>14s}" for e in events))
        for method in methods:
            row = []
            for evname in events:
                phases = []
                for pi, p in enumerate(pitches):
                    idx = p[f"idx_{evname}"]
                    if idx is None: continue
                    if method == "pelvis_x":
                        sig = p["joint_mat"][:, 0]
                        phases.append(phase_at_index(sig, SAMPLE_HZ, idx))
                    elif method == "bat_axis_angle":
                        head = p["bat_head"]; handle = p["bat_handle"]
                        valid = ~(np.any(np.isnan(head), axis=1) | np.any(np.isnan(handle), axis=1))
                        if valid.sum() < N_PRE_OSC_FRAMES * 0.5: continue
                        axis = head - handle
                        bat_angle = np.arctan2(axis[:, 1], axis[:, 2])
                        phases.append(phase_at_index(bat_angle, SAMPLE_HZ, idx))
                    elif method == "pca_pc1pc2":
                        traj = pca_result["pca_trajs"][pi]
                        phases.append(pca_phase_at_index(traj, idx))
                phases = np.array([p for p in phases if not np.isnan(p)])
                R = np.abs(np.mean(np.exp(1j * phases))) if len(phases) > 0 else np.nan
                row.append(f"R={R:.2f} (n={len(phases)})")
            print(f"  {method:18s}  " + "  ".join(f"{r:>14s}" for r in row))

    print("\n=== 8. write per-pitch feature CSV ===")
    csv_path = f"{prefix}_features.csv"
    write_feature_csv(pitches, bat_feats, joint_stds, csv_path)
    print(f"  wrote {csv_path}")

    print(f"\nAll outputs:")
    print(f"  {prefix}_1_joint_heatmap.png")
    print(f"  {prefix}_2_freqs.png")
    print(f"  {prefix}_3_bat.png")
    print(f"  {prefix}_4_pca.png")
    print(f"  {prefix}_5_phases.png")
    print(f"  {prefix}_features.csv")


if __name__ == "__main__":
    main()
