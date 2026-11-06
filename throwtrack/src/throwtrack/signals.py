"""Turning sparse, noisy keypoint tracks into dense time series."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .schema import KP, PoseTrack


def interp_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of length <= max_gap along axis 0.

    Runs that are longer, or that touch either end of the series, stay NaN.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    flat = x.reshape(len(x), -1)
    n = len(x)
    idx = np.arange(n)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = np.isnan(col)
        if not bad.any() or bad.all():
            continue
        edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        keep = ((ends - starts) <= max_gap) & (starts > 0) & (ends < n)
        if not keep.any():
            continue
        good = ~bad
        filled = np.interp(idx, idx[good], col[good])
        for s0, e0 in zip(starts[keep], ends[keep]):
            col[s0:e0] = filled[s0:e0]
    return flat.reshape(x.shape)


def rolling_nanmedian(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median that ignores NaNs (edges use a shrunken window)."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x.copy()
    half = window // 2
    padded = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows -> NaN
        return np.nanmedian(view, axis=1)


@dataclass
class DenseTrack:
    """A PoseTrack resampled onto every frame between its first and last detection."""

    track_id: int
    frames: np.ndarray        # (T,) consecutive frame indices
    xy: np.ndarray            # (T, 17, 2) pixel coords, NaN where unknown
    conf: np.ndarray          # (T, 17) confidence, 0 where unknown
    scale: np.ndarray         # (T,) body scale in px (rolling median torso length)

    def index_of(self, frame: float) -> int:
        return int(np.clip(round(frame - self.frames[0]), 0, len(self.frames) - 1))


def torso_length(xy: np.ndarray) -> np.ndarray:
    """Distance between shoulder midpoint and hip midpoint, per frame."""
    sho = 0.5 * (xy[:, KP["left_shoulder"]] + xy[:, KP["right_shoulder"]])
    hip = 0.5 * (xy[:, KP["left_hip"]] + xy[:, KP["right_hip"]])
    return np.linalg.norm(sho - hip, axis=1)


def densify(track: PoseTrack, fps: float, max_gap: int = 6, min_conf: float = 0.2,
            scale_window_s: float = 2.0) -> DenseTrack:
    """Resample a track onto consecutive frames, dropping low-confidence points and
    filling short gaps. Body scale is a rolling median of torso length so that it
    follows the player as they move toward or away from the camera."""
    frames = np.arange(track.start, track.end + 1)
    T = len(frames)
    xy = np.full((T, track.keypoints.shape[1], 2), np.nan)
    conf = np.zeros((T, track.keypoints.shape[1]))
    pos = track.frames - track.start
    kp = track.keypoints
    ok = kp[:, :, 2] >= min_conf
    obs_xy = np.where(ok[..., None], kp[:, :, :2], np.nan)
    xy[pos] = obs_xy
    conf[pos] = np.where(ok, kp[:, :, 2], 0.0)
    xy = interp_short_gaps(xy, max_gap)

    torso = torso_length(xy)
    window = max(3, int(round(scale_window_s * fps)) | 1)
    scale = rolling_nanmedian(torso, window)
    if np.all(np.isnan(scale)):
        # No torso visible at all: fall back to keypoint spread.
        with np.errstate(all="ignore"):
            spread = np.nanmax(xy[:, :, 1], axis=1) - np.nanmin(xy[:, :, 1], axis=1)
        scale = rolling_nanmedian(spread / 3.0, window)
    scale = interp_short_gaps(scale, T)  # fill edges/holes from neighbours
    if np.isnan(scale).any():
        fill = np.nanmedian(scale) if not np.all(np.isnan(scale)) else 1.0
        scale = np.where(np.isnan(scale), fill, scale)
    scale = np.maximum(scale, 1.0)
    return DenseTrack(track.track_id, frames, xy, conf, scale)


def central_velocity(p: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference velocity along axis 0 (units per second); NaN at the ends."""
    v = np.full_like(p, np.nan, dtype=np.float64)
    if len(p) >= 3:
        v[1:-1] = (p[2:] - p[:-2]) * (fps / 2.0)
    return v


def foot_point(xy: np.ndarray) -> np.ndarray:
    """Ground-contact point per frame: ankle midpoint, falling back to a single ankle."""
    la, ra = xy[:, KP["left_ankle"]], xy[:, KP["right_ankle"]]
    mid = 0.5 * (la + ra)
    mid = np.where(np.isnan(mid), la, mid)
    mid = np.where(np.isnan(mid), ra, mid)
    return mid


def stature_px(xy: np.ndarray) -> np.ndarray:
    """Pixel distance from ankle midpoint up to the nose, per frame."""
    return np.linalg.norm(xy[:, KP["nose"]] - foot_point(xy), axis=1)
