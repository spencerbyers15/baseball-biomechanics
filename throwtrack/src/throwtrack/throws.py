"""Throw detection from 2D pose tracks.

A throw is a short, isolated burst of throwing-hand speed relative to the
shoulder. Speeds are measured in body lengths per second (torso lengths), so
the same thresholds work for a player 30 ft or 200 ft from the camera.

Per candidate we compute a handful of interpretable features and turn them
into P(throw) with a logistic scorer. The default weights are hand-set; once
labeled throws exist, ``ThrowScorer.fit`` re-learns them from data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import ARMS, KP, PoseTrack, ThrowEvent
from .signals import DenseTrack, central_velocity, densify, foot_point

FEATURES = ["peak_speed", "isolation", "overhead", "kp_conf"]


@dataclass
class ThrowDetectorConfig:
    min_speed: float = 8.0             # candidate threshold, torso lengths / s
    refractory_s: float = 1.0          # one throw per player per this window
    isolation_window_s: tuple = (0.35, 1.2)
    overhead_window_s: float = 0.3
    max_gap_frames: int = 6
    min_kp_conf: float = 0.2
    report_threshold: float = 0.5      # throws below this are dropped
    review_threshold: float = 0.8      # kept throws below this are flagged for review
    off_arm_penalty: float = 0.5       # multiply confidence of non-dominant-arm throws


@dataclass
class ThrowScorer:
    """Logistic model over candidate features -> P(real throw)."""

    weights: dict = field(default_factory=lambda: {
        # Hand-set starting point; refit with ThrowScorer.fit on labeled throws.
        # Isolation does most of the work: real throws are single bursts
        # (isolation >= ~1.8) while arm circles and jogging are periodic (~1.1).
        # kp_conf carries no weight: motion blur lowers keypoint confidence at
        # release, so it anti-correlates with real throws.
        "bias": -16.8,
        "peak_speed": 0.3,       # per torso-length/s of wrist-vs-shoulder speed
        "isolation": 6.0,        # peak / max speed 0.35-1.2 s either side, clipped to [0, 4]
        "overhead": 2.0,         # max wrist height above shoulder (torso lengths), clipped
        "kp_conf": 0.0,
    })

    @staticmethod
    def _vector(f: dict) -> np.ndarray:
        return np.array([
            1.0,
            f["peak_speed"],
            float(np.clip(f["isolation"], 0.0, 4.0)),
            float(np.clip(f["overhead"], -1.0, 1.0)),
            f["kp_conf"],
        ])

    def _w(self) -> np.ndarray:
        return np.array([self.weights[k] for k in ["bias"] + FEATURES])

    def score(self, f: dict) -> float:
        z = float(self._vector(f) @ self._w())
        return float(1.0 / (1.0 + np.exp(-z)))

    def fit(self, feature_dicts: list[dict], labels: list[int], l2: float = 1e-2,
            iters: int = 200) -> "ThrowScorer":
        """Refit weights by L2-regularised logistic regression (Newton steps)."""
        X = np.stack([self._vector(f) for f in feature_dicts])
        y = np.asarray(labels, dtype=float)
        w = self._w()
        reg = l2 * np.eye(len(w))
        reg[0, 0] = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(X @ w)))
            grad = X.T @ (p - y) + reg @ w
            hess = (X * (p * (1 - p))[:, None]).T @ X + reg + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.weights = dict(zip(["bias"] + FEATURES, map(float, w)))
        return self


def arm_speed(dense: DenseTrack, arm: str, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Wrist speed relative to the same-side shoulder, in torso lengths per second.

    Returns (speed (T,), relative velocity (T, 2) in px/s)."""
    sho, _, wri = ARMS[arm]
    rel = dense.xy[:, wri] - dense.xy[:, sho]
    vel = central_velocity(rel, fps)
    speed = np.linalg.norm(vel, axis=1) / dense.scale
    return speed, vel


def _local_peaks(s: np.ndarray, min_value: float) -> np.ndarray:
    if len(s) < 3:
        return np.array([], dtype=int)
    x = np.where(np.isnan(s), -np.inf, s)
    mid = x[1:-1]
    is_peak = (mid >= x[:-2]) & (mid > x[2:]) & (mid >= min_value)
    return np.flatnonzero(is_peak) + 1


def _subframe(s: np.ndarray, i: int) -> float:
    """Parabolic interpolation of the peak position around index i."""
    if i <= 0 or i >= len(s) - 1 or np.isnan(s[i - 1 : i + 2]).any():
        return float(i)
    a, b, c = s[i - 1], s[i], s[i + 1]
    denom = a - 2 * b + c
    if denom >= 0:
        return float(i)
    return float(i + np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def _features(dense: DenseTrack, arm: str, speed: np.ndarray, vel: np.ndarray, i: int,
              fps: float, cfg: ThrowDetectorConfig) -> dict:
    T = len(speed)
    sho, elb, wri = ARMS[arm]
    w1, w2 = (int(round(w * fps)) for w in cfg.isolation_window_s)
    lo = np.r_[max(0, i - w2) : max(0, i - w1 + 1)]
    hi = np.r_[min(T, i + w1) : min(T, i + w2 + 1)]
    around = np.concatenate([speed[lo], speed[hi]])
    around = around[~np.isnan(around)]
    background = float(np.max(around)) if len(around) else 0.0
    isolation = speed[i] / max(background, 1.0)

    ow = int(round(cfg.overhead_window_s * fps))
    a, b = max(0, i - ow), min(T, i + ow + 1)
    height = (dense.xy[a:b, sho, 1] - dense.xy[a:b, wri, 1]) / dense.scale[a:b]  # +: wrist above
    overhead = float(np.nanmax(height)) if np.isfinite(height).any() else 0.0

    k = 3
    c = dense.conf[max(0, i - k) : i + k + 1][:, [sho, elb, wri]]
    kp_conf = float(np.mean(c)) if c.size else 0.0

    v = vel[i] if np.isfinite(vel[i]).all() else np.zeros(2)
    direction = v / (np.linalg.norm(v) + 1e-9)

    feet = foot_point(dense.xy[i : i + 1])[0]
    la, ra = dense.xy[i, KP["left_ankle"]], dense.xy[i, KP["right_ankle"]]
    stride = float(np.linalg.norm(la - ra) / dense.scale[i]) if np.isfinite([*la, *ra]).all() else np.nan
    return {
        "peak_speed": float(speed[i]),
        "isolation": float(isolation),
        "overhead": overhead,
        "kp_conf": kp_conf,
        "dir_x": float(direction[0]),
        "dir_y": float(direction[1]),
        "scale_px": float(dense.scale[i]),
        "stride": stride,
        "foot_x": float(feet[0]),
        "foot_y": float(feet[1]),
    }


def detect_throws_in_track(track: PoseTrack, fps: float, cfg: ThrowDetectorConfig | None = None,
                           scorer: ThrowScorer | None = None, keep_all: bool = False,
                           dense: DenseTrack | None = None) -> list[ThrowEvent]:
    """Detect throws for one person. ``keep_all`` returns every candidate (for labeling/fitting)."""
    cfg = cfg or ThrowDetectorConfig()
    scorer = scorer or ThrowScorer()
    if len(track) < 5:
        return []
    if dense is None:
        dense = densify(track, fps, max_gap=cfg.max_gap_frames, min_conf=cfg.min_kp_conf)

    cands = []
    for arm in ARMS:
        speed, vel = arm_speed(dense, arm, fps)
        for i in _local_peaks(speed, cfg.min_speed):
            cands.append((speed[i], arm, i, speed, vel))

    # Non-maximum suppression across both arms.
    cands.sort(key=lambda c: -c[0])
    refractory = cfg.refractory_s * fps
    taken: list[int] = []
    events: list[ThrowEvent] = []
    for _, arm, i, speed, vel in cands:
        if any(abs(i - j) < refractory for j in taken):
            continue
        taken.append(i)
        f = _features(dense, arm, speed, vel, i, fps, cfg)
        frame = dense.frames[0] + _subframe(speed, i)
        events.append(ThrowEvent(track_id=track.track_id, frame=frame, time_s=frame / fps,
                                 arm=arm, confidence=scorer.score(f), features=f))

    # Handedness: the arm with the most confident throws is the throwing arm.
    strong = [e for e in events if e.confidence >= cfg.report_threshold]
    if len(strong) >= 3:
        counts = {a: sum(e.arm == a for e in strong) for a in ARMS}
        dominant = max(counts, key=counts.get)
        for e in events:
            e.features["dominant_arm"] = dominant
            if e.arm != dominant:
                e.confidence *= cfg.off_arm_penalty

    events.sort(key=lambda e: e.frame)
    if keep_all:
        return events
    kept = [e for e in events if e.confidence >= cfg.report_threshold]
    for e in kept:
        e.needs_review = e.confidence < cfg.review_threshold
    return kept


def detect_throws(tracks: list[PoseTrack], fps: float, cfg: ThrowDetectorConfig | None = None,
                  scorer: ThrowScorer | None = None, dense: dict | None = None) -> list[ThrowEvent]:
    out: list[ThrowEvent] = []
    for t in tracks:
        d = dense.get(t.track_id) if dense else None
        out.extend(detect_throws_in_track(t, fps, cfg, scorer, dense=d))
    out.sort(key=lambda e: e.time_s)
    return out
