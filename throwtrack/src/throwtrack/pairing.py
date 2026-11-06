"""Linking each throw to its receiver.

Partners in catch play throw in strict alternation (A B A B ...), while two
players from different pairs interleave at random (alternation ~0.5). Over a
sliding window we score every pair of players on:

* alternation rate of their merged throw sequence,
* whether each switch happens within a plausible catch-and-return delay,
* whether each thrower's release direction (image x) points at the other,
* how balanced their throw counts are,

then take a greedy maximum matching. Windows make this robust to players
joining, leaving and switching partners. Throwers with no partner and a fast,
regular rhythm are labelled wall drills (or by a staff-drawn wall zone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schema import ThrowEvent
from .signals import DenseTrack, foot_point


@dataclass
class PairingConfig:
    window_s: float = 90.0              # half-width of the partner-inference window
    step_s: float = 30.0
    min_throws: int = 4                 # combined throws in a window to score a pair
    min_pair_score: float = 0.3
    return_window_s: tuple = (0.8, 15.0)
    wall_max_interval_s: float = 7.0
    wall_zones: list = field(default_factory=list)   # image polygons [[x, y], ...]


def _foot_at(dense: Optional[DenseTrack], frame: float) -> Optional[np.ndarray]:
    if dense is None or frame < dense.frames[0] or frame > dense.frames[-1]:
        return None
    p = foot_point(dense.xy[[dense.index_of(frame)]])[0]
    return p if np.isfinite(p).all() else None


def direction_agreement(throw: ThrowEvent, target: Optional[DenseTrack]) -> float:
    """1 if the release direction points at the target in image x, 0 if away,
    0.5 if unknown or mostly vertical (target roughly in line with the camera)."""
    foot = _foot_at(target, throw.frame)
    dx = throw.features.get("dir_x")
    if foot is None or dx is None or abs(dx) < 0.3:
        return 0.5
    sep = foot[0] - throw.features["foot_x"]
    scale = throw.features.get("scale_px", 1.0)
    if abs(sep) < scale:          # target almost straight behind/in front: uninformative
        return 0.5
    return 1.0 if np.sign(sep) == np.sign(dx) else 0.0


def pair_score(ta: list[ThrowEvent], tb: list[ThrowEvent], da: Optional[DenseTrack],
               db: Optional[DenseTrack], cfg: PairingConfig) -> float:
    n = len(ta) + len(tb)
    if n < cfg.min_throws or not ta or not tb:
        return 0.0
    merged = sorted([(t.time_s, 0) for t in ta] + [(t.time_s, 1) for t in tb])
    labels = np.array([m[1] for m in merged])
    times = np.array([m[0] for m in merged])
    switch = labels[1:] != labels[:-1]
    alt = switch.mean()
    gaps = np.diff(times)[switch]
    lo, hi = cfg.return_window_s
    timing = float(np.mean((gaps >= lo) & (gaps <= hi))) if len(gaps) else 0.0
    agree = np.mean([direction_agreement(t, db) for t in ta] + [direction_agreement(t, da) for t in tb])
    balance = min(len(ta), len(tb)) / max(len(ta), len(tb))
    alt_score = float(np.clip((alt - 0.5) / 0.45, 0.0, 1.0))
    support = 1.0 - np.exp(-n / 6.0)
    return alt_score * timing * (0.4 + 0.6 * agree) * np.sqrt(balance) * support


def _point_in_poly(pt, poly) -> bool:
    x, y = pt
    inside = False
    poly = np.asarray(poly, float)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def match_partners(throws: list[ThrowEvent], dense: dict, cfg: PairingConfig) -> list[tuple[float, dict]]:
    """[(window_center_s, {track_id: (partner_id, score)})] over the session."""
    if not throws:
        return []
    by_track: dict = {}
    for t in throws:
        by_track.setdefault(t.track_id, []).append(t)
    t0, t1 = throws[0].time_s, throws[-1].time_s
    centers = np.arange(t0, t1 + cfg.step_s, cfg.step_s)
    out = []
    ids = sorted(by_track)
    for c in centers:
        win = {i: [t for t in by_track[i] if abs(t.time_s - c) <= cfg.window_s] for i in ids}
        scores = []
        for ai, a in enumerate(ids):
            for b in ids[ai + 1 :]:
                s = pair_score(win[a], win[b], dense.get(a), dense.get(b), cfg)
                if s >= cfg.min_pair_score:
                    scores.append((s, a, b))
        scores.sort(reverse=True)
        taken, m = set(), {}
        for s, a, b in scores:
            if a in taken or b in taken:
                continue
            taken |= {a, b}
            m[a] = (b, s)
            m[b] = (a, s)
        out.append((float(c), m))
    return out


def link_throws(throws: list[ThrowEvent], dense: dict, cfg: Optional[PairingConfig] = None) -> None:
    """Fill target_type / receiver_id / receiver_confidence on each throw in place."""
    cfg = cfg or PairingConfig()
    throws.sort(key=lambda t: t.time_s)
    windows = match_partners(throws, dense, cfg)
    if not windows:
        return
    centers = np.array([w[0] for w in windows])
    by_track: dict = {}
    for t in throws:
        by_track.setdefault(t.track_id, []).append(t.time_s)
    lo, hi = cfg.return_window_s

    for t in throws:
        m = windows[int(np.argmin(np.abs(centers - t.time_s)))][1]
        partner = m.get(t.track_id)
        if partner is not None and _foot_at(dense.get(partner[0]), t.frame) is not None:
            pid, score = partner
            p_times = np.array(by_track.get(pid, []))
            dts = p_times - t.time_s
            local = 1.0 if np.any(((dts >= lo) & (dts <= hi)) | ((-dts >= lo) & (-dts <= hi))) else 0.7
            agree = direction_agreement(t, dense.get(pid))
            dir_factor = {1.0: 1.0, 0.5: 0.85, 0.0: 0.5}[agree]
            t.target_type, t.receiver_id = "partner", pid
            t.receiver_confidence = float(np.sqrt(score) * local * dir_factor)
            continue
        # No partner: wall drill?
        foot = (t.features.get("foot_x"), t.features.get("foot_y"))
        if cfg.wall_zones and None not in foot and any(_point_in_poly(foot, z) for z in cfg.wall_zones):
            t.target_type, t.receiver_confidence = "wall", 0.9
            continue
        own = np.array(by_track[t.track_id])
        near = own[np.abs(own - t.time_s) <= 60.0]
        if len(near) >= 3 and np.median(np.diff(near)) <= cfg.wall_max_interval_s:
            t.target_type, t.receiver_confidence = "wall", 0.6
        else:
            t.target_type, t.receiver_confidence = "unknown", 0.0
