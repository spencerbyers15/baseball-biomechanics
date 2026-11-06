"""Session analysis: pose tracks in, fully annotated throws out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .ball_type import BallPalette
from .geometry import GroundModel, robust_position
from .pairing import PairingConfig, link_throws
from .schema import METERS_PER_FOOT, PoseTrack, ThrowEvent
from .signals import densify
from .throws import ThrowDetectorConfig, ThrowScorer, detect_throws
from .velocity import KinematicVelocityModel, assign_intensity, assign_velocity

DEFAULT_HEIGHT_M = 1.88   # typical MLB pitcher (6'2") when the roster has no height


@dataclass
class SessionConfig:
    fps: float
    throw: ThrowDetectorConfig = field(default_factory=ThrowDetectorConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    scorer: Optional[ThrowScorer] = None
    ground: Optional[GroundModel] = None
    palette: Optional[BallPalette] = None
    velocity_model: Optional[KinematicVelocityModel] = None
    roster: dict = field(default_factory=dict)        # track_id -> {"name": str, "height_m": float}
    receiver_review_below: float = 0.5
    ball_type_review_below: float = 0.3


# (throw) -> RGB pixels near the throwing hand just before release, or None
BallPixelsFn = Callable[[ThrowEvent], Optional[np.ndarray]]


def analyze_session(tracks: list[PoseTrack], cfg: SessionConfig,
                    ball_pixels_fn: Optional[BallPixelsFn] = None) -> list[ThrowEvent]:
    dense = {t.track_id: densify(t, cfg.fps, cfg.throw.max_gap_frames, cfg.throw.min_kp_conf)
             for t in tracks if len(t) >= 5}
    throws = detect_throws(tracks, cfg.fps, cfg.throw, cfg.scorer, dense=dense)
    link_throws(throws, dense, cfg.pairing)
    heights = {tid: float(info.get("height_m", DEFAULT_HEIGHT_M)) for tid, info in cfg.roster.items()}

    if cfg.ground is not None:
        for t in throws:
            if t.target_type != "partner":
                continue
            a = robust_position(cfg.ground, dense[t.track_id], t.frame, cfg.fps,
                                heights.get(t.track_id, DEFAULT_HEIGHT_M))
            b = robust_position(cfg.ground, dense[t.receiver_id], t.frame, cfg.fps,
                                heights.get(t.receiver_id, DEFAULT_HEIGHT_M))
            d = float(np.linalg.norm(a - b))
            if np.isfinite(d):
                t.distance_ft = d / METERS_PER_FOOT
                t.distance_sd_ft = cfg.ground.rel_error * t.distance_ft

    if cfg.palette is not None and ball_pixels_fn is not None:
        for t in throws:
            px = ball_pixels_fn(t)
            if px is not None and len(px):
                t.ball_type, t.ball_type_confidence, _ = cfg.palette.classify(px)

    masses = cfg.palette.masses if cfg.palette else {}
    assign_velocity(throws, cfg.velocity_model, heights, masses)
    assign_intensity(throws)

    for t in throws:
        if t.target_type == "partner" and t.receiver_confidence < cfg.receiver_review_below:
            t.needs_review = True
        if t.target_type == "unknown":
            t.needs_review = True
        if cfg.palette is not None and ball_pixels_fn is not None and \
                t.ball_type_confidence < cfg.ball_type_review_below:
            t.needs_review = True
    return throws
