"""Data types shared by every stage of the pipeline.

Everything downstream of the video layer works on ``PoseTrack`` objects (one
tracked person, per-frame 2D keypoints) and produces ``ThrowEvent`` objects
(one detected throw, filled in stage by stage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# COCO-17 keypoint order, used by RTMPose, YOLO-pose and most 2D pose models.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
N_KP = len(KEYPOINT_NAMES)

# arm -> (shoulder, elbow, wrist) keypoint indices
ARMS = {
    "right": (KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"]),
    "left": (KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"]),
}

METERS_PER_FOOT = 0.3048
MPS_PER_MPH = 0.44704


@dataclass
class PoseTrack:
    """One tracked person.

    frames:    (T,) int frame indices, strictly increasing. Frames where the
               person was not detected are simply absent.
    keypoints: (T, 17, 3) float array of (x, y, confidence) in pixels.
    """

    track_id: int
    frames: np.ndarray
    keypoints: np.ndarray

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=np.int64)
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64)
        if self.keypoints.ndim != 3 or self.keypoints.shape[1:] != (N_KP, 3):
            raise ValueError(f"keypoints must be (T, {N_KP}, 3), got {self.keypoints.shape}")
        if len(self.frames) != len(self.keypoints):
            raise ValueError("frames and keypoints differ in length")
        order = np.argsort(self.frames, kind="stable")
        self.frames = self.frames[order]
        self.keypoints = self.keypoints[order]
        if len(self.frames) > 1 and np.any(np.diff(self.frames) == 0):
            raise ValueError(f"track {self.track_id} has duplicate frame indices")

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class ThrowEvent:
    """One detected throw. Detection fills the first block; later stages fill the rest."""

    track_id: int
    frame: float              # estimated release frame (sub-frame precision)
    time_s: float             # release time from the start of the video
    arm: str                  # "right" | "left"
    confidence: float         # P(this is a real throw), from the throw scorer
    features: dict = field(default_factory=dict)

    # Linking (pairing.py)
    target_type: str = "unknown"          # "partner" | "wall" | "unknown"
    receiver_id: Optional[int] = None
    receiver_confidence: float = 0.0

    # Distance (geometry.py)
    distance_ft: Optional[float] = None
    distance_sd_ft: Optional[float] = None

    # Ball type (ball_type.py)
    ball_type: Optional[str] = None
    ball_type_confidence: float = 0.0

    # Velocity / intensity (velocity.py)
    velo_mph: Optional[float] = None
    velo_lo_mph: Optional[float] = None
    velo_hi_mph: Optional[float] = None
    velo_source: Optional[str] = None     # "flight" | "kinematic"
    intensity_pct: Optional[float] = None

    needs_review: bool = False
