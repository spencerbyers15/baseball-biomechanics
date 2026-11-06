"""Image -> ground-plane mapping, for partner distance.

Two calibration modes, so staff can pick whichever fits the setup:

* ``HomographyGround`` — the accurate one. Four or more ground points with
  known positions (cones, or foul line / bullpen markings measured once per
  camera spot), clicked once. Feet positions map exactly to the ground plane.
* ``PersonHeightGround`` — no setup. Uses each player's known height (roster)
  as a vertical ruler: depth = focal_px * stature / stature_px. Needs only the
  camera's field of view (from the device model) and a roughly level camera.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .schema import METERS_PER_FOOT
from .signals import DenseTrack, foot_point, stature_px

# Ankle keypoint to nose, as a fraction of standing height.
NOSE_ABOVE_ANKLE = 0.87


class GroundModel(Protocol):
    def to_ground(self, dense: DenseTrack, idx: np.ndarray, height_m: float) -> np.ndarray:
        """(len(idx), 2) ground positions in metres (NaN where unknown)."""
        ...

    rel_error: float   # typical relative distance error, for reporting uncertainty


def fit_homography(img_pts, world_pts) -> np.ndarray:
    """Normalised DLT. img_pts, world_pts: (N >= 4, 2). Returns H with world ~ H @ img."""
    src = np.asarray(img_pts, float)
    dst = np.asarray(world_pts, float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need >= 4 matching point pairs")

    def norm(p):
        c = p.mean(axis=0)
        s = math.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Ts, Td = norm(src), norm(dst)
    sh = (Ts @ np.c_[src, np.ones(len(src))].T).T
    dh = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    A = []
    for (x, y, _), (u, v, _) in zip(sh, dh):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


@dataclass
class HomographyGround:
    H: np.ndarray
    rel_error: float = 0.03

    @classmethod
    def from_points(cls, image_points, ground_points_m) -> "HomographyGround":
        return cls(fit_homography(image_points, ground_points_m))

    def to_ground(self, dense: DenseTrack, idx: np.ndarray, height_m: float) -> np.ndarray:
        feet = foot_point(dense.xy[idx])
        out = np.full((len(idx), 2), np.nan)
        ok = np.isfinite(feet).all(axis=1)
        if ok.any():
            out[ok] = apply_homography(self.H, feet[ok])
        return out


@dataclass
class PersonHeightGround:
    image_width: int
    image_height: int
    hfov_deg: float
    rel_error: float = 0.10

    @property
    def fpx(self) -> float:
        return (self.image_width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    def to_ground(self, dense: DenseTrack, idx: np.ndarray, height_m: float) -> np.ndarray:
        xy = dense.xy[idx]
        h_px = stature_px(xy)
        feet = foot_point(xy)
        Z = self.fpx * NOSE_ABOVE_ANKLE * height_m / h_px
        X = (feet[:, 0] - self.image_width / 2) * Z / self.fpx
        out = np.stack([X, Z], axis=1)
        out[~np.isfinite(out).all(axis=1)] = np.nan
        return out


def load_calibration(path: str | Path | dict) -> Optional[GroundModel]:
    """calibration.json:
    {"mode": "homography", "image_points": [[x, y], ...], "ground_points_ft": [[X, Y], ...]}
    {"mode": "person_height", "image_size": [1920, 1080], "hfov_deg": 80}
    """
    cfg = path if isinstance(path, dict) else json.loads(Path(path).read_text())
    mode = cfg.get("mode")
    if mode == "homography":
        g = np.asarray(cfg["ground_points_ft"], float) * METERS_PER_FOOT
        return HomographyGround.from_points(cfg["image_points"], g)
    if mode == "person_height":
        w, h = cfg["image_size"]
        return PersonHeightGround(int(w), int(h), float(cfg["hfov_deg"]))
    raise ValueError(f"unknown calibration mode {mode!r}")


def robust_position(model: GroundModel, dense: DenseTrack, frame: float, fps: float,
                    height_m: float, window_s: float = 0.5) -> np.ndarray:
    """Median ground position over a short window (robust to the stride and to noise)."""
    c = dense.index_of(frame)
    w = int(round(window_s * fps))
    idx = np.arange(max(0, c - w), min(len(dense.frames), c + w + 1))
    pts = model.to_ground(dense, idx, height_m)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return np.array([np.nan, np.nan])
    return np.median(pts, axis=0)
