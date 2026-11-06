"""Training-ball type from color.

Baseline: every pixel near the throwing hand just before release votes for
the nearest palette color in HSV-cone space (hue only matters when the pixel
is saturated, so white/gray balls and saturated plyo balls separate cleanly).
Colors listed under ``ignore`` (skin, glove leather, uniform) absorb their
pixels without voting. Votes are pooled over several frames per throw.

The palette is a staff-editable JSON file, so a new ball set is a config
change, not a retrain. Planned upgrade once labeled clips exist: a small CNN
on the same hand crops (the same recipe as the repo's 98.9%-accurate
EfficientNet player classifier).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Example palette. Edit to match the balls actually in use; mass/diameter feed
# the drag model in velocity.py.
DEFAULT_PALETTE = {
    "balls": [
        {"name": "baseball", "rgb": [235, 232, 222], "mass_g": 145, "diameter_mm": 73},
        {"name": "plyo_100g", "rgb": [200, 40, 45], "mass_g": 100, "diameter_mm": 70},
        {"name": "plyo_150g", "rgb": [235, 200, 40], "mass_g": 150, "diameter_mm": 75},
        {"name": "plyo_450g", "rgb": [40, 90, 200], "mass_g": 450, "diameter_mm": 95},
        {"name": "plyo_1kg", "rgb": [40, 150, 70], "mass_g": 1000, "diameter_mm": 110},
    ],
    "ignore": [
        {"name": "skin_light", "rgb": [225, 180, 150]},
        {"name": "skin_dark", "rgb": [120, 80, 60]},
        {"name": "glove_brown", "rgb": [140, 90, 50]},
        {"name": "uniform_gray", "rgb": [150, 150, 150]},
        {"name": "uniform_navy", "rgb": [30, 40, 70]},
    ],
    "tolerance": 0.2,
    "min_pixels": 6,
}


def rgb_to_cone(rgb: np.ndarray) -> np.ndarray:
    """RGB (0-255) -> HSV cone coordinates (s*cos h, s*sin h, v), each in [-1, 1]."""
    x = np.asarray(rgb, dtype=float).reshape(-1, 3) / 255.0
    mx, mn = x.max(axis=1), x.min(axis=1)
    delta = mx - mn
    s = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0.0)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    h = np.zeros(len(x))
    nz = delta > 1e-9
    rmax = nz & (mx == r)
    gmax = nz & (mx == g) & ~rmax
    bmax = nz & ~rmax & ~gmax
    h[rmax] = ((g - b)[rmax] / delta[rmax]) % 6
    h[gmax] = (b - r)[gmax] / delta[gmax] + 2
    h[bmax] = (r - g)[bmax] / delta[bmax] + 4
    ang = h * (np.pi / 3)
    return np.stack([s * mx * np.cos(ang), s * mx * np.sin(ang), mx], axis=1)


@dataclass
class BallPalette:
    balls: list
    ignore: list = field(default_factory=list)
    tolerance: float = 0.2
    min_pixels: int = 6

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "BallPalette":
        d = DEFAULT_PALETTE if path is None else json.loads(Path(path).read_text())
        return cls(d["balls"], d.get("ignore", []), d.get("tolerance", 0.2), d.get("min_pixels", 6))

    @property
    def masses(self) -> dict:
        return {b["name"]: b.get("mass_g") for b in self.balls}

    @property
    def diameters(self) -> dict:
        return {b["name"]: b.get("diameter_mm") for b in self.balls}

    def classify(self, pixels_rgb: np.ndarray) -> tuple[Optional[str], float, dict]:
        """Returns (ball name or None, confidence in [0, 1], vote counts)."""
        px = np.asarray(pixels_rgb).reshape(-1, 3)
        if len(px) == 0:
            return None, 0.0, {}
        refs = self.balls + self.ignore
        centers = rgb_to_cone(np.array([r["rgb"] for r in refs]))
        d = np.linalg.norm(rgb_to_cone(px)[:, None, :] - centers[None], axis=2)
        nearest = d.argmin(axis=1)
        close = d[np.arange(len(px)), nearest] <= self.tolerance
        votes = {b["name"]: int(np.sum(close & (nearest == i))) for i, b in enumerate(self.balls)}
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        best, n1 = ranked[0]
        n2 = ranked[1][1] if len(ranked) > 1 else 0
        if n1 < self.min_pixels:
            return None, 0.0, votes
        margin = (n1 - n2) / (n1 + n2)
        support = 1.0 - np.exp(-n1 / (3.0 * self.min_pixels))
        return best, float(margin * support), votes


def hand_region(xy_frame: np.ndarray, arm: str, scale_px: float, image_shape) -> Optional[tuple]:
    """Bounding box (x0, y0, x1, y1) around where the ball sits in the throwing hand:
    the wrist pushed a little further along the forearm."""
    from .schema import ARMS

    _, elb, wri = ARMS[arm]
    e, w = xy_frame[elb], xy_frame[wri]
    if not (np.isfinite(e).all() and np.isfinite(w).all()):
        return None
    c = w + 0.3 * (w - e)
    r = 0.45 * scale_px
    h, wid = image_shape[:2]
    x0, y0 = int(max(0, c[0] - r)), int(max(0, c[1] - r))
    x1, y1 = int(min(wid, c[0] + r)), int(min(h, c[1] + r))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1
