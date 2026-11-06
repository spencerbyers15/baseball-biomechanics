"""Throw velocity and intensity.

Two independent estimates, used in this order of preference:

1. **Flight physics** — when the ball is tracked from release to catch, the
   partner distance and the flight time give release speed directly, with a
   quadratic-drag correction that depends on the ball's mass and size (so a
   2 kg plyo ball and a baseball are handled by the same formula).
2. **Kinematic regression** — throwing-hand speed (from pose, converted to m/s
   with the player's height), stride, distance and ball mass, mapped to radar
   velocity by a ridge regression fit on the throws that have a radar reading.
   Reports a prediction interval, not just a point estimate.

Intensity is velocity as a percentage of the player's own max for that ball
type, which is what a workload model needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schema import MPS_PER_MPH, ThrowEvent

G = 9.81          # m/s^2
RHO_AIR = 1.2     # kg/m^3
TORSO_FRACTION = 0.27   # shoulder-midpoint to hip-midpoint as a fraction of stature


def drag_k(mass_g: float = 145.0, diameter_mm: float = 73.0, cd: float = 0.35) -> float:
    """Quadratic drag constant k (1/m) in dv/dt = -k |v| v."""
    area = math.pi * (diameter_mm / 2000.0) ** 2
    return RHO_AIR * cd * area / (2.0 * mass_g / 1000.0)


def simulate_flight(speed_mps: float, launch_deg: float, distance_m: float, k: float,
                    release_h: float = 1.9, dt: float = 1 / 2000) -> tuple[float, float]:
    """Integrate a 2D flight with quadratic drag until the ball covers ``distance_m``.

    Returns (flight_time_s, height_at_distance_m); height is -inf if it lands short."""
    a = math.radians(launch_deg)
    x, y = 0.0, release_h
    vx, vy = speed_mps * math.cos(a), speed_mps * math.sin(a)
    t = 0.0
    while x < distance_m:
        v = math.hypot(vx, vy)
        ax, ay = -k * v * vx, -k * v * vy - G
        # semi-implicit Euler is plenty at dt = 0.5 ms
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        if y < 0 or vx <= 0:
            return t, -math.inf
    return t, y


def launch_angle_for(distance_m: float, speed_mps: float, k: float, release_h: float = 1.9,
                     catch_h: float = 1.4, dt: float = 1 / 500) -> float:
    """Low-trajectory launch angle that arrives at ``catch_h`` (bisection)."""
    lo, hi = -15.0, 40.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        _, h = simulate_flight(speed_mps, mid, distance_m, k, release_h, dt=dt)
        if h < catch_h:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def release_speed_from_flight(distance_m: float, flight_time_s: float, k: float,
                              dh_m: float = -0.5) -> float:
    """Release speed (m/s) from horizontal distance and flight time.

    Horizontal: with drag dv/dx = -k v, the time to cover D is (e^{kD} - 1) / (k v0x).
    Vertical: no-drag ballistic, dh_m = catch height - release height.
    """
    D, T = distance_m, flight_time_s
    vx = D / T if k <= 0 else (math.exp(k * D) - 1.0) / (k * T)
    vy = (dh_m + 0.5 * G * T * T) / T
    return math.hypot(vx, vy)


# ---------------------------------------------------------------------------
# Kinematic regression


FEATURES = ["hand_speed_mps", "stride", "distance_ft", "ball_mass_g"]


def kinematic_features(throw: ThrowEvent, player_height_m: float = 1.88,
                       ball_mass_g: Optional[float] = None) -> dict:
    """Features for the regression. Hand speed is converted from torso-lengths/s to m/s."""
    f = throw.features
    torso_m = TORSO_FRACTION * player_height_m
    return {
        "hand_speed_mps": f.get("peak_speed", np.nan) * torso_m,
        "stride": f.get("stride", np.nan),
        "distance_ft": throw.distance_ft if throw.distance_ft is not None else np.nan,
        "ball_mass_g": ball_mass_g if ball_mass_g is not None else np.nan,
    }


@dataclass
class KinematicVelocityModel:
    """Ridge regression from kinematic features to radar velocity (mph)."""

    l2: float = 1.0
    z_clip: float = 3.0
    feature_names: list = field(default_factory=lambda: list(FEATURES))
    mean: Optional[list] = None
    std: Optional[list] = None
    coef: Optional[list] = None
    intercept: float = 0.0
    resid_sd: float = 0.0
    cov: Optional[list] = None   # (Z^T Z + l2 I)^-1, for prediction intervals
    n: int = 0
    offsets: dict = field(default_factory=dict)   # per-player residual mean, shrunk toward 0
    shrink_n: float = 5.0                         # throws needed for half weight on a player's offset

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[r.get(k, np.nan) for k in self.feature_names] for r in rows], dtype=float)
        return X

    def fit(self, rows: list[dict], mph: list[float], groups: Optional[list] = None) -> "KinematicVelocityModel":
        """Ridge fit. ``groups`` (e.g. player ids) adds a shrunken per-player offset,
        so a player who throws harder than his hand speed suggests is calibrated."""
        X = self._matrix(rows)
        y = np.asarray(mph, dtype=float)
        seen = np.isfinite(X).any(axis=0)
        mean = np.zeros(X.shape[1])
        mean[seen] = np.nanmean(X[:, seen], axis=0)   # all-missing column -> constant 0
        X = np.where(np.isnan(X), mean, X)          # impute missing with the mean
        std = X.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)        # constant column -> no-op
        Z = (X - mean) / std
        A = Z.T @ Z + self.l2 * np.eye(Z.shape[1])
        cov = np.linalg.inv(A)
        coef = cov @ Z.T @ (y - y.mean())
        resid = y - (y.mean() + Z @ coef)
        dof = max(1, len(y) - Z.shape[1] - 1)
        self.mean, self.std, self.coef = mean.tolist(), std.tolist(), coef.tolist()
        self.intercept = float(y.mean())
        self.resid_sd = float(np.sqrt(np.sum(resid ** 2) / dof))
        self.cov = cov.tolist()
        self.n = len(y)
        self.offsets = {}
        if groups is not None:
            g = np.asarray([str(x) for x in groups])
            for key in np.unique(g):
                r = resid[g == key]
                self.offsets[key] = float(r.sum() / (len(r) + self.shrink_n))
            adj = resid - np.array([self.offsets[k] for k in g])
            self.resid_sd = float(np.sqrt(np.sum(adj ** 2) / max(1, dof - len(self.offsets))))
        return self

    def predict(self, row: dict, group=None) -> tuple[float, float]:
        """Returns (mph, standard deviation of the prediction)."""
        if self.coef is None:
            raise RuntimeError("model is not fitted")
        x = self._matrix([row])[0]
        mean = np.asarray(self.mean)
        x = np.where(np.isnan(x), mean, x)
        # Clip so a feature that barely varied in training (e.g. every radar throw
        # at 60 ft) cannot extrapolate wildly on a 120 ft throw.
        z = np.clip((x - mean) / np.asarray(self.std), -self.z_clip, self.z_clip)
        mph = self.intercept + float(z @ np.asarray(self.coef)) + self.offsets.get(str(group), 0.0)
        leverage = float(z @ np.asarray(self.cov) @ z)
        sd = self.resid_sd * math.sqrt(1.0 + 1.0 / max(self.n, 1) + leverage)
        return mph, sd

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "KinematicVelocityModel":
        m = cls()
        m.__dict__.update(d)
        return m


def match_radar(throws: list[ThrowEvent], radar: list[dict], tol_s: float = 1.0) -> list[tuple[int, float]]:
    """One-to-one match of radar readings to detected throws by time.

    ``radar`` rows: {"time_s": float, "mph": float, optional "track_id": int}.
    A reading with a track_id only matches that player's throws.
    Returns [(throw_index, mph)]."""
    pairs = []
    for ri, r in enumerate(radar):
        for ti, t in enumerate(throws):
            if r.get("track_id") is not None and int(r["track_id"]) != t.track_id:
                continue
            dt = abs(t.time_s - float(r["time_s"]))
            if dt <= tol_s:
                pairs.append((dt, ti, ri))
    pairs.sort()
    used_t, used_r, out = set(), set(), []
    for _, ti, ri in pairs:
        if ti in used_t or ri in used_r:
            continue
        used_t.add(ti)
        used_r.add(ri)
        out.append((ti, float(radar[ri]["mph"])))
    return sorted(out)


def assign_velocity(throws: list[ThrowEvent], model: Optional[KinematicVelocityModel],
                    heights_m: dict, ball_masses_g: dict, z: float = 1.645) -> None:
    """Fill velo_mph (+ 90% interval by default) from the kinematic model."""
    if model is None or model.coef is None:
        return
    for t in throws:
        if t.velo_source == "flight":
            continue
        mass = ball_masses_g.get(t.ball_type) if t.ball_type else None
        row = kinematic_features(t, heights_m.get(t.track_id, 1.88), mass)
        mph, sd = model.predict(row, group=t.track_id)
        t.velo_mph, t.velo_lo_mph, t.velo_hi_mph = mph, mph - z * sd, mph + z * sd
        t.velo_source = "kinematic"


def assign_intensity(throws: list[ThrowEvent], player_max_mph: Optional[dict] = None) -> None:
    """Intensity = % of the player's max for that ball type.

    Uses velocity when available, otherwise peak hand speed as a relative proxy.
    ``player_max_mph`` {(track_id, ball_type): mph} supplies maxima from past sessions."""
    groups: dict = {}
    for t in throws:
        groups.setdefault((t.track_id, t.ball_type), []).append(t)
    for key, ts in groups.items():
        velos = [t.velo_mph for t in ts if t.velo_mph is not None]
        if velos:
            ref = max(velos + ([player_max_mph[key]] if player_max_mph and key in player_max_mph else []))
            for t in ts:
                if t.velo_mph is not None:
                    t.intensity_pct = 100.0 * t.velo_mph / ref
        else:
            speeds = [t.features.get("peak_speed", 0.0) for t in ts]
            ref = max(speeds) if speeds else 0.0
            if ref > 0:
                for t, s in zip(ts, speeds):
                    t.intensity_pct = 100.0 * s / ref


def mph(mps: float) -> float:
    return mps / MPS_PER_MPH
