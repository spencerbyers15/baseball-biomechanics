"""Synthetic throwing sessions with ground truth, for tests and the no-GPU demo.

Simulates what the pose layer would output for a fixed camera watching
several throwing pairs and a wall-drill player: 3D skeletons with throwing,
catching, walking and arm-circle motions, projected through a pinhole camera,
then corrupted with keypoint noise, dropouts, motion blur at release and short
detection gaps. Players join and leave mid-session and one pair backs up
(long toss), so linking and distance logic get exercised.

This is a test harness, not a claim about real-world accuracy: it checks the
logic end to end. Real accuracy comes from the labeled sessions the club provides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .schema import KP, N_KP, PoseTrack
from .velocity import drag_k, launch_angle_for, simulate_flight

# Base pose for a 1.85 m player in local (right, up, forward) metres.
_BASE = np.zeros((N_KP, 3))
for name, (r, u, f) in {
    "nose": (0.0, 1.68, 0.09), "left_eye": (-0.035, 1.71, 0.08), "right_eye": (0.035, 1.71, 0.08),
    "left_ear": (-0.08, 1.69, 0.0), "right_ear": (0.08, 1.69, 0.0),
    "left_shoulder": (-0.20, 1.48, 0.0), "right_shoulder": (0.20, 1.48, 0.0),
    "left_elbow": (-0.24, 1.18, 0.0), "right_elbow": (0.24, 1.18, 0.0),
    "left_wrist": (-0.26, 0.92, 0.05), "right_wrist": (0.26, 0.92, 0.05),
    "left_hip": (-0.11, 0.98, 0.0), "right_hip": (0.11, 0.98, 0.0),
    "left_knee": (-0.12, 0.52, 0.02), "right_knee": (0.12, 0.52, 0.02),
    "left_ankle": (-0.12, 0.08, 0.0), "right_ankle": (0.12, 0.08, 0.0),
}.items():
    _BASE[KP[name]] = (r, u, f)

BALLS = {
    "baseball": {"mass_g": 145.0, "diameter_mm": 73.0},
    "plyo_450g": {"mass_g": 450.0, "diameter_mm": 95.0},
    "plyo_1kg": {"mass_g": 1000.0, "diameter_mm": 110.0},
}


@dataclass
class Camera:
    width: int = 1920
    height: int = 1080
    hfov_deg: float = 80.0
    height_m: float = 1.6
    pitch_deg: float = 6.0      # tilted down

    @property
    def fpx(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World (X right, Y up, Z forward) -> pixels. Returns (uv, depth)."""
        th = math.radians(self.pitch_deg)
        d = P - np.array([0.0, self.height_m, 0.0])
        fwd = np.array([0.0, -math.sin(th), math.cos(th)])
        up = np.array([0.0, math.cos(th), math.sin(th)])
        xc, yc, zc = d[..., 0], -(d @ up), d @ fwd
        u = self.width / 2 + self.fpx * xc / zc
        v = self.height / 2 + self.fpx * yc / zc
        return np.stack([u, v], axis=-1), zc

    def ground_to_image(self, XZ: np.ndarray) -> np.ndarray:
        XZ = np.atleast_2d(XZ)
        P = np.stack([XZ[:, 0], np.zeros(len(XZ)), XZ[:, 1]], axis=1)
        return self.project(P)[0]


@dataclass
class SimPlayer:
    pid: int
    height_m: float
    arm: str                                      # throwing arm
    t_enter: float
    t_exit: float
    stations: list = field(default_factory=list)  # [(t_from, X, Z)]
    target: object = None                         # partner pid, or ("wall", (X, Z))
    entry_offset: tuple = (-6.0, 0.0)


@dataclass
class SimSession:
    fps: float
    duration_s: float
    camera: Camera
    players: dict
    tracks: list
    throws: list          # ground truth dicts
    calibration: dict     # cones: image points + ground points (ft)


def _ease(p):
    p = np.clip(p, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * p)


def _lerp(a, b, w):
    return a + (b - a) * np.asarray(w)[..., None]


class _Motion:
    """Accumulates local joint offsets for one player over a time grid."""

    def __init__(self, t: np.ndarray, scale: float, arm: str):
        self.t = t
        self.s = scale
        self.sg = 1.0 if arm == "right" else -1.0
        self.throw_wri = KP[f"{arm}_wrist"]
        glove = "left" if arm == "right" else "right"
        self.glove_wri = KP[f"{glove}_wrist"]
        self.lead_ankle, self.lead_knee = KP[f"{glove}_ankle"], KP[f"{glove}_knee"]
        self.L = np.repeat(_BASE[None] * scale, len(t), axis=0)
        self.body_shift = np.zeros(len(t))

    def _w(self, r, u, f):
        return np.array([r, u, f]) * self.s

    def _path(self, idx, keypoints, times, mode):
        """Piecewise path through keypoints at the given times (relative to t)."""
        for (a, b), (ta, tb), m in zip(zip(keypoints[:-1], keypoints[1:]), zip(times[:-1], times[1:]), mode):
            sel = (self.t >= ta) & (self.t < tb)
            if not sel.any():
                continue
            p = (self.t[sel] - ta) / max(tb - ta, 1e-9)
            w = {"ease": _ease(p), "accel": p ** 2, "decel": 1 - (1 - p) ** 2}[m]
            self.L[sel, idx] = _lerp(a, b, w)

    def throw(self, tr: float, hand_speed: float):
        sg = self.sg
        W0, Wc = self._w(sg * 0.26, 0.92, 0.05), self._w(sg * 0.32, 1.72, -0.50)
        Wr, Wf = self._w(sg * 0.18, 1.72, 0.50), self._w(-sg * 0.30, 0.85, 0.45)
        Ta = 2 * np.linalg.norm(Wr - Wc) / hand_speed
        Tf = 2 * np.linalg.norm(Wf - Wr) / hand_speed
        self._path(self.throw_wri, [W0, Wc, Wr, Wf, W0],
                   [tr - 0.9, tr - Ta, tr, tr + Tf, tr + Tf + 0.8], ["ease", "accel", "decel", "ease"])
        G0, Gup, Gt = self._w(-sg * 0.26, 0.92, 0.05), self._w(-sg * 0.12, 1.45, 0.45), self._w(-sg * 0.22, 1.15, 0.12)
        self._path(self.glove_wri, [G0, Gup, Gt, G0], [tr - 0.6, tr - 0.1, tr + 0.25, tr + 0.9], ["ease"] * 3)
        # stride + weight shift
        sel = (self.t >= tr - 0.45) & (self.t < tr + 1.1)
        p_in = _ease((self.t[sel] - (tr - 0.45)) / 0.35)
        p_out = _ease((self.t[sel] - (tr + 0.5)) / 0.6)
        shift = (p_in - p_out) * self.s
        self.L[sel, self.lead_ankle, 2] += 0.7 * shift
        self.L[sel, self.lead_knee, 2] += 0.5 * shift
        self.body_shift[sel] += 0.35 * shift

    def catch(self, tc: float):
        sg = self.sg
        G0, Gc, Gch = self._w(-sg * 0.26, 0.92, 0.05), self._w(-sg * 0.10, 1.50, 0.40), self._w(-sg * 0.05, 1.25, 0.20)
        self._path(self.glove_wri, [G0, Gc, Gch, G0], [tc - 0.45, tc, tc + 0.35, tc + 0.9], ["ease"] * 3)
        W0, Wch = self._w(sg * 0.26, 0.92, 0.05), self._w(sg * 0.05, 1.25, 0.20)
        self._path(self.throw_wri, [W0, Wch, W0], [tc + 0.1, tc + 0.45, tc + 1.0], ["ease"] * 2)

    def arm_circles(self, t0: float, dur: float, hz: float):
        sel = (self.t >= t0) & (self.t < t0 + dur)
        tt = self.t[sel] - t0
        center = self._w(self.sg * 0.22, 1.48, 0.0)
        ph = 2 * np.pi * hz * tt
        circle = center + self.s * 0.62 * np.stack([np.zeros_like(ph), -np.cos(ph), np.sin(ph)], axis=1)
        w = np.minimum(_ease(tt / 0.3), _ease((dur - tt) / 0.3))
        self.L[sel, self.throw_wri] = _lerp(self.L[sel, self.throw_wri], circle, w)

    def walk(self, moving: np.ndarray):
        ph = 2 * np.pi * 0.9 * self.t
        swing = 0.22 * self.s * np.sin(ph) * moving
        self.L[:, self.throw_wri, 2] += swing
        self.L[:, self.glove_wri, 2] -= swing
        self.L[:, KP["left_ankle"], 2] += 0.3 * self.s * np.sin(ph) * moving
        self.L[:, KP["right_ankle"], 2] -= 0.3 * self.s * np.sin(ph) * moving

    def finish(self) -> np.ndarray:
        L = self.L.copy()
        upper = [i for i in range(N_KP) if i not in (KP["left_ankle"], KP["right_ankle"],
                                                    KP["left_knee"], KP["right_knee"])]
        L[:, upper, 2] += self.body_shift[:, None]
        # elbows follow shoulder and wrist
        for side in ("left", "right"):
            sho, elb, wri = KP[f"{side}_shoulder"], KP[f"{side}_elbow"], KP[f"{side}_wrist"]
            L[:, elb] = 0.5 * (L[:, sho] + L[:, wri]) + np.array([0.0, -0.03, -0.06]) * self.s
        return L


def _root_path(p: SimPlayer, t: np.ndarray, walk_speed: float = 1.3):
    """Root (X, Z) per frame, and a 0/1 'moving' flag. Walks between stations."""
    entry = np.array(p.stations[0][1:]) + np.array(p.entry_offset)
    waypoints = [(p.t_enter, entry)] + [(tf, np.array([x, z])) for tf, x, z in p.stations]
    XZ = np.zeros((len(t), 2))
    moving = np.zeros(len(t))
    XZ[:] = entry
    cur, cur_t = entry, p.t_enter
    for tf, dest in waypoints[1:]:
        start = max(tf, cur_t)
        dur = np.linalg.norm(dest - cur) / walk_speed
        sel = t >= start
        w = np.clip((t[sel] - start) / max(dur, 1e-9), 0, 1)
        XZ[sel] = cur + (dest - cur) * w[:, None]
        moving[sel] = ((w > 0) & (w < 1)).astype(float)
        cur, cur_t = dest, start + dur
    return XZ, moving


def _noisy_track(pid, frames, uv, depth, scale_px, rng, cam) -> PoseTrack | None:
    T = len(frames)
    kp = np.zeros((T, N_KP, 3))
    sigma = 1.0 + 0.02 * scale_px[:, None]
    kp[:, :, :2] = uv + rng.normal(size=uv.shape) * sigma[..., None]
    kp[:, :, 2] = rng.uniform(0.6, 0.95, size=(T, N_KP))
    # motion blur on fast wrists/elbows: lower confidence, extra error
    for wri, elb in ((KP["left_wrist"], KP["left_elbow"]), (KP["right_wrist"], KP["right_elbow"])):
        speed = np.zeros(T)
        speed[1:] = np.linalg.norm(np.diff(uv[:, wri], axis=0), axis=1) / scale_px[1:]
        blur = speed > 0.5
        for j in (wri, elb):
            kp[blur, j, 2] = rng.uniform(0.3, 0.5, size=blur.sum())
            kp[blur, j, :2] += rng.normal(size=(blur.sum(), 2)) * (0.15 * scale_px[blur])[:, None]
    # random dropouts: low confidence and a wild position
    drop = rng.random((T, N_KP)) < 0.03
    kp[drop, 2] = rng.uniform(0.0, 0.15, size=drop.sum())
    kp[drop, :2] += rng.normal(size=(drop.sum(), 2)) * 40
    # visible only when in front of the camera and inside the frame
    feet = 0.5 * (uv[:, KP["left_ankle"]] + uv[:, KP["right_ankle"]])
    inside = (depth[:, 0] > 1) & (feet[:, 0] > 0) & (feet[:, 0] < cam.width) & (uv[:, KP["nose"], 1] > 0) \
        & (feet[:, 1] < cam.height)
    # short detection gaps
    gap = np.zeros(T, bool)
    starts = np.flatnonzero(rng.random(T) < 0.004)
    for s in starts:
        gap[s : s + rng.integers(1, 6)] = True
    keep = inside & ~gap
    if keep.sum() < 10:
        return None
    return PoseTrack(pid, frames[keep], kp[keep])


def build_session(duration_s: float = 360.0, fps: float = 30.0, seed: int = 0,
                  camera: Camera | None = None) -> SimSession:
    """Three throwing pairs (one oblique, one that backs up for long toss, one that
    joins late) plus a wall-drill player who uses plyo balls."""
    rng = np.random.default_rng(seed)
    cam = camera or Camera()
    D = duration_s
    players = {
        # pair A: lateral at 20 m depth, leaves 60 s before the end
        1: SimPlayer(1, 1.90, "right", 0.0, D - 60, [(0, -9.0, 20.0)], 2),
        2: SimPlayer(2, 1.85, "right", 0.0, D - 60, [(0, 9.0, 20.0)], 1, (6.0, 0.0)),
        # pair B: lateral at 30 m depth, backs up from 60 ft to ~100 ft halfway through
        3: SimPlayer(3, 1.93, "left", 5.0, D, [(0, -9.0, 30.0), (D / 2, -15.0, 30.0)], 4),
        4: SimPlayer(4, 1.88, "right", 5.0, D, [(0, 9.0, 30.0), (D / 2, 15.0, 30.0)], 3, (6.0, 0.0)),
        # pair C: oblique (partner further away in depth), joins at 60 s
        5: SimPlayer(5, 1.83, "right", 60.0, D, [(0, -2.0, 38.0)], 6),
        6: SimPlayer(6, 1.96, "right", 60.0, D, [(0, 12.0, 50.0)], 5, (6.0, 0.0)),
        # wall drills, facing a wall 4 m to the left
        7: SimPlayer(7, 1.88, "right", 0.0, D, [(0, -16.0, 24.0)], ("wall", (-20.0, 24.0))),
    }

    t = np.arange(0, D, 1 / fps)
    motions = {pid: _Motion(t, p.height_m / 1.85, p.arm) for pid, p in players.items()}
    roots = {pid: _root_path(p, t) for pid, p in players.items()}

    def pos(pid, time):
        i = min(int(time * fps), len(t) - 1)
        return roots[pid][0][i]

    def settled(pid, time):
        i = min(int(time * fps), len(t) - 1)
        p = players[pid]
        return p.t_enter + 8 <= time <= p.t_exit - 3 and roots[pid][1][max(0, i - 30) : i + 30].max() == 0

    gt = []
    # warm-up arm circles right after arriving
    for pid, p in players.items():
        motions[pid].arm_circles(p.t_enter + 6.5, 5.0, hz=rng.uniform(1.0, 1.4))

    for a, b in ((1, 2), (3, 4), (5, 6)):
        base = rng.uniform(22, 31)            # ball speed m/s for this pair
        tnow = max(players[a].t_enter, players[b].t_enter) + 14 + rng.uniform(0, 3)
        thrower, catcher = a, b
        while tnow < min(players[a].t_exit, players[b].t_exit) - 5:
            if not (settled(thrower, tnow) and settled(catcher, tnow)):
                tnow += 1.0
                continue
            dist = float(np.linalg.norm(pos(thrower, tnow) - pos(catcher, tnow)))
            v = base * rng.uniform(0.92, 1.08) * (1 + 0.15 * (dist > 25))
            k = drag_k(**BALLS["baseball"])
            ang = launch_angle_for(dist, v, k)
            flight, _ = simulate_flight(v, ang, dist, k, dt=1 / 500)
            motions[thrower].throw(tnow, 0.92 * v)
            motions[catcher].catch(tnow + flight)
            gt.append({"thrower": thrower, "time_s": tnow, "target": catcher, "distance_m": dist,
                       "ball_speed_mps": v, "flight_s": flight, "ball_type": "baseball"})
            tnow = tnow + flight + rng.uniform(2.5, 6.0)
            thrower, catcher = catcher, thrower

    wall = players[7]
    tnow = wall.t_enter + 14
    while tnow < wall.t_exit - 5:
        ball = "plyo_450g" if (tnow // 90) % 2 == 0 else "plyo_1kg"
        v = rng.uniform(14, 18) if ball == "plyo_450g" else rng.uniform(10, 13)
        motions[7].throw(tnow, v)
        motions[7].catch(tnow + 0.9)      # rebound
        gt.append({"thrower": 7, "time_s": tnow, "target": "wall", "distance_m": 4.0,
                   "ball_speed_mps": v, "flight_s": None, "ball_type": ball})
        tnow += rng.uniform(3.5, 6.0)

    tracks = []
    for pid, p in players.items():
        m = motions[pid]
        XZ, moving = roots[pid]
        m.walk(moving)
        L = m.finish()
        # facing: toward target when standing, along motion when walking
        tgt = p.target
        tgt_xy = np.repeat(np.asarray(tgt[1], float)[None], len(t), 0) if isinstance(tgt, tuple) else roots[tgt][0]
        face = tgt_xy - XZ
        vel = np.gradient(XZ, axis=0)
        face = np.where(moving[:, None] > 0, vel, face)
        face /= np.linalg.norm(face, axis=1, keepdims=True) + 1e-9
        right = np.stack([face[:, 1], -face[:, 0]], axis=1)
        X = XZ[:, None, 0] + L[..., 0] * right[:, None, 0] + L[..., 2] * face[:, None, 0]
        Z = XZ[:, None, 1] + L[..., 0] * right[:, None, 1] + L[..., 2] * face[:, None, 1]
        P = np.stack([X, L[..., 1], Z], axis=-1)
        uv, depth = cam.project(P)
        present = (t >= p.t_enter) & (t < p.t_exit)
        frames = np.flatnonzero(present)
        hip_mid = 0.5 * (uv[:, KP["left_hip"]] + uv[:, KP["right_hip"]])
        sho_mid = 0.5 * (uv[:, KP["left_shoulder"]] + uv[:, KP["right_shoulder"]])
        scale_px = np.linalg.norm(sho_mid - hip_mid, axis=1)
        tr = _noisy_track(pid, frames, uv[frames], depth[frames], scale_px[frames], rng, cam)
        if tr is not None:
            tracks.append(tr)

    cones_ground = np.array([[-15.0, 18.0], [15.0, 18.0], [-20.0, 45.0], [20.0, 45.0]])
    calibration = {
        "mode": "homography",
        "image_points": cam.ground_to_image(cones_ground).round(1).tolist(),
        "ground_points_ft": (cones_ground / 0.3048).round(2).tolist(),
    }
    gt.sort(key=lambda g: g["time_s"])
    return SimSession(fps, duration_s, cam, players, tracks, gt, calibration)
