"""Skeletal hierarchy + forward kinematics for the 20 tracked bones.

REST POSE SOURCE: extracted from MLB's actual generic-lod.gltf
(fv-assets.mlb.com/models/generic/generic-lod.gltf). All translations are
the GLTF's parent-relative offsets converted from centimeters to feet.
All rest-pose rotations are taken verbatim from the GLTF.

Coordinate convention (GLTF, matches MLB stadium frame for rendering):
  X: lateral, +X = player's LEFT (note: opposite of anatomical "RT" naming)
  Y: vertical, +Y = up (away from ground)
  Z: depth

Each bone's local +Y axis points along the bone toward the child — so
KneeRT's translation (0, 1.31, 0) ft means the knee is "down the leg" 1.31
ft from the right hip in the hip's local frame, where the hip's local Y
points along the leg.

Animation rotations from packedQuats are applied as a POST-multiply on
top of the rest rotation: final_local_R = rest_R @ animated_R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CM_TO_FT = 1.0 / 30.48


# Canonical bone IDs (from metadata.json's boneIdMap)
BONE_NAMES = {
    0: "Pelvis", 1: "HipMaster",
    2: "HipRT", 3: "KneeRT", 4: "FootRT",
    10: "HipLT", 11: "KneeLT", 12: "FootLT",
    18: "TorsoA", 19: "TorsoB", 20: "Neck", 21: "Neck2",
    25: "ClavicleRT", 26: "ShoulderRT", 27: "ElbowRT", 28: "HandRT",
    64: "ClavicleLT", 65: "ShoulderLT", 66: "ElbowLT", 67: "HandLT",
}

TRACKED_NODE_IDS = [0, 1, 2, 3, 4, 10, 11, 12, 18, 19, 20, 21,
                    25, 26, 27, 28, 64, 65, 66, 67]


# Parent-child hierarchy (from GLTF)
PARENTS = {
    0: None,           # Pelvis is the root (its parent in GLTF is joint_Char, ignored)
    1: 0,              # HipMaster from Pelvis
    2: 1,              # HipRT from HipMaster
    3: 2,              # KneeRT from HipRT
    4: 3,              # FootRT from KneeRT
    10: 1,             # HipLT from HipMaster
    11: 10,            # KneeLT from HipLT
    12: 11,            # FootLT from KneeLT
    18: 0,             # TorsoA from Pelvis  (NOT HipMaster — corrected from GLTF)
    19: 18,            # TorsoB from TorsoA
    20: 19,            # Neck from TorsoB
    21: 20,            # Neck2 from Neck
    25: 19,            # ClavicleRT from TorsoB
    26: 25,            # ShoulderRT
    27: 26,            # ElbowRT
    28: 27,            # HandRT
    64: 19,            # ClavicleLT from TorsoB
    65: 64,            # ShoulderLT
    66: 65,            # ElbowLT
    67: 66,            # HandLT
}


# Rest-pose translations (parent-frame), copied from generic-lod.gltf in cm
# then converted to feet. Pelvis is the root so its translation is irrelevant
# (rootPos in the wire data overrides it).
_REST_TRANSLATION_CM = {
    0:  (0.0, 99.0, 0.0),
    1:  (0.0, -3.0, 0.0),
    2:  (-8.9085, 3.2819, 0.0),
    3:  (-0.0004, 40.0386, -0.0000),
    4:  (0.0, 43.1506, -0.0000),
    10: (8.9085, 3.2819, 0.0),
    11: (0.0004, 40.0386, -0.0000),
    12: (0.0, 43.1506, -0.0000),
    18: (-0.0000, 7.1587, -0.0000),
    19: (0.1480, 16.8565, -0.0000),
    20: (0.0001, 25.0525, -0.0000),
    21: (-0.0001, 4.0671, 0.0),
    25: (2.9598, 16.8258, -4.1050),
    26: (-0.0000, 17.2647, 0.0002),
    27: (-0.0000, 23.0000, 0.0),
    28: (0.0, 24.8136, -0.0001),
    64: (2.9598, 16.8260, 4.1049),
    65: (0.0, 17.2647, 0.0003),
    66: (0.0, 23.0000, 0.0),
    67: (0.0, 24.8134, 0.0001),
}
REST_OFFSET = {bid: np.array([t[0]*CM_TO_FT, t[1]*CM_TO_FT, t[2]*CM_TO_FT])
               for bid, t in _REST_TRANSLATION_CM.items()}


# Rest-pose rotations (parent-frame), copied from generic-lod.gltf, (x, y, z, w).
REST_ROTATION = {
    0:  (0.000, 0.707, 0.000, 0.707),
    1:  (-0.707, -0.000, -0.707, 0.000),
    2:  (0.018, -0.706, 0.045, 0.707),
    3:  (-0.006, -0.006, -0.076, 0.997),
    4:  (0.532, -0.466, 0.466, 0.532),
    10: (0.018, 0.706, -0.045, 0.707),
    11: (-0.006, 0.006, 0.076, 0.997),
    12: (-0.466, -0.532, -0.532, 0.466),
    18: (-0.000, -0.000, 0.026, 1.000),
    19: (-0.000, 0.000, -0.058, 0.998),
    20: (0.000, 0.000, 0.179, 0.984),
    21: (0.000, 0.000, -0.107, 0.994),
    25: (-0.785, -0.034, 0.009, 0.619),
    26: (-0.220, -0.031, 0.017, 0.975),
    27: (-0.000, -0.000, 0.078, 0.997),
    28: (0.000, -0.000, -0.000, 1.000),
    64: (0.009, -0.619, -0.785, 0.034),
    65: (-0.220, 0.031, -0.017, 0.975),
    66: (-0.000, 0.000, -0.078, 0.997),
    67: (-0.000, 0.000, -0.000, 1.000),
}


# Stick-figure connections — bone segments to draw between joints
SKELETON_CONNECTIONS = [
    (0, 1),    # pelvis ↔ hipmaster
    (1, 2), (2, 3), (3, 4),       # right leg
    (1, 10), (10, 11), (11, 12),  # left leg
    (0, 18), (18, 19),            # spine lower (pelvis -> torsoA -> torsoB)
    (19, 20), (20, 21),           # neck + head
    (19, 25), (25, 26), (26, 27), (27, 28),  # right arm
    (19, 64), (64, 65), (65, 66), (66, 67),  # left arm
]


# ────────────────────────────────────────────────────────────
# Quaternion math
# ────────────────────────────────────────────────────────────


def quat_to_rot_matrix(q):
    """Convert (x,y,z,w) quaternion to a 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s*(y*y + z*z), s*(x*y - w*z),     s*(x*z + w*y)],
            [s*(x*y + w*z),     1 - s*(x*x + z*z), s*(y*z - w*x)],
            [s*(x*z - w*y),     s*(y*z + w*x),     1 - s*(x*x + y*y)],
        ]
    )


def quat_multiply(a, b):
    """Quaternion multiplication, (x,y,z,w) convention. result = a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )


# ────────────────────────────────────────────────────────────
# Forward kinematics
# ────────────────────────────────────────────────────────────


@dataclass
class WorldSkeleton:
    bone_world_pos: dict[int, np.ndarray]


def forward_kinematics(
    root_pos: tuple[float, float, float],
    scale: float,
    node_ids: list[int],
    quats_xyzw: list[tuple[float, float, float, float]],
    compose_with_rest: bool = False,
) -> WorldSkeleton:
    """Apply per-bone local rotations to the GLTF rest pose.

    For each bone:
      effective_local_rotation = (rest_rotation @ animated_rotation) if compose_with_rest
                               else animated_rotation
      world_R[bone] = world_R[parent] @ effective_local_rotation
      world_pos[bone] = world_pos[parent] + world_R[parent] @ rest_translation[bone]

    `root_pos` overrides the Pelvis world position (the rootPos field from
    the wire data). The Pelvis local rotation is taken from the data
    (animated_rotation for bone 0).
    """
    quat_by_bone = dict(zip(node_ids, quats_xyzw))

    world_R: dict[int, np.ndarray] = {}
    world_pos: dict[int, np.ndarray] = {}

    def depth(b: int) -> int:
        d = 0
        while PARENTS.get(b) is not None:
            b = PARENTS[b]
            d += 1
        return d

    sorted_bones = sorted(REST_OFFSET.keys(), key=depth)

    for b in sorted_bones:
        parent = PARENTS[b]
        local_offset = REST_OFFSET[b] * scale

        rest_q = REST_ROTATION.get(b, (0.0, 0.0, 0.0, 1.0))
        anim_q = quat_by_bone.get(b, (0.0, 0.0, 0.0, 1.0))
        if compose_with_rest:
            local_q = quat_multiply(rest_q, anim_q)
        else:
            local_q = anim_q
        local_R = quat_to_rot_matrix(local_q)

        if parent is None:
            world_pos[b] = np.array(root_pos, dtype=float)
            world_R[b] = local_R
        else:
            world_pos[b] = world_pos[parent] + world_R[parent] @ local_offset
            world_R[b] = world_R[parent] @ local_R

    return WorldSkeleton(bone_world_pos=world_pos)
