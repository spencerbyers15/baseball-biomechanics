"""FlatBuffer schema readers for MLB FieldVision binary segments.

Schemas extracted from gd.min.js (Gameday frontend bundle) on 2026-05-06.
Vtable field offsets are taken straight from the static add* methods in
each class (e.g. addUid is field 0 → vtable offset 4, addRootPos is field
1 → vtable offset 6, etc.).

These mirror only the fields we need to extract per-actor pelvis positions
and the data we'll later use for full forward-kinematic skeleton recovery.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from .flatbuf_runtime import ByteBuffer


# ────────────────────────────────────────────────────────────
# Vec3Wire — 12-byte inline struct (3 × float32)
# ────────────────────────────────────────────────────────────


@dataclass
class Vec3:
    x: float
    y: float
    z: float


def read_vec3(bb: ByteBuffer, pos: int) -> Vec3:
    return Vec3(
        bb.read_float32(pos),
        bb.read_float32(pos + 4),
        bb.read_float32(pos + 8),
    )


# ────────────────────────────────────────────────────────────
# ActorPoseWire — per-actor pose data within a frame
# Vtable layout (from JS bundle):
#   field 0 (offset  4): uid           uint32      (MLB player ID)
#   field 1 (offset  6): rootPos       Vec3 inline (pelvis world position)
#   field 2 (offset  8): packedQuats   [uint32]    (bone rotations, smallest-three)
#   field 3 (offset 10): nodeIds       [uint16]    (bone IDs, parallel to packedQuats)
#   field 4 (offset 12): ground        float32
#   field 5 (offset 14): apex          float32
#   field 6 (offset 16): scale         float32
#   field 7 (offset 18): batRootPos    Vec3 inline (bat handle position if present)
# ────────────────────────────────────────────────────────────


@dataclass
class ActorPose:
    uid: int
    rootPos: Optional[Vec3]
    packedQuats: list[int]
    nodeIds: list[int]
    ground: float
    apex: float
    scale: float
    batRootPos: Optional[Vec3]


def read_actor_pose(bb: ByteBuffer, pos: int) -> ActorPose:
    o_uid = bb.field_offset(pos, 4)
    o_root = bb.field_offset(pos, 6)
    o_quats = bb.field_offset(pos, 8)
    o_nodes = bb.field_offset(pos, 10)
    o_ground = bb.field_offset(pos, 12)
    o_apex = bb.field_offset(pos, 14)
    o_scale = bb.field_offset(pos, 16)
    o_bat = bb.field_offset(pos, 18)

    uid = bb.read_uint32(pos + o_uid) if o_uid else 0
    rootPos = read_vec3(bb, pos + o_root) if o_root else None

    packed_quats: list[int] = []
    if o_quats:
        v = bb.vector_data(pos + o_quats)
        n = bb.vector_len(pos + o_quats)
        packed_quats = [bb.read_uint32(v + 4 * i) for i in range(n)]

    node_ids: list[int] = []
    if o_nodes:
        v = bb.vector_data(pos + o_nodes)
        n = bb.vector_len(pos + o_nodes)
        node_ids = [bb.read_uint16(v + 2 * i) for i in range(n)]

    ground = bb.read_float32(pos + o_ground) if o_ground else 0.0
    apex = bb.read_float32(pos + o_apex) if o_apex else 0.0
    scale = bb.read_float32(pos + o_scale) if o_scale else 0.0
    batRootPos = read_vec3(bb, pos + o_bat) if o_bat else None

    return ActorPose(uid, rootPos, packed_quats, node_ids, ground, apex, scale, batRootPos)


# ────────────────────────────────────────────────────────────
# TrackingFrameWire — one frame
# Vtable (from JS bundle's static add* methods):
#   field 0  (offset  4): actorPoses   [ActorPose]
#   field 1  (offset  6): ballPosition Vec3 inline
#   field 2  (offset  8): gameEvents   [GameEvent]
#   field 3  (offset 10): trackedEvents [TrackedEvent]
#   field 4  (offset 12): inferredBat  (table)
#   field 5  (offset 14): ballPolynomials [BallPolynomial]
#   field 6  (offset 16): rawJoints    [RawJoint]
#   field 7  (offset 18): num          int32 (frame number)
#   field 8  (offset 20): time         float64
#   field 9  (offset 22): timestamp    string (ISO-8601)
#   field 10 (offset 24): isGap        int8
#   field 11 (offset 26): gapDuration  float64
# ────────────────────────────────────────────────────────────


@dataclass
class TrackingFrame:
    num: int
    time: float
    timestamp: Optional[str]
    isGap: bool
    gapDuration: float
    ballPosition: Optional[Vec3]
    actorPoses: list[ActorPose] = field(default_factory=list)


def read_tracking_frame(bb: ByteBuffer, pos: int) -> TrackingFrame:
    o_actors = bb.field_offset(pos, 4)
    o_ball = bb.field_offset(pos, 6)
    o_num = bb.field_offset(pos, 18)
    o_time = bb.field_offset(pos, 20)
    o_ts = bb.field_offset(pos, 22)
    o_gap = bb.field_offset(pos, 24)
    o_gap_dur = bb.field_offset(pos, 26)

    num = bb.read_int32(pos + o_num) if o_num else 0
    time_v = bb.read_float64(pos + o_time) if o_time else 0.0
    timestamp = bb.string(pos + o_ts) if o_ts else None
    isGap = bool(bb.read_int8(pos + o_gap)) if o_gap else False
    gap_dur = bb.read_float64(pos + o_gap_dur) if o_gap_dur else 0.0
    ballPos = read_vec3(bb, pos + o_ball) if o_ball else None

    actors: list[ActorPose] = []
    if o_actors:
        v = bb.vector_data(pos + o_actors)
        n = bb.vector_len(pos + o_actors)
        for i in range(n):
            elem_pos = bb.indirect(v + 4 * i)
            actors.append(read_actor_pose(bb, elem_pos))

    return TrackingFrame(num, time_v, timestamp, isGap, gap_dur, ballPos, actors)


# ────────────────────────────────────────────────────────────
# TrackingDataWire — the file root
# Vtable:
#   field 0 (offset 4): version    string
#   field 1 (offset 6): frames     [TrackingFrame]
#   field 2 (offset 8): limits     LimitsWire
#   field 3 (offset 10): eventKeyFrame  GameEventWire
# ────────────────────────────────────────────────────────────


@dataclass
class TrackingData:
    version: Optional[str]
    frames: list[TrackingFrame] = field(default_factory=list)


def read_tracking_data(data: bytes) -> TrackingData:
    bb = ByteBuffer(data)
    root_off = bb.read_int32(0)  # offset to root table from byte 0
    pos = root_off

    o_ver = bb.field_offset(pos, 4)
    o_frames = bb.field_offset(pos, 6)

    version = bb.string(pos + o_ver) if o_ver else None
    frames: list[TrackingFrame] = []
    if o_frames:
        v = bb.vector_data(pos + o_frames)
        n = bb.vector_len(pos + o_frames)
        for i in range(n):
            elem_pos = bb.indirect(v + 4 * i)
            frames.append(read_tracking_frame(bb, elem_pos))

    return TrackingData(version, frames)


# ────────────────────────────────────────────────────────────
# Packed-quaternion decoder (smallest-three encoding, 32-bit per quaternion)
# Layout: 2 bits for dropped-axis index | 10 bits y | 10 bits z | 10 bits w   (or similar)
# We don't yet know MLB's exact bit layout. The function below returns the raw
# uint32; bit-layout TBD once we visually validate via a known-good frame.
# ────────────────────────────────────────────────────────────


def unpack_smallest_three(packed: int) -> tuple[float, float, float, float]:
    """
    Decode a 32-bit packed unit quaternion using the standard smallest-three scheme:
    bits 30-31: index of the largest-magnitude component (the one omitted)
    bits 20-29: 10-bit signed component a (mapped to [-1/sqrt(2), +1/sqrt(2)])
    bits 10-19: 10-bit signed component b
    bits  0- 9: 10-bit signed component c

    Bit ordering above is a guess — may need bit-flip/mirror after validation.
    Returns (qx, qy, qz, qw) with the dropped component reconstructed.
    """
    INV_SQRT2 = 0.7071067811865476
    drop = (packed >> 30) & 0x3
    a_raw = (packed >> 20) & 0x3FF
    b_raw = (packed >> 10) & 0x3FF
    c_raw = packed & 0x3FF

    def _decode_signed_10(v: int) -> float:
        # Map 10-bit unsigned [0..1023] to signed range [-1/sqrt(2), +1/sqrt(2)]
        return ((v / 1023.0) * 2.0 - 1.0) * INV_SQRT2

    a = _decode_signed_10(a_raw)
    b = _decode_signed_10(b_raw)
    c = _decode_signed_10(c_raw)

    # Reconstruct dropped axis: q_largest = sqrt(1 - a^2 - b^2 - c^2)
    largest = max(0.0, 1.0 - a * a - b * b - c * c) ** 0.5

    components = [0.0, 0.0, 0.0, 0.0]
    components[drop] = largest
    other_indices = [i for i in range(4) if i != drop]
    components[other_indices[0]] = a
    components[other_indices[1]] = b
    components[other_indices[2]] = c

    return tuple(components)  # (x, y, z, w)
