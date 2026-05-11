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
# SkeletalPlayerWire — raw Hawk-Eye joint positions for one player
# Vtable layout (from JS bundle, getRootAsSkeletalPlayerWire):
#   field 0 (offset  4): positionId    uint16
#   field 1 (offset  6): trackId       uint32
#   field 2 (offset  8): jointPositions [float32]  (3 floats per joint = x,y,z)
#   field 3 (offset 10): jointIds       [uint32]   (one per joint)
#   field 4 (offset 12): playerId       uint32     (MLB player ID — direct, no labels.json join)
#   field 5 (offset 14): jerseyNumber   uint32
#   field 6 (offset 16): roleId         uint32
# ────────────────────────────────────────────────────────────


@dataclass
class SkeletalPlayer:
    positionId: int
    trackId: int
    playerId: int
    jerseyNumber: int
    roleId: int
    jointIds: list[int]
    # joint world positions: list of (x, y, z) tuples in stadium feet, Y up
    jointPositions: list[tuple[float, float, float]]


def read_skeletal_player(bb: ByteBuffer, pos: int) -> SkeletalPlayer:
    o_pos_id = bb.field_offset(pos, 4)
    o_track = bb.field_offset(pos, 6)
    o_jpos = bb.field_offset(pos, 8)
    o_jids = bb.field_offset(pos, 10)
    o_player = bb.field_offset(pos, 12)
    o_jersey = bb.field_offset(pos, 14)
    o_role = bb.field_offset(pos, 16)

    positionId = bb.read_uint16(pos + o_pos_id) if o_pos_id else 0
    trackId = bb.read_uint32(pos + o_track) if o_track else 0
    playerId = bb.read_uint32(pos + o_player) if o_player else 0
    jerseyNumber = bb.read_uint32(pos + o_jersey) if o_jersey else 0
    roleId = bb.read_uint32(pos + o_role) if o_role else 0

    joint_ids: list[int] = []
    if o_jids:
        v = bb.vector_data(pos + o_jids)
        n = bb.vector_len(pos + o_jids)
        joint_ids = [bb.read_uint32(v + 4 * i) for i in range(n)]

    joint_positions: list[tuple[float, float, float]] = []
    if o_jpos:
        v = bb.vector_data(pos + o_jpos)
        n_floats = bb.vector_len(pos + o_jpos)
        # n_floats should be 3 * len(joint_ids)
        for i in range(n_floats // 3):
            x = bb.read_float32(v + 4 * (3 * i + 0))
            y = bb.read_float32(v + 4 * (3 * i + 1))
            z = bb.read_float32(v + 4 * (3 * i + 2))
            joint_positions.append((x, y, z))

    return SkeletalPlayer(
        positionId=positionId,
        trackId=trackId,
        playerId=playerId,
        jerseyNumber=jerseyNumber,
        roleId=roleId,
        jointIds=joint_ids,
        jointPositions=joint_positions,
    )


# ────────────────────────────────────────────────────────────
# TrackingBatPositionWire — bat orientation per frame
# Vtable layout (from JS bundle, getRootAsTrackingBatPositionWire):
#   field 0 (offset 4): headPosition   Vec3 inline (bat barrel/tip world pos)
#   field 1 (offset 6): handlePosition Vec3 inline (bat grip world pos)
# ────────────────────────────────────────────────────────────


@dataclass
class InferredBat:
    headPosition: Optional[Vec3]
    handlePosition: Optional[Vec3]


def read_inferred_bat(bb: ByteBuffer, pos: int) -> InferredBat:
    o_head = bb.field_offset(pos, 4)
    o_handle = bb.field_offset(pos, 6)
    head = read_vec3(bb, pos + o_head) if o_head else None
    handle = read_vec3(bb, pos + o_handle) if o_handle else None
    return InferredBat(headPosition=head, handlePosition=handle)


# ────────────────────────────────────────────────────────────
# GameEventWire — wrapper for a discrete game event in a frame.
# Vtable from extract_event_offsets.py output:
#   field 0 (offset 4): dataType    uint8   (union discriminator; 7 = PlayEvent)
#   field 1 (offset 6): data        indirect to typed sub-table (e.g. PlayEventDataWire)
#   field 2 (offset 8): time        float64
#   field 3 (offset 10): isKeyFramed int8
# We only decode PlayEventDataWire (dataType=7) for play_id; statsapi
# provides count/atbat/handedness without needing wire decode.
# ────────────────────────────────────────────────────────────


@dataclass
class GameEvent:
    dataType: int
    time: Optional[float]
    isKeyFramed: bool
    playId: Optional[str]   # populated when dataType==7 (PlayEvent), else None


def read_game_event(bb: ByteBuffer, pos: int) -> GameEvent:
    o_dt = bb.field_offset(pos, 4)
    o_data = bb.field_offset(pos, 6)
    o_time = bb.field_offset(pos, 8)
    o_kf = bb.field_offset(pos, 10)

    dt = bb.read_uint8(pos + o_dt) if o_dt else 0
    time_v = bb.read_float64(pos + o_time) if o_time else None
    kf = bool(bb.read_int8(pos + o_kf)) if o_kf else False
    play_id: Optional[str] = None
    if dt == 7 and o_data:
        # PlayEventDataWire is at the indirect offset; PlayId is its vtoff 8
        pe_pos = bb.indirect(pos + o_data)
        o_pid = bb.field_offset(pe_pos, 8)
        if o_pid:
            play_id = bb.string(pe_pos + o_pid)
    return GameEvent(dataType=dt, time=time_v, isKeyFramed=kf, playId=play_id)


# ────────────────────────────────────────────────────────────
# TrackedEventWire — flat 14-field record. Self-discriminating via eventType (string).
# We extract just the fields needed for pitch segmentation: eventType, x, y, z.
# Other fields exist (timestamp, batSide, pitchHand, atBatNumber, pitchNumber,
# pickoffNumber, szTop, szBot, position, eventTypeId) but are sentinel
# (-1 / "not-set") for the events we observe in practice.
# ────────────────────────────────────────────────────────────


@dataclass
class TrackedEvent:
    eventType: Optional[str]
    eventTypeId: Optional[int]
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    timestamp: Optional[str]


def read_tracked_event(bb: ByteBuffer, pos: int) -> TrackedEvent:
    o_ts = bb.field_offset(pos, 4)
    o_etype = bb.field_offset(pos, 20)
    o_x = bb.field_offset(pos, 22)
    o_y = bb.field_offset(pos, 24)
    o_z = bb.field_offset(pos, 26)
    o_etid = bb.field_offset(pos, 30)

    ts = bb.string(pos + o_ts) if o_ts else None
    etype = bb.string(pos + o_etype) if o_etype else None
    x = bb.read_float32(pos + o_x) if o_x else None
    y = bb.read_float32(pos + o_y) if o_y else None
    z = bb.read_float32(pos + o_z) if o_z else None
    etid = bb.read_int8(pos + o_etid) if o_etid else None
    return TrackedEvent(eventType=etype, eventTypeId=etid, x=x, y=y, z=z, timestamp=ts)


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
    rawJoints: list[SkeletalPlayer] = field(default_factory=list)
    inferredBat: Optional[InferredBat] = None
    gameEvents: list[GameEvent] = field(default_factory=list)        # ← new
    trackedEvents: list[TrackedEvent] = field(default_factory=list)  # ← new


def read_tracking_frame(bb: ByteBuffer, pos: int) -> TrackingFrame:
    o_actors = bb.field_offset(pos, 4)
    o_ball = bb.field_offset(pos, 6)
    o_inferred = bb.field_offset(pos, 12)  # inferredBat (TrackingBatPositionWire)
    o_raw = bb.field_offset(pos, 16)  # rawJoints (SkeletalPlayerWire vector)
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

    raw: list[SkeletalPlayer] = []
    if o_raw:
        v = bb.vector_data(pos + o_raw)
        n = bb.vector_len(pos + o_raw)
        for i in range(n):
            elem_pos = bb.indirect(v + 4 * i)
            raw.append(read_skeletal_player(bb, elem_pos))

    inferred = None
    if o_inferred:
        # inferredBat is referenced via __indirect (offset to a separate table)
        inferred = read_inferred_bat(bb, bb.indirect(pos + o_inferred))

    o_ge = bb.field_offset(pos, 8)
    o_te = bb.field_offset(pos, 10)

    game_events: list[GameEvent] = []
    if o_ge:
        v = bb.vector_data(pos + o_ge)
        n = bb.vector_len(pos + o_ge)
        for i in range(n):
            elem_pos = bb.indirect(v + 4 * i)
            game_events.append(read_game_event(bb, elem_pos))

    tracked_events: list[TrackedEvent] = []
    if o_te:
        v = bb.vector_data(pos + o_te)
        n = bb.vector_len(pos + o_te)
        for i in range(n):
            elem_pos = bb.indirect(v + 4 * i)
            tracked_events.append(read_tracked_event(bb, elem_pos))

    return TrackingFrame(num, time_v, timestamp, isGap, gap_dur, ballPos,
                         actors, raw, inferred, game_events, tracked_events)


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
    Decode a 32-bit packed unit quaternion. EXACT reproduction of the
    JS encoder in gd.@bvg_poser.min.js (zP.unpack):

      bits 0-1 : omitted-axis index i (0=x, 1=y, 2=z, 3=w)
      bits 2-11: component at slot (i+3)%4   (10-bit signed)
      bits 12-21: component at slot (i+2)%4   (10-bit signed)
      bits 22-31: component at slot (i+1)%4   (10-bit signed)

    10-bit decoding (sign-magnitude with sign bit at position 9):
      if msb set: value -= 512 from the masked-off form
      result = (raw / 511) * maxValue

    maxValue = 0.7072 (≈ 1/sqrt(2)). Reconstructed component[i] = sqrt(1 - sum²).

    Returns (qx, qy, qz, qw).
    """
    MAX_VALUE = 0.7072
    i = packed & 0x3
    shifts = (22, 12, 2)
    components = [0.0, 0.0, 0.0, 0.0]
    sum_sq = 0.0
    for s in range(3):
        raw = (packed >> shifts[s]) & 0x3FF
        # 10-bit sign-magnitude per the JS dequantizeFloat
        if raw & 0x200:  # 512 (sign bit at position 9)
            raw ^= 0x200
            raw -= 512
        # raw is now in [-512, +511]
        val = (raw / 511.0) * MAX_VALUE
        components[(i + s + 1) % 4] = val
        sum_sq += val * val
    # Reconstruct the omitted (largest-magnitude) component
    components[i] = max(0.0, 1.0 - sum_sq) ** 0.5
    # Canonicalize sign so w >= 0 (matches the JS jB.negate behavior)
    if components[3] < 0:
        components = [-c for c in components]
    return tuple(components)  # (x, y, z, w)
