# Field offsets extracted from gd.@bvg_poser.min.js (1,714,355 chars)

## GameEventWire

| vtable_offset | field |
|---|---|
| 4 | DataType |
| 6 | Data |
| 8 | Time |
| 10 | IsKeyFramed |

## TrackedEventWire

| vtable_offset | field |
|---|---|
| 4 | Timestamp |
| 6 | BatSide |
| 8 | PitchHand |
| 10 | AtBatNumber |
| 12 | PitchNumber |
| 14 | PickoffNumber |
| 16 | SzTop |
| 18 | SzBot |
| 20 | EventType |
| 22 | X |
| 24 | Y |
| 26 | Z |
| 28 | Position |
| 30 | EventTypeId |

## PlayEventDataWire

| vtable_offset | field |
|---|---|
| 4 | Action |
| 6 | Index |
| 8 | PlayId |
| 10 | Strikezone |

## CountEventDataWire

| vtable_offset | field |
|---|---|
| 4 | Balls |
| 6 | Strikes |
| 8 | Outs |

## AtBatEventDataWire

| vtable_offset | field |
|---|---|
| 4 | Action |
| 6 | Index |
| 8 | BattingOrderIndex |
| 10 | BatterId |

## HandedEventDataWire

| vtable_offset | field |
|---|---|
| 4 | Action |
| 6 | Side |
| 8 | ActorId |

## BallPitchDataWire

| vtable_offset | field |
|---|---|
| 4 | Speed |
| 6 | Type |
| 8 | SzTop |
| 10 | SzBot |
| 12 | PitchType |
| 14 | ReleaseData |
| 16 | TrajectoryData |

## BatImpactEventDataWire

  (no static add* methods found -- confirmed empty table: startObject(0), zero fields, marker event only.)

## InningEventDataWire

| vtable_offset | field |
|---|---|
| 4 | CurrentInning |
| 6 | Action |
| 8 | BattingTeamId |
| 10 | FieldingTeamId |

---

## GameEvent dataType union dispatch

Source: `yM` enum + `_M()` switch in bundle. `GameEventWire.dataType()` reads a uint8 at vtable_offset 4.

| dataType | class |
|---|---|
| 0 | NONE |
| 1 | CountEventDataWire |
| 2 | TeamScoreEventDataWire |
| 3 | BattingOrderEventDataWire |
| 4 | LiveActionEventDataWire |
| 5 | InningEventDataWire |
| 6 | AtBatEventDataWire |
| 7 | PlayEventDataWire |
| 8 | HandedEventDataWire |
| 9 | PositionAssignmentEventDataWire |
| 10 | GumboTimecodeEventDataWire |
| 11 | StatusEventDataWire |
| 12 | BatImpactEventDataWire |
| 13 | HighFrequencyBatMarkerEventDataWire |
| 14 | ABSEventDataWire |

## TrackedEvent dataType union dispatch

Source: `VT` enum + `jT()` switch in bundle. `TrackedEventWire` does not carry a union `data` field — instead, the tracked-event type is indicated by a separate `dataType` field on the parent `TrackingFrameWire` tracked-events vector. The tracked event payload type enum is:

| dataType | class |
|---|---|
| 0 | NONE |
| 1 | BallBounceDataWire |
| 2 | BallHitDataWire |
| 3 | BallHitLaunchWire |
| 4 | BallHitRefinedWire |
| 5 | BallPitchDataWire |
| 6 | BallPitchRefinedWire |
| 7 | BallThrowData |
| 8 | BallPickOffDataWire |
| 9 | BallPitchReleasePointWire |
| 10 | BallPitchSpinWire |

