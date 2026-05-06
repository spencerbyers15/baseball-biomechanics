# MLB FieldVision Mannequin Skeletal Tracking — Reverse Engineering Toolkit

## What This Is

A toolkit for extracting real-time skeletal tracking data from MLB's Gameday 3D 
(FieldVision) system. Hawk-Eye's 12-camera array tracks **103 bones per player at 
30fps** during every MLB game. This data streams to your browser to drive the 3D 
player models — and these tools let you capture and analyze it.

## Architecture (Discovered via Live Reverse Engineering)

```
Hawk-Eye Cameras (12 per stadium)
        │
        ▼
fieldvision-hls.mlbinfra.com/mannequin/{gamePk}/1.6.2/
├── metadata.json      ← 103 bone names + 6 bat bones + venue info
├── labels.json        ← Maps segment# → {actor: MLB_PLAYER_ID, type: "fielder"|"batter"|...}
├── manifest.json      ← Segment timeline: [{index, startTime, duration:5, isGap}]
├── 0.bin              ← Binary skeletal data (5-second segments, ~400-600KB each)
├── 1.bin              
├── ...
└── N.bin              ← Segments increment every 5 seconds during the game
        │
        ▼
gd.@bvg_poser.min.js  (FieldVision "Poser" engine)
        │
        ▼
Three.js Scene Graph
├── World (Object3D)
│   ├── Ball
│   ├── StrikeZone
│   ├── Stadium (venue-specific GLB model)
│   └── Armature × 17 (one per tracked entity)
│       └── SkinnedMesh × ~10 (Arms, Jersey, Pants, Head, Shoes, etc.)
│           └── Skeleton (103 bones)
│               ├── joint_Pelvis (bone 0) ← ROOT
│               ├── joint_HipRT (bone 2) → KneeRT → FootRT → ToeRT
│               ├── joint_HipLT (bone 10) → KneeLT → FootLT → ToeLT
│               ├── joint_TorsoA (bone 18) → TorsoB → Neck → Head
│               ├── joint_ShoulderRT (bone 26) → ElbowRT → HandRT → Fingers...
│               ├── joint_ShoulderLT (bone 65) → ElbowLT → HandLT → Fingers...
│               └── (+ roll bones, weapon slots, eye bones)
        │
        ▼
Canvas (WebGL2, ANGLE Metal Renderer)
```

## Authentication

Mannequin endpoint requires two headers:
- `Authorization: Bearer <token>` (from MLB's auth system)
- `x-mannequin-client: <identifier>`

The capture script piggybacks on the page's existing authenticated requests.

## Quick Start

### Method 1: Scene Graph Capture (Recommended — decoded world-space positions)

1. Open any game in Gameday 3D view on mlb.com
2. Open DevTools console (F12 → Console)
3. Paste `mannequin_capture.js` and press Enter
4. The tool auto-captures schema files and binary segments
5. Then paste the `BoneCapture` block from the script to capture decoded bone positions
6. Run `BoneCapture.start(30)` to capture 30 seconds
7. Run `BoneCapture.download()` to save as JSON

### Method 2: Binary Segment Capture (Raw Hawk-Eye data)

1. Same setup as above
2. After pasting `mannequin_capture.js`:
   - `MannequinCapture.status()` — check progress
   - `MannequinCapture.downloadSchemaOnly()` — get metadata/labels/manifest
   - `MannequinCapture.downloadAll()` — get everything

### Analyzing the Data

```bash
# Full analysis with biomechanics summary
python bone_analyzer.py bone_capture_XXXX.json

# Export most active player to CSV
python bone_analyzer.py bone_capture_XXXX.json --csv output.csv

# Analyze specific player
python bone_analyzer.py bone_capture_XXXX.json --player 3

# Analyze raw binary segments
python mannequin_decoder.py analyze mannequin_954.bin
python mannequin_decoder.py dump mannequin_954.bin --offset 1200 --count 200
```

## Bone ID Map (103 Player Bones + 6 Bat Bones)

### Core Skeleton
| ID | Joint | ID | Joint |
|----|-------|----|-------|
| 0 | Pelvis (ROOT) | 18 | TorsoA |
| 1 | HipMaster | 19 | TorsoB |
| 2 | HipRT | 20 | Neck |
| 3 | KneeRT | 21 | Neck2 |
| 4 | FootRT | 22 | Head |
| 10 | HipLT | 23 | EyeRT |
| 11 | KneeLT | 24 | EyeLT |
| 12 | FootLT | | |

### Arms
| ID | Joint | ID | Joint |
|----|-------|----|-------|
| 25 | ClavicleRT | 64 | ClavicleLT |
| 26 | ShoulderRT | 65 | ShoulderLT |
| 27 | ElbowRT | 66 | ElbowLT |
| 28 | HandRT | 67 | HandLT |
| 29 | WeaponRT | 68 | WeaponLT |

### Fingers (RT/LT symmetric, IDs 31-55 / 70-94)
Each finger: A (base) → B → C → D → end
- Pinky, Ring, Middle (Fingers), Index, Thumb

### Roll Bones (IDs 8-9, 16-17, 56-63, 95-102)
ThighRoll, WristRoll, ForeArmRoll, UpperArmRoll, ClavicleRoll (RT/LT)

### Bat (IDs 103-108)
bottom, handle, body, midbody, spot, top

## Binary Segment Format (.bin)

Each 5-second segment is a custom binary format:
- **Bytes 0-3**: Root offset (uint32 LE) = 20
- **Bytes 32-39**: Start timestamp (float64 LE, Unix epoch seconds)
- **Bytes 40-47**: End timestamp (float64 LE)
- **Bytes 48-55**: Metadata (record count, duration marker)
- **Bytes ~64-200**: Player offset table (descending uint32 offsets)
- **Bytes ~200+**: Per-player data blocks with bone transforms

## Coordinate System

World-space coordinates appear to be in **feet** relative to the ballpark:
- Home plate is approximately at origin
- X axis: lateral (negative = 3B side, positive = 1B side)
- Y axis: vertical (varies by player stance)
- Z axis: depth (negative = toward outfield)

## What You Can Do With This

- **Pitching biomechanics**: Track arm slot, stride length, hip-shoulder separation
- **Hitting mechanics**: Bat path, swing plane, launch point
- **Fielding analysis**: First step quickness, route efficiency with body pose
- **Injury risk**: Arm angle consistency, fatigue detection via posture changes
- **Player comparison**: Overlay skeletal data between players/at-bats
