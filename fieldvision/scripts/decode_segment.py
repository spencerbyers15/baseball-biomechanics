"""Decode a FieldVision .bin segment using the FlatBuffer schemas extracted
from gd.min.js. Outputs JSON of frame-by-frame actor poses (uid, rootPos,
quaternion count) so we can validate the parser before tackling forward
kinematics.

Usage:
    python scripts/decode_segment.py samples/binary_capture_823141/mlb_823141_segment_0.bin
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldvision.wire_schemas import read_tracking_data


def main(path: str) -> None:
    data = Path(path).read_bytes()
    print(f"file: {path}  ({len(data):,} bytes)")
    td = read_tracking_data(data)
    print(f"version: {td.version}")
    print(f"frames:  {len(td.frames)}")
    print()

    if not td.frames:
        print("No frames decoded.")
        return

    f0 = td.frames[0]
    print(f"--- Frame[0] ---")
    print(f"  num={f0.num}  time={f0.time:.3f}  ts={f0.timestamp}")
    print(f"  isGap={f0.isGap}  ballPos={f0.ballPosition}")
    print(f"  actors: {len(f0.actorPoses)}")
    for i, a in enumerate(f0.actorPoses):
        bones_with_data = len(a.nodeIds)
        print(f"    actor[{i}] uid={a.uid}  rootPos={a.rootPos}  "
              f"bones_tracked={bones_with_data}  "
              f"ground={a.ground:.3f} apex={a.apex:.3f} scale={a.scale:.3f}  "
              f"bat={a.batRootPos}")

    # Sanity: pelvis Y should be roughly waist-high (3 to 4 feet)
    print()
    print(f"--- Pelvis Y sanity check across frame[0] actors ---")
    for i, a in enumerate(f0.actorPoses):
        if a.rootPos:
            y = a.rootPos.y
            ok = 1.0 < y < 6.5
            print(f"  actor[{i}] uid={a.uid}  pelvis_y={y:7.3f}  {'✓' if ok else '✗'}")

    # Last frame
    fl = td.frames[-1]
    print()
    print(f"--- Frame[{len(td.frames) - 1}] (last) ---")
    print(f"  num={fl.num}  time={fl.time:.3f}  ts={fl.timestamp}  isGap={fl.isGap}")
    print(f"  actors: {len(fl.actorPoses)}")

    # Save full decoded structure as JSON for inspection
    out_path = Path(path).with_suffix(".decoded.json")
    out_data = {
        "version": td.version,
        "frame_count": len(td.frames),
        "frames": [
            {
                "num": f.num,
                "time": f.time,
                "timestamp": f.timestamp,
                "isGap": f.isGap,
                "ballPosition": (f.ballPosition.__dict__ if f.ballPosition else None),
                "actorPoses": [
                    {
                        "uid": a.uid,
                        "rootPos": (a.rootPos.__dict__ if a.rootPos else None),
                        "ground": a.ground,
                        "apex": a.apex,
                        "scale": a.scale,
                        "nodeIds": a.nodeIds,
                        "packedQuatsCount": len(a.packedQuats),
                        "batRootPos": (a.batRootPos.__dict__ if a.batRootPos else None),
                    }
                    for a in f.actorPoses
                ],
            }
            for f in td.frames
        ],
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print()
    print(f"Decoded JSON: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "samples/binary_capture_823141/mlb_823141_segment_0.bin")
