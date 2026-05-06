"""Reverse-engineering aid: dump the structure of a .bin segment so we can
figure out the per-bone payload format. The previous session decoded the
header (root_offset=20, timestamps at bytes 32-47, descending player offset
table around bytes 64-200) but never finished the per-bone payload.

This dumps each player block's first 64 bytes as both hex and float32 so
patterns are visible.
"""

import struct
import sys
from pathlib import Path


def main(filepath: str) -> None:
    data = Path(filepath).read_bytes()
    print(f"file: {filepath}")
    print(f"size: {len(data):,} bytes")
    print()

    # Header
    root_offset = struct.unpack_from("<I", data, 0)[0]
    start_ts = struct.unpack_from("<d", data, 32)[0]
    end_ts = struct.unpack_from("<d", data, 40)[0]
    print(f"root_offset (byte 0): {root_offset}")
    print(f"start_ts:             {start_ts:.3f}  ({struct.unpack_from('<d', data, 32)[0]})")
    print(f"end_ts:               {end_ts:.3f}")
    print(f"duration:             {end_ts - start_ts:.3f}s")
    print()

    # Bytes 48-71: misc metadata after timestamps
    print("Bytes 48-71 (24 bytes after timestamps):")
    print("  hex:      " + data[48:72].hex(" "))
    print("  uint32x6: " + " ".join(str(struct.unpack_from('<I', data, 48 + i)[0]) for i in range(0, 24, 4)))
    print("  int32x6:  " + " ".join(str(struct.unpack_from('<i', data, 48 + i)[0]) for i in range(0, 24, 4)))
    print("  float32x6:" + " ".join(f"{struct.unpack_from('<f', data, 48 + i)[0]:8.3f}" for i in range(0, 24, 4)))
    print()

    # Find offset table — uint32s starting around byte 72
    # Walk forward, collecting plausible offsets (1 <= val < filesize)
    offsets: list[tuple[int, int]] = []
    pos = 72
    while pos + 4 <= 220:
        v = struct.unpack_from("<I", data, pos)[0]
        if 100 < v < len(data):
            offsets.append((pos, v))
        else:
            break
        pos += 4
    print(f"Offset table ({len(offsets)} entries):")
    for table_pos, target in offsets:
        print(f"  @{table_pos:4d}  ->  {target}")
    print()

    # Sort target offsets ascending; each block runs from offset[i] to offset[i+1]
    targets = sorted(t for _, t in offsets)
    targets.append(len(data))  # last block extends to EOF

    print("Player block sizes (ascending):")
    for i in range(len(targets) - 1):
        start, end = targets[i], targets[i + 1]
        print(f"  block[{i}]  bytes {start}..{end}  ({end - start} bytes)")
    print()

    # Dump first 80 bytes of each block as float32 + uint32 + hex
    for i in range(min(3, len(targets) - 1)):
        start = targets[i]
        end = min(targets[i + 1], start + 80)
        print(f"=== Block[{i}] @ {start} (first {end - start} bytes) ===")
        n_floats = (end - start) // 4
        for j in range(n_floats):
            off = start + j * 4
            f = struct.unpack_from("<f", data, off)[0]
            i32 = struct.unpack_from("<i", data, off)[0]
            u32 = struct.unpack_from("<I", data, off)[0]
            hex_chunk = data[off:off + 4].hex()
            note = ""
            if abs(f) < 1000 and abs(f) > 0.0001:
                note = "  ← plausible float"
            elif 0 < u32 < 10_000_000:
                note = f"  ← uint {u32}"
            print(f"  +{j * 4:3d} ({off:5d})  hex={hex_chunk}  f32={f:14.4f}  i32={i32:12d}{note}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/binary_capture_823141/mlb_823141_segment_0.bin")
