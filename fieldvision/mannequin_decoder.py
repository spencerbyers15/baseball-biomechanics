"""
MLB FieldVision Mannequin Skeletal Data Decoder
================================================
Decodes the binary .bin segments from:
  fieldvision-hls.mlbinfra.com/mannequin/{gamePk}/1.6.2/*.bin

These segments contain Hawk-Eye skeletal tracking data:
  - 103 bones per player at 30fps
  - 5-second segments (~150 frames)
  - Multiple players per segment

Usage:
  # Analyze a single segment's binary structure
  python mannequin_decoder.py analyze segment_954.bin

  # Decode all segments in a directory
  python mannequin_decoder.py decode ./captured_data/ --output joints.csv

  # Dump raw float values from a segment for manual inspection
  python mannequin_decoder.py dump segment_954.bin --offset 1200 --count 200

Requirements:
  pip install numpy pandas  (optional, for CSV export)
"""

import struct
import json
import os
import sys
import glob
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# Bone ID Map (from metadata.json)
# ═══════════════════════════════════════════════════════════

BONE_ID_MAP = {
    0: "joint_Pelvis", 1: "joint_HipMaster",
    2: "joint_HipRT", 3: "joint_KneeRT", 4: "joint_FootRT",
    5: "joint_BallRT", 6: "joint_ToeRT", 7: "joint_ToeRT_end",
    8: "joint_ThighRollRT", 9: "joint_ThighRollRT_end",
    10: "joint_HipLT", 11: "joint_KneeLT", 12: "joint_FootLT",
    13: "joint_BallLT", 14: "joint_ToeLT", 15: "joint_ToeLT_end",
    16: "joint_ThighRollLT", 17: "joint_ThighRollLT_end",
    18: "joint_TorsoA", 19: "joint_TorsoB",
    20: "joint_Neck", 21: "joint_Neck2", 22: "joint_Head",
    23: "joint_EyeRT", 24: "joint_EyeLT",
    25: "joint_ClavicleRT", 26: "joint_ShoulderRT",
    27: "joint_ElbowRT", 28: "joint_HandRT",
    29: "joint_WeaponRT", 30: "joint_WeaponRT_end",
    31: "joint_PinkyART", 32: "joint_PinkyBRT",
    33: "joint_PinkyCRT", 34: "joint_PinkyDRT", 35: "joint_PinkyDRT_end",
    36: "joint_RingART", 37: "joint_RingBRT",
    38: "joint_RingCRT", 39: "joint_RingDRT", 40: "joint_RingDRT_end",
    41: "joint_FingersART", 42: "joint_FingersBRT",
    43: "joint_FingersCRT", 44: "joint_FingersDRT", 45: "joint_FingersDRT_end",
    46: "joint_IndexART", 47: "joint_IndexBRT",
    48: "joint_IndexCRT", 49: "joint_IndexDRT", 50: "joint_IndexDRT_end",
    51: "joint_ThumbART", 52: "joint_ThumbBRT",
    53: "joint_ThumbCRT", 54: "joint_ThumbDRT", 55: "joint_ThumbDRT_end",
    56: "joint_WristRollRT", 57: "joint_WristRollRT_end",
    58: "joint_ForeArmRollRT", 59: "joint_ForeArmRollRT_end",
    60: "joint_UpperArmRollRT", 61: "joint_UpperArmRollRT_end",
    62: "joint_ClavicleRollRT", 63: "joint_ClavicleRollRT_end",
    64: "joint_ClavicleLT", 65: "joint_ShoulderLT",
    66: "joint_ElbowLT", 67: "joint_HandLT",
    68: "joint_WeaponLT", 69: "joint_WeaponLT_end",
    70: "joint_PinkyALT", 71: "joint_PinkyBLT",
    72: "joint_PinkyCLT", 73: "joint_PinkyDLT", 74: "joint_PinkyDLT_end",
    75: "joint_RingALT", 76: "joint_RingBLT",
    77: "joint_RingCLT", 78: "joint_RingDLT", 79: "joint_RingDLT_end",
    80: "joint_FingersALT", 81: "joint_FingersBLT",
    82: "joint_FingersCLT", 83: "joint_FingersDLT", 84: "joint_FingersDLT_end",
    85: "joint_IndexALT", 86: "joint_IndexBLT",
    87: "joint_IndexCLT", 88: "joint_IndexDLT", 89: "joint_IndexDLT_end",
    90: "joint_ThumbALT", 91: "joint_ThumbBLT",
    92: "joint_ThumbCLT", 93: "joint_ThumbDLT", 94: "joint_ThumbDLT_end",
    95: "joint_WristRollLT", 96: "joint_WristRollLT_end",
    97: "joint_ForeArmRollLT", 98: "joint_ForeArmRollLT_end",
    99: "joint_UpperArmRollLT", 100: "joint_UpperArmRollLT_end",
    101: "joint_ClavicleRollLT", 102: "joint_ClavicleRollLT_end",
}

BAT_BONE_MAP = {
    103: "bat_bottom", 104: "bat_handle", 105: "bat_body",
    106: "bat_midbody", 107: "bat_spot", 108: "bat_top",
}

# Key biomechanical joints (subset for quick analysis)
KEY_JOINTS = [
    0,   # Pelvis (root)
    2,   # HipRT
    3,   # KneeRT
    4,   # FootRT
    10,  # HipLT
    11,  # KneeLT
    12,  # FootLT
    18,  # TorsoA
    19,  # TorsoB
    20,  # Neck
    22,  # Head
    26,  # ShoulderRT
    27,  # ElbowRT
    28,  # HandRT
    65,  # ShoulderLT
    66,  # ElbowLT
    67,  # HandLT
]


class MannequinSegment:
    """Parser for a single .bin mannequin segment."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        with open(filepath, 'rb') as f:
            self.data = f.read()
        self.size = len(self.data)

    def read_uint32(self, offset):
        return struct.unpack_from('<I', self.data, offset)[0]

    def read_int32(self, offset):
        return struct.unpack_from('<i', self.data, offset)[0]

    def read_uint16(self, offset):
        return struct.unpack_from('<H', self.data, offset)[0]

    def read_float32(self, offset):
        return struct.unpack_from('<f', self.data, offset)[0]

    def read_float64(self, offset):
        return struct.unpack_from('<d', self.data, offset)[0]

    def get_header(self):
        """Parse the segment header."""
        header = {}
        header['root_offset'] = self.read_uint32(0)  # FlatBuffer root

        # Timestamps
        if self.size >= 48:
            header['start_time'] = self.read_float64(32)
            header['end_time'] = self.read_float64(40)
            header['duration'] = header['end_time'] - header['start_time']

            # Convert to human-readable
            st = datetime.fromtimestamp(header['start_time'], tz=timezone.utc)
            header['start_time_str'] = st.strftime('%Y-%m-%d %H:%M:%S.%f UTC')

        # Read uint32 values in header region (48-200) to find offsets
        header['header_uint32'] = []
        for i in range(48, min(200, self.size), 4):
            header['header_uint32'].append((i, self.read_uint32(i)))

        return header

    def find_player_offsets(self):
        """Find the descending offset table that points to player data blocks."""
        offsets = []
        # The offset table appears around bytes 64-200
        # Look for descending sequences of uint32 values
        for start in range(56, min(300, self.size - 4), 4):
            val = self.read_uint32(start)
            if 200 < val < self.size and val not in [o[1] for o in offsets]:
                offsets.append((start, val))

        # Filter to find the descending sequence (player offsets go high to low)
        if len(offsets) < 3:
            return offsets

        # Find the longest descending subsequence
        desc_runs = []
        current_run = [offsets[0]]
        for i in range(1, len(offsets)):
            if offsets[i][1] < current_run[-1][1]:
                current_run.append(offsets[i])
            else:
                if len(current_run) >= 3:
                    desc_runs.append(current_run)
                current_run = [offsets[i]]
        if len(current_run) >= 3:
            desc_runs.append(current_run)

        if desc_runs:
            # Return the longest descending run
            best = max(desc_runs, key=len)
            return best

        return offsets

    def scan_coordinate_triplets(self, start_offset=100, end_offset=None, min_val=-500, max_val=500):
        """Scan for xyz coordinate triplets in the data."""
        if end_offset is None:
            end_offset = min(self.size, start_offset + 5000)

        triplets = []
        for i in range(start_offset, end_offset - 12, 4):
            try:
                x = self.read_float32(i)
                y = self.read_float32(i + 4)
                z = self.read_float32(i + 8)

                # Filter for plausible coordinate values
                vals = [x, y, z]
                if all(min_val < v < max_val for v in vals):
                    if any(abs(v) > 0.01 for v in vals):  # Not all zeros
                        import math
                        if all(math.isfinite(v) for v in vals):
                            triplets.append({
                                'offset': i,
                                'x': round(x, 4),
                                'y': round(y, 4),
                                'z': round(z, 4)
                            })
            except struct.error:
                break

        return triplets

    def dump_floats(self, offset, count=100):
        """Dump sequential float32 values from a given offset."""
        values = []
        for i in range(count):
            pos = offset + i * 4
            if pos + 4 > self.size:
                break
            try:
                val = self.read_float32(pos)
                values.append((pos, round(val, 6)))
            except struct.error:
                break
        return values

    def analyze(self):
        """Full analysis of the segment structure."""
        header = self.get_header()
        player_offsets = self.find_player_offsets()
        triplets = self.scan_coordinate_triplets()

        print(f"\n{'='*60}")
        print(f"Segment: {self.filename} ({self.size:,} bytes)")
        print(f"{'='*60}")

        if 'start_time_str' in header:
            print(f"Time:     {header['start_time_str']}")
            print(f"Duration: {header.get('duration', 0):.1f}s")

        print(f"\nPlayer offset table ({len(player_offsets)} entries):")
        for table_offset, data_offset in player_offsets[:20]:
            print(f"  @{table_offset:4d} -> offset {data_offset:5d}")

        print(f"\nCoordinate triplets found: {len(triplets)}")
        if triplets:
            print("First 10 triplets:")
            for t in triplets[:10]:
                print(f"  @{t['offset']:5d}: ({t['x']:8.3f}, {t['y']:8.3f}, {t['z']:8.3f})")

        return {
            'header': header,
            'player_offsets': player_offsets,
            'triplets': triplets
        }


class MannequinDecoder:
    """Full decoder using metadata, labels, and manifest."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.metadata = None
        self.labels = None
        self.manifest = None
        self._load_schema()

    def _load_schema(self):
        meta_path = self.data_dir / 'metadata.json'
        labels_path = self.data_dir / 'labels.json'
        manifest_path = self.data_dir / 'manifest.json'

        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata: {len(self.metadata.get('boneIdMap', {}))} bones")

        if labels_path.exists():
            with open(labels_path) as f:
                self.labels = json.load(f)
            print(f"Loaded labels: {len(self.labels)} segment labels")

        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            records = self.manifest.get('records', [])
            print(f"Loaded manifest: {len(records)} records, status={self.manifest.get('status')}")

    def get_segment_label(self, segment_name):
        """Get the player info for a segment from labels.json."""
        # Extract segment number from filename (e.g., 'mannequin_954.bin' -> '954')
        num = segment_name.replace('mannequin_', '').replace('.bin', '')
        if self.labels and num in self.labels:
            return self.labels[num]
        return None

    def get_segment_time(self, segment_index):
        """Get the timestamp for a segment from manifest.json."""
        if self.manifest and 'records' in self.manifest:
            records = self.manifest['records']
            if segment_index < len(records):
                return records[segment_index]
        return None

    def decode_all(self, output_path=None):
        """Decode all .bin segments in the data directory."""
        bin_files = sorted(
            glob.glob(str(self.data_dir / 'mannequin_*.bin')) +
            glob.glob(str(self.data_dir / '*.bin'))
        )

        if not bin_files:
            print(f"No .bin files found in {self.data_dir}")
            return

        print(f"\nDecoding {len(bin_files)} segments...")
        all_results = []

        for filepath in bin_files:
            seg = MannequinSegment(filepath)
            result = seg.analyze()
            result['filename'] = os.path.basename(filepath)
            result['label'] = self.get_segment_label(os.path.basename(filepath))
            all_results.append(result)

        if output_path:
            self._export_results(all_results, output_path)

        return all_results

    def _export_results(self, results, output_path):
        """Export decoded results to JSON."""
        # Serialize (remove non-serializable items)
        export = []
        for r in results:
            entry = {
                'filename': r['filename'],
                'label': r.get('label'),
                'start_time': r['header'].get('start_time'),
                'duration': r['header'].get('duration'),
                'player_count': len(r['player_offsets']),
                'triplet_count': len(r['triplets']),
                'triplets_sample': r['triplets'][:50]
            }
            export.append(entry)

        with open(output_path, 'w') as f:
            json.dump(export, f, indent=2)
        print(f"Exported to {output_path}")


def cmd_analyze(filepath):
    """Analyze a single segment."""
    seg = MannequinSegment(filepath)
    seg.analyze()


def cmd_dump(filepath, offset=0, count=100):
    """Dump raw float values."""
    seg = MannequinSegment(filepath)
    values = seg.dump_floats(offset, count)
    print(f"\nFloat32 dump from offset {offset} ({count} values):")
    for pos, val in values:
        marker = ' <<' if abs(val) > 0.01 and abs(val) < 500 else ''
        print(f"  @{pos:5d}: {val:12.4f}{marker}")


def cmd_decode(data_dir, output=None):
    """Decode all segments in a directory."""
    decoder = MannequinDecoder(data_dir)
    decoder.decode_all(output_path=output)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'analyze' and len(sys.argv) >= 3:
        cmd_analyze(sys.argv[2])

    elif cmd == 'dump' and len(sys.argv) >= 3:
        offset = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == '--offset' else 0
        count = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[5] == '--count' else 200
        # Simpler arg parsing
        kwargs = {}
        args = sys.argv[3:]
        for i in range(0, len(args) - 1, 2):
            if args[i] == '--offset':
                kwargs['offset'] = int(args[i + 1])
            elif args[i] == '--count':
                kwargs['count'] = int(args[i + 1])
        cmd_dump(sys.argv[2], **kwargs)

    elif cmd == 'decode' and len(sys.argv) >= 3:
        output = None
        if '--output' in sys.argv:
            idx = sys.argv.index('--output')
            output = sys.argv[idx + 1]
        cmd_decode(sys.argv[2], output=output)

    else:
        print(__doc__)


if __name__ == '__main__':
    main()
