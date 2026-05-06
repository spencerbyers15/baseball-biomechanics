"""
MLB FieldVision Bone Capture Analyzer
=====================================
Analyzes the JSON bone capture data extracted from Gameday 3D.

Usage:
  python bone_analyzer.py path/to/bone_capture_XXXX.json

Outputs:
  - Summary statistics
  - Per-player joint trajectories  
  - CSV export of key joint positions over time
  - Velocity/acceleration analysis for biomechanics
"""

import json
import csv
import sys
import math
import os
from collections import defaultdict
from pathlib import Path


BONE_ID_MAP = {
    0: "Pelvis", 1: "HipMaster", 2: "HipRT", 3: "KneeRT", 4: "FootRT",
    5: "BallRT", 6: "ToeRT", 10: "HipLT", 11: "KneeLT", 12: "FootLT",
    13: "BallLT", 14: "ToeLT", 18: "TorsoA", 19: "TorsoB",
    20: "Neck", 21: "Neck2", 22: "Head", 23: "EyeRT", 24: "EyeLT",
    25: "ClavicleRT", 26: "ShoulderRT", 27: "ElbowRT", 28: "HandRT",
    29: "WeaponRT", 64: "ClavicleLT", 65: "ShoulderLT",
    66: "ElbowLT", 67: "HandLT", 68: "WeaponLT",
}

# The 17 most biomechanically important joints
KEY_JOINTS = {
    0: "Pelvis", 2: "HipRT", 3: "KneeRT", 4: "FootRT",
    10: "HipLT", 11: "KneeLT", 12: "FootLT",
    18: "TorsoA", 19: "TorsoB", 20: "Neck", 22: "Head",
    26: "ShoulderRT", 27: "ElbowRT", 28: "HandRT",
    65: "ShoulderLT", 66: "ElbowLT", 67: "HandLT",
}


def load_capture(filepath):
    """Load a bone capture JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"Loaded: {filepath}")
    print(f"  Frames: {data.get('frameCount', len(data.get('frames', [])))}")
    print(f"  Players/frame: {data.get('playerCountPerFrame', '?')}")
    
    # Use embedded boneIdMap if available, otherwise use default
    bone_map = data.get('boneIdMap', BONE_ID_MAP)
    
    return data['frames'], bone_map


def distance_3d(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def velocity_3d(p1, p2, dt):
    """Velocity vector between two points over time dt."""
    if dt == 0:
        return [0, 0, 0]
    return [(b - a) / dt for a, b in zip(p1, p2)]


def speed_3d(p1, p2, dt):
    """Speed (scalar) between two points."""
    return distance_3d(p1, p2) / dt if dt > 0 else 0


def analyze_player_motion(frames, player_idx):
    """Analyze a single player's motion across all frames."""
    positions = defaultdict(list)  # bone_id -> list of [x, y, z]
    timestamps = []
    
    for frame in frames:
        if player_idx >= len(frame['players']):
            continue
        player = frame['players'][player_idx]
        timestamps.append(frame['timestamp'])
        
        bones = player['bonePositions']
        for bone_id_str, pos in bones.items():
            bone_id = int(bone_id_str)
            positions[bone_id].append(pos)
    
    return positions, timestamps


def compute_joint_velocities(positions, timestamps, bone_id):
    """Compute velocity time series for a specific joint."""
    pts = positions.get(bone_id, [])
    if len(pts) < 2:
        return []
    
    velocities = []
    for i in range(1, len(pts)):
        dt = (timestamps[i] - timestamps[i-1]) / 1000.0  # ms -> seconds
        if dt > 0:
            spd = speed_3d(pts[i-1], pts[i], dt)
            velocities.append({
                'frame': i,
                'timestamp': timestamps[i],
                'speed_fps': spd,          # feet per second
                'speed_mph': spd * 0.6818, # convert ft/s to mph
                'position': pts[i]
            })
    return velocities


def compute_joint_angles(positions, frame_idx, joint_a, joint_b, joint_c):
    """Compute angle at joint_b formed by joint_a-joint_b-joint_c."""
    pa = positions.get(joint_a, [])
    pb = positions.get(joint_b, [])
    pc = positions.get(joint_c, [])
    
    if frame_idx >= len(pa) or frame_idx >= len(pb) or frame_idx >= len(pc):
        return None
    
    a, b, c = pa[frame_idx], pb[frame_idx], pc[frame_idx]
    
    # Vectors BA and BC
    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]
    
    # Dot product and magnitudes
    dot = sum(ba[i] * bc[i] for i in range(3))
    mag_ba = math.sqrt(sum(x**2 for x in ba))
    mag_bc = math.sqrt(sum(x**2 for x in bc))
    
    if mag_ba == 0 or mag_bc == 0:
        return None
    
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def find_most_active_player(frames):
    """Find the player with the most movement (likely the pitcher or batter)."""
    max_movement = 0
    most_active = 0
    
    for pidx in range(len(frames[0]['players'])):
        total_movement = 0
        for i in range(1, min(len(frames), 30)):  # Check first 30 frames
            p_curr = frames[i]['players'][pidx]['bonePositions']
            p_prev = frames[i-1]['players'][pidx]['bonePositions']
            
            # Sum movement of hand joints (most dynamic)
            for bone_id in ['28', '67']:  # HandRT, HandLT
                if bone_id in p_curr and bone_id in p_prev:
                    total_movement += distance_3d(p_curr[bone_id], p_prev[bone_id])
        
        if total_movement > max_movement:
            max_movement = total_movement
            most_active = pidx
    
    return most_active, max_movement


def export_csv(frames, player_idx, output_path, joints=None):
    """Export joint positions to CSV for a specific player."""
    if joints is None:
        joints = KEY_JOINTS
    
    headers = ['frame', 'timestamp_ms']
    for bone_id, name in sorted(joints.items()):
        headers.extend([f'{name}_x', f'{name}_y', f'{name}_z'])
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for frame_idx, frame in enumerate(frames):
            if player_idx >= len(frame['players']):
                continue
            
            row = [frame_idx, frame['timestamp']]
            player = frame['players'][player_idx]
            bones = player['bonePositions']
            
            for bone_id, name in sorted(joints.items()):
                pos = bones.get(str(bone_id), [0, 0, 0])
                row.extend([round(v, 4) for v in pos])
            
            writer.writerow(row)
    
    print(f"Exported {len(frames)} frames to {output_path}")


def print_summary(frames, bone_map):
    """Print a comprehensive summary of the capture."""
    print(f"\n{'='*70}")
    print(f"CAPTURE SUMMARY")
    print(f"{'='*70}")
    
    n_frames = len(frames)
    n_players = len(frames[0]['players']) if frames else 0
    
    # Time range
    t_start = frames[0]['timestamp']
    t_end = frames[-1]['timestamp']
    duration_s = (t_end - t_start) / 1000.0
    fps = n_frames / duration_s if duration_s > 0 else 0
    
    print(f"Frames:        {n_frames}")
    print(f"Duration:      {duration_s:.2f}s")
    print(f"Frame rate:    {fps:.1f} fps")
    print(f"Players:       {n_players}")
    print(f"Bones/player:  {len(frames[0]['players'][0]['bonePositions']) if n_players > 0 else 0}")
    
    # Find most active player
    most_active, movement = find_most_active_player(frames)
    print(f"\nMost active player: #{most_active} (total hand movement: {movement:.1f} ft)")
    
    # Print position summary for each player (pelvis position = field location)
    print(f"\n{'Player':<8} {'Pelvis X':>10} {'Pelvis Y':>10} {'Pelvis Z':>10} {'Visible':>8} {'Movement':>10}")
    print("-" * 60)
    
    for pidx in range(n_players):
        p = frames[0]['players'][pidx]
        pelvis = p['bonePositions'].get('0', [0, 0, 0])
        
        # Compute total movement across capture
        total_mvmt = 0
        for i in range(1, n_frames):
            p_curr = frames[i]['players'][pidx]['bonePositions'].get('0', [0,0,0])
            p_prev = frames[i-1]['players'][pidx]['bonePositions'].get('0', [0,0,0])
            total_mvmt += distance_3d(p_curr, p_prev)
        
        marker = " <-- active" if pidx == most_active else ""
        print(f"  #{pidx:<5} {pelvis[0]:>10.1f} {pelvis[1]:>10.1f} {pelvis[2]:>10.1f} {str(p['visible']):>8} {total_mvmt:>10.1f}{marker}")
    
    # Biomechanics snapshot for the most active player
    print(f"\n{'='*70}")
    print(f"BIOMECHANICS SNAPSHOT (Player #{most_active}, Frame 0)")
    print(f"{'='*70}")
    
    positions, timestamps = analyze_player_motion(frames, most_active)
    
    # Key joint angles
    angles = {
        'R Elbow': compute_joint_angles(positions, 0, 26, 27, 28),   # Shoulder-Elbow-Hand
        'L Elbow': compute_joint_angles(positions, 0, 65, 66, 67),
        'R Knee':  compute_joint_angles(positions, 0, 2, 3, 4),      # Hip-Knee-Foot
        'L Knee':  compute_joint_angles(positions, 0, 10, 11, 12),
        'R Shoulder': compute_joint_angles(positions, 0, 20, 26, 27), # Neck-Shoulder-Elbow
        'L Shoulder': compute_joint_angles(positions, 0, 20, 65, 66),
        'Torso': compute_joint_angles(positions, 0, 0, 18, 20),      # Pelvis-TorsoA-Neck
    }
    
    for name, angle in angles.items():
        if angle is not None:
            print(f"  {name:<15}: {angle:6.1f}°")
    
    # Peak hand speed
    hand_r_vels = compute_joint_velocities(positions, timestamps, 28)
    hand_l_vels = compute_joint_velocities(positions, timestamps, 67)
    
    if hand_r_vels:
        peak_r = max(hand_r_vels, key=lambda v: v['speed_mph'])
        print(f"\n  Peak R Hand speed: {peak_r['speed_mph']:.1f} mph (frame {peak_r['frame']})")
    if hand_l_vels:
        peak_l = max(hand_l_vels, key=lambda v: v['speed_mph'])
        print(f"  Peak L Hand speed: {peak_l['speed_mph']:.1f} mph (frame {peak_l['frame']})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python bone_analyzer.py <capture.json>              # Full analysis")
        print("  python bone_analyzer.py <capture.json> --csv out.csv  # Export to CSV")
        print("  python bone_analyzer.py <capture.json> --player 3     # Analyze player 3")
        return
    
    filepath = sys.argv[1]
    frames, bone_map = load_capture(filepath)
    
    # Parse optional args
    player_idx = None
    csv_path = None
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == '--player' and i + 1 < len(sys.argv):
            player_idx = int(sys.argv[i + 1])
        if arg == '--csv' and i + 1 < len(sys.argv):
            csv_path = sys.argv[i + 1]
    
    print_summary(frames, bone_map)
    
    if csv_path:
        pidx = player_idx if player_idx is not None else find_most_active_player(frames)[0]
        export_csv(frames, pidx, csv_path)
    
    if player_idx is not None:
        print(f"\n{'='*70}")
        print(f"DETAILED ANALYSIS: Player #{player_idx}")
        print(f"{'='*70}")
        positions, timestamps = analyze_player_motion(frames, player_idx)
        
        for bone_id, name in sorted(KEY_JOINTS.items()):
            vels = compute_joint_velocities(positions, timestamps, bone_id)
            if vels:
                peak = max(vels, key=lambda v: v['speed_mph'])
                avg = sum(v['speed_mph'] for v in vels) / len(vels)
                print(f"  {name:<15}: peak={peak['speed_mph']:6.1f} mph  avg={avg:5.1f} mph")


if __name__ == '__main__':
    main()
