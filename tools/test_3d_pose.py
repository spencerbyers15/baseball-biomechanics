#!/usr/bin/env python
"""Test current MediaPipe 3D pose estimation quality."""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.insert(0, str(Path("F:/Claude_Projects/baseball-biomechanics")))

from src.pose.mediapipe_backend import MediaPipeBackend
from src.segmentation.sam3_tracker import SAM3Tracker

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
OUTPUT_DIR = PROJECT_ROOT / "data/debug/3d_pose_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_sample_frames(n_videos=5):
    """Get sample frames with pitcher crops."""
    videos_dir = PROJECT_ROOT / "data/videos/2024"
    all_videos = list(videos_dir.rglob("*.mp4"))

    import random
    sample_videos = random.sample(all_videos, min(n_videos, len(all_videos)))

    frames = []
    for video_path in sample_videos:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frame from middle (pitch release area)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.5))
        ret, frame = cap.read()
        cap.release()

        if ret:
            frames.append((video_path.stem, frame))

    return frames


def plot_3d_skeleton(keypoints, output_path, title="3D Pose"):
    """Plot 3D skeleton from keypoints."""
    # Extract coordinates
    names = [kp.name for kp in keypoints]
    x = [kp.x for kp in keypoints]
    y = [kp.y for kp in keypoints]
    z = [kp.z if kp.z is not None else 0 for kp in keypoints]

    # Key body landmarks indices
    key_joints = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]

    # Get indices
    indices = {name: i for i, name in enumerate(names)}

    # Create figure
    fig = plt.figure(figsize=(12, 5))

    # 2D view (x, y)
    ax1 = fig.add_subplot(121)
    for joint in key_joints:
        if joint in indices:
            idx = indices[joint]
            color = 'b' if 'left' in joint else 'r'
            ax1.scatter(x[idx], y[idx], c=color, s=50)
            ax1.annotate(joint.replace('left_', 'L_').replace('right_', 'R_'),
                        (x[idx], y[idx]), fontsize=6)

    # Draw connections
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ]

    for start, end in connections:
        if start in indices and end in indices:
            ax1.plot([x[indices[start]], x[indices[end]]],
                    [y[indices[start]], y[indices[end]]], 'g-', alpha=0.5)

    ax1.set_title("2D View (x, y)")
    ax1.invert_yaxis()  # Image coords
    ax1.set_aspect('equal')

    # 3D view
    ax2 = fig.add_subplot(122, projection='3d')
    for joint in key_joints:
        if joint in indices:
            idx = indices[joint]
            color = 'b' if 'left' in joint else 'r'
            ax2.scatter(x[idx], z[idx], -y[idx], c=color, s=50)

    for start, end in connections:
        if start in indices and end in indices:
            ax2.plot([x[indices[start]], x[indices[end]]],
                    [z[indices[start]], z[indices[end]]],
                    [-y[indices[start]], -y[indices[end]]], 'g-', alpha=0.5)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z (depth)')
    ax2.set_zlabel('-Y')
    ax2.set_title("3D View (x, z, y)")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def analyze_z_values(keypoints):
    """Analyze z-value distribution."""
    z_vals = [kp.z for kp in keypoints if kp.z is not None]
    if not z_vals:
        return None

    return {
        "min": min(z_vals),
        "max": max(z_vals),
        "range": max(z_vals) - min(z_vals),
        "mean": np.mean(z_vals),
        "std": np.std(z_vals),
    }


def main():
    print("Initializing backends...")
    pose_backend = MediaPipeBackend(model_complexity=2)  # Heavy model
    segmenter = SAM3Tracker()

    print("Getting sample frames...")
    frames = get_sample_frames(5)
    print(f"Got {len(frames)} frames")

    print("\n=== Testing MediaPipe 3D pose ===")
    for video_name, frame in frames:
        print(f"\nProcessing: {video_name}")

        # Segment to get pitcher crop
        seg_results = segmenter.segment_all_players(frame, 0)

        if seg_results.get("pitcher") and seg_results["pitcher"].cropped_frame is not None:
            crop = seg_results["pitcher"].cropped_frame
            print(f"  Pitcher crop: {crop.shape}")
        else:
            # Use center crop as fallback
            h, w = frame.shape[:2]
            crop = frame[int(h*0.3):int(h*0.9), int(w*0.3):int(w*0.7)]
            print(f"  Using center crop: {crop.shape}")

        # Run pose estimation
        pose_result = pose_backend.process_frame(crop, 0)

        if pose_result.is_valid:
            z_stats = analyze_z_values(pose_result.keypoints)
            if z_stats:
                print(f"  Pose detected! Z-values: range={z_stats['range']:.3f}, std={z_stats['std']:.3f}")

                # Create visualization
                output_path = OUTPUT_DIR / f"{video_name}_3d_pose.png"
                plot_3d_skeleton(pose_result.keypoints, output_path, title=video_name)

                # Also save annotated frame
                annotated = pose_backend.draw_pose(crop, pose_result)
                cv2.imwrite(str(OUTPUT_DIR / f"{video_name}_2d_overlay.jpg"), annotated)
            else:
                print("  No z-values available")
        else:
            print("  No pose detected")

    print(f"\n=== Done! Check outputs in {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()
