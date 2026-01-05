#!/usr/bin/env python
"""
Train YOLOv8-Pose for bat keypoint detection.

Run this after labeling with bat_keypoint_labeler.py.
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DATA_DIR = PROJECT_ROOT / "data/labels/bat_keypoints/yolo_pose"
OUTPUT_DIR = PROJECT_ROOT / "models/yolo_bat_pose"


def train():
    """Train YOLOv8-Pose for bat keypoint detection."""

    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found.")
        print("Run bat_keypoint_labeler.py first, label some frames, and save.")
        return

    # Count images
    images = list((DATA_DIR / "images").glob("*.png"))
    print(f"Found {len(images)} labeled images")

    if len(images) < 50:
        print("Warning: Less than 50 images. Consider labeling more for better results.")

    # Load YOLOv8 pose model (nano version)
    model = YOLO("yolov8n-pose.pt")

    # Train
    print("\nStarting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=150,
        imgsz=640,
        batch=16,
        patience=30,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        verbose=False,
    )

    print(f"\nTraining complete!")
    print(f"Best model saved to: {OUTPUT_DIR}/train/weights/best.pt")


def test_model():
    """Test the trained model on sample frames."""
    import cv2
    import numpy as np

    model_path = OUTPUT_DIR / "train/weights/best.pt"
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        return

    model = YOLO(str(model_path))

    # Test on a few frames
    test_dir = PROJECT_ROOT / "data/debug/bat_pose_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = PROJECT_ROOT / "data/videos/2024"
    videos = list(videos_dir.rglob("*.mp4"))[:5]

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        # Get early frame (bat in stance)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.2))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        results = model(frame, verbose=False)[0]

        # Custom visualization for bat keypoints
        vis = frame.copy()

        if results.keypoints is not None and len(results.keypoints) > 0:
            for kpts in results.keypoints.data:
                kpts = kpts.cpu().numpy()
                if len(kpts) >= 2:
                    knob = kpts[0][:2].astype(int)
                    cap_pt = kpts[1][:2].astype(int)

                    # Draw bat line
                    cv2.line(vis, tuple(knob), tuple(cap_pt), (0, 255, 0), 3)

                    # Draw keypoints
                    cv2.circle(vis, tuple(knob), 8, (255, 100, 0), -1)  # Knob - blue
                    cv2.circle(vis, tuple(cap_pt), 8, (0, 100, 255), -1)  # Cap - red
                    cv2.putText(vis, "Knob", (knob[0]+10, knob[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
                    cv2.putText(vis, "Cap", (cap_pt[0]+10, cap_pt[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

        output_path = test_dir / f"{video_path.stem}_bat_pose.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"Saved: {output_path.name}")

    print(f"\nTest outputs in {test_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test trained model")
    args = parser.parse_args()

    if args.test:
        test_model()
    else:
        train()
