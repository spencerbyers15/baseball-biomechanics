#!/usr/bin/env python
"""Test pre-trained YOLO for baseball detection using COCO sports_ball class."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
OUTPUT_DIR = PROJECT_ROOT / "data/debug/ball_detection_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# COCO class 32 = sports ball
SPORTS_BALL_CLASS = 32


def extract_test_frames(n_videos=15, frames_per_video=3):
    """Extract frames from videos, focusing on pitch-release area."""
    videos_dir = PROJECT_ROOT / "data/videos/2024"
    all_videos = list(videos_dir.rglob("*.mp4"))

    import random
    sample_videos = random.sample(all_videos, min(n_videos, len(all_videos)))

    frames = []
    for video_path in sample_videos:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 50:
            cap.release()
            continue

        # Sample frames from middle of video (where pitch happens)
        frame_indices = [
            int(total_frames * 0.3),  # Early
            int(total_frames * 0.5),  # Middle (likely ball in flight)
            int(total_frames * 0.7),  # Late
        ]

        for idx in frame_indices[:frames_per_video]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((f"{video_path.stem}_f{idx}", frame))

        cap.release()

    return frames


def detect_ball_yolo(frame, model):
    """Detect sports ball using YOLO."""
    # Run detection for sports ball class only
    results = model(frame, classes=[SPORTS_BALL_CLASS], conf=0.1, verbose=False)

    if not results or len(results) == 0:
        return []

    result = results[0]
    if not hasattr(result, "boxes") or result.boxes is None:
        return []

    detections = []
    boxes = result.boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())

        # Calculate centroid
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "bbox": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            "centroid": (cx, cy),
            "confidence": conf,
        })

    return detections


def visualize_detection(frame, detections, output_path):
    """Draw detection visualization."""
    vis = frame.copy()

    for det in detections:
        x, y, w, h = det["bbox"]
        cx, cy = det["centroid"]
        conf = det["confidence"]

        # Draw bbox
        color = (0, 255, 0) if conf > 0.3 else (0, 165, 255)
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        # Draw centroid
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        # Label
        label = f"Ball {conf:.0%}"
        cv2.putText(vis, label, (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Stats
    n_high = sum(1 for d in detections if d["confidence"] > 0.3)
    cv2.putText(vis, f"Detections: {len(detections)} (high conf: {n_high})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), vis)


def main():
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    print("Extracting test frames...")
    frames = extract_test_frames(15, 3)
    print(f"Got {len(frames)} frames")

    print("\n=== Testing ball detection (COCO sports_ball class) ===")
    total_detections = 0
    high_conf_count = 0

    for frame_name, frame in frames:
        detections = detect_ball_yolo(frame, model)

        n_high = sum(1 for d in detections if d["confidence"] > 0.3)
        total_detections += len(detections)
        high_conf_count += n_high

        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            print(f"{frame_name}: {len(detections)} det, best conf: {best['confidence']:.0%}")
        else:
            print(f"{frame_name}: No detections")

        output_path = OUTPUT_DIR / f"{frame_name}_ball.jpg"
        visualize_detection(frame, detections, output_path)

    print(f"\n=== Summary ===")
    print(f"Total frames: {len(frames)}")
    print(f"Frames with detections: {sum(1 for n, f in frames for d in [detect_ball_yolo(f, model)] if d)}")
    print(f"High confidence (>30%): {high_conf_count}")
    print(f"Check outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
