#!/usr/bin/env python
"""Test YOLO-World for baseball detection with text prompts."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLOWorld

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
OUTPUT_DIR = PROJECT_ROOT / "data/debug/ball_yoloworld_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
            int(total_frames * 0.3),
            int(total_frames * 0.5),
            int(total_frames * 0.7),
        ]

        for idx in frame_indices[:frames_per_video]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((f"{video_path.stem}_f{idx}", frame))

        cap.release()

    return frames


def detect_ball_yoloworld(frame, model):
    """Detect baseball using YOLO-World."""
    results = model(frame, conf=0.05, verbose=False)

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
        cls_id = int(box.cls[0].cpu().numpy())

        # Calculate centroid
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "bbox": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            "centroid": (cx, cy),
            "confidence": conf,
            "class_id": cls_id,
        })

    return detections


def visualize_detection(frame, detections, output_path):
    """Draw detection visualization."""
    vis = frame.copy()

    for det in detections:
        x, y, w, h = det["bbox"]
        cx, cy = det["centroid"]
        conf = det["confidence"]

        # Color by confidence
        color = (0, 255, 0) if conf > 0.3 else (0, 165, 255) if conf > 0.1 else (0, 0, 255)
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        # Draw centroid
        cv2.circle(vis, (int(cx), int(cy)), 5, (255, 0, 255), -1)

        # Label
        label = f"baseball {conf:.0%}"
        cv2.putText(vis, label, (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Stats
    n_high = sum(1 for d in detections if d["confidence"] > 0.3)
    n_med = sum(1 for d in detections if 0.1 < d["confidence"] <= 0.3)
    cv2.putText(vis, f"Det: {len(detections)} (high: {n_high}, med: {n_med})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), vis)


def main():
    print("Loading YOLO-World model...")
    model = YOLOWorld("yolov8s-world.pt")

    # Set custom classes for baseball detection
    model.set_classes(["baseball", "ball"])
    print("Classes set to: baseball, ball")

    print("Extracting test frames...")
    frames = extract_test_frames(15, 3)
    print(f"Got {len(frames)} frames")

    print("\n=== Testing YOLO-World baseball detection ===")
    total_high_conf = 0
    total_med_conf = 0
    frames_with_detection = 0

    for frame_name, frame in frames:
        detections = detect_ball_yoloworld(frame, model)

        n_high = sum(1 for d in detections if d["confidence"] > 0.3)
        n_med = sum(1 for d in detections if 0.1 < d["confidence"] <= 0.3)

        total_high_conf += n_high
        total_med_conf += n_med
        if detections:
            frames_with_detection += 1

        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            print(f"{frame_name}: {len(detections)} det, best: {best['confidence']:.0%}")
        else:
            print(f"{frame_name}: No detections")

        output_path = OUTPUT_DIR / f"{frame_name}_yoloworld.jpg"
        visualize_detection(frame, detections, output_path)

    print(f"\n=== Summary ===")
    print(f"Total frames: {len(frames)}")
    print(f"Frames with detections: {frames_with_detection}")
    print(f"High confidence (>30%): {total_high_conf}")
    print(f"Medium confidence (10-30%): {total_med_conf}")
    print(f"Check outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
