#!/usr/bin/env python
"""Test script for bat barrel detection model.

Usage:
    python tools/test_bat_barrel.py [--input PATH] [--output PATH]

Examples:
    # Test on a single image
    python tools/test_bat_barrel.py --input data/test_frame.jpg

    # Test on directory of images
    python tools/test_bat_barrel.py --input data/test_frames/ --output data/debug/barrel_test/

    # Test on video
    python tools/test_bat_barrel.py --input data/videos/sample.mp4 --output data/debug/barrel_video/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
MODEL_WEIGHTS = PROJECT_ROOT / "models/yolo_bat_barrel/train/weights/best.pt"

# Keypoint visualization
KEYPOINT_NAMES = ["cap", "middle", "beginning"]
KEYPOINT_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue
SKELETON = [[0, 1], [1, 2]]


def draw_predictions(img: np.ndarray, results) -> np.ndarray:
    """Draw keypoint predictions on image."""
    img_out = img.copy()

    for result in results:
        if result.keypoints is None:
            continue

        keypoints = result.keypoints.data.cpu().numpy()
        boxes = result.boxes

        for i, kps in enumerate(keypoints):
            # Get confidence from box if available
            conf = boxes.conf[i].item() if boxes is not None and len(boxes.conf) > i else 0

            # Draw skeleton first
            for j, k in SKELETON:
                if kps[j][2] > 0.5 and kps[k][2] > 0.5:  # visibility threshold
                    pt1 = (int(kps[j][0]), int(kps[j][1]))
                    pt2 = (int(kps[k][0]), int(kps[k][1]))
                    cv2.line(img_out, pt1, pt2, (255, 255, 0), 2)

            # Draw keypoints
            for idx, (x, y, v) in enumerate(kps):
                if v > 0.5:  # visibility threshold
                    pt = (int(x), int(y))
                    color = KEYPOINT_COLORS[idx]
                    cv2.circle(img_out, pt, 6, color, -1)
                    cv2.circle(img_out, pt, 8, (255, 255, 255), 2)

                    # Label
                    label = KEYPOINT_NAMES[idx]
                    cv2.putText(img_out, label, (pt[0] + 10, pt[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                    cv2.putText(img_out, label, (pt[0] + 10, pt[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw confidence
            if conf > 0:
                cv2.putText(img_out, f"Bat: {conf:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return img_out


def process_image(model, img_path: Path, output_dir: Path = None, show: bool = True):
    """Process a single image."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load: {img_path}")
        return

    # Run inference
    results = model(img, verbose=False)

    # Draw predictions
    img_out = draw_predictions(img, results)

    # Save if output dir specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img_out)
        print(f"Saved: {out_path}")

    # Show
    if show:
        cv2.imshow("Bat Barrel Detection", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_directory(model, input_dir: Path, output_dir: Path):
    """Process all images in a directory."""
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"Found {len(images)} images")

    output_dir.mkdir(parents=True, exist_ok=True)

    detections = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, verbose=False)

        # Check if detection found
        has_detection = any(r.keypoints is not None and len(r.keypoints.data) > 0
                          for r in results)
        if has_detection:
            detections += 1

        img_out = draw_predictions(img, results)
        cv2.imwrite(str(output_dir / img_path.name), img_out)

    print(f"\nProcessed {len(images)} images")
    print(f"Detections: {detections}/{len(images)} ({100*detections/len(images):.1f}%)")


def process_video(model, video_path: Path, output_dir: Path):
    """Process a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (video_path.stem + "_detected.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    detections = 0

    print(f"Processing {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        has_detection = any(r.keypoints is not None and len(r.keypoints.data) > 0
                          for r in results)
        if has_detection:
            detections += 1

        frame_out = draw_predictions(frame, results)
        out.write(frame_out)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    print(f"\nProcessed {frame_idx} frames")
    print(f"Detections: {detections}/{frame_idx} ({100*detections/frame_idx:.1f}%)")
    print(f"Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Test bat barrel detection model")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input image, directory, or video")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for results")
    parser.add_argument("--weights", type=Path, default=MODEL_WEIGHTS,
                        help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display results (batch mode)")
    args = parser.parse_args()

    # Check weights exist
    if not args.weights.exists():
        print(f"Model weights not found: {args.weights}")
        print("Please train the model first:")
        print("  python tools/train_yolo_bat_barrel.py")
        return

    # Load model
    from ultralytics import YOLO
    print(f"Loading model: {args.weights}")
    model = YOLO(str(args.weights))
    model.conf = args.conf

    # Process input
    if args.input.is_file():
        if args.input.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video
            output_dir = args.output or PROJECT_ROOT / "data/debug/barrel_video"
            process_video(model, args.input, output_dir)
        else:
            # Single image
            process_image(model, args.input, args.output, show=not args.no_show)

    elif args.input.is_dir():
        # Directory of images
        output_dir = args.output or PROJECT_ROOT / "data/debug/barrel_test"
        process_directory(model, args.input, output_dir)

    else:
        print(f"Input not found: {args.input}")


if __name__ == "__main__":
    main()
