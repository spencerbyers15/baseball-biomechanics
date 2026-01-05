#!/usr/bin/env python
"""
Train YOLOv8 for baseball detection.

Run this after labeling with baseball_labeler.py.
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DATA_DIR = PROJECT_ROOT / "data/labels/baseball/yolo"
OUTPUT_DIR = PROJECT_ROOT / "models/yolo_baseball"


def train():
    """Train YOLOv8 for baseball detection."""

    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found.")
        print("Run baseball_labeler.py first, label some frames, and save.")
        return

    # Count images
    images = list((DATA_DIR / "images").glob("*.png"))
    print(f"Found {len(images)} labeled images")

    if len(images) < 50:
        print("Warning: Less than 50 images. Consider labeling more for better results.")

    # Load YOLOv8 nano (fast, good for small objects)
    model = YOLO("yolov8n.pt")

    # Train
    print("\nStarting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,  # Early stopping
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

    model_path = OUTPUT_DIR / "train/weights/best.pt"
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        return

    model = YOLO(str(model_path))

    # Test on a few frames
    test_dir = PROJECT_ROOT / "data/debug/baseball_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = PROJECT_ROOT / "data/videos/2024"
    videos = list(videos_dir.rglob("*.mp4"))[:5]

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        results = model(frame, verbose=False)[0]
        annotated = results.plot()

        output_path = test_dir / f"{video_path.stem}_baseball.jpg"
        cv2.imwrite(str(output_path), annotated)
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
