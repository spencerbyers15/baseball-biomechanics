#!/usr/bin/env python
"""
Train YOLOv8 for baseball detection.

Run this after labeling with baseball_labeler.py.

Usage:
    python tools/train_yolo_baseball.py [options]
    python tools/train_yolo_baseball.py --test  # Test trained model
"""

import argparse
import shutil
import random
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DATA_DIR = PROJECT_ROOT / "data/labels/baseball/yolo"
OUTPUT_DIR = PROJECT_ROOT / "models/yolo_baseball"


def create_train_val_split(yolo_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    """Create train/val split from labeled images."""
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"

    # Get all images with corresponding labels
    all_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    # Filter out images already in train/val subdirs
    labeled_images = [
        img for img in all_images
        if (labels_dir / (img.stem + ".txt")).exists()
        and "train" not in str(img) and "val" not in str(img)
    ]

    if not labeled_images:
        raise ValueError(f"No labeled images found in {images_dir}")

    print(f"Found {len(labeled_images)} labeled images")

    # Count positives vs negatives
    positive_count = 0
    negative_count = 0
    for img_path in labeled_images:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.stat().st_size > 0:
            positive_count += 1
        else:
            negative_count += 1

    print(f"  Positives (with ball): {positive_count}")
    print(f"  Negatives (no ball): {negative_count}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(labeled_images)

    val_count = int(len(labeled_images) * val_ratio)
    val_images = labeled_images[:val_count]
    train_images = labeled_images[val_count:]

    print(f"Split: {len(train_images)} train, {len(val_images)} val")

    # Create split directories
    train_img_dir = images_dir / "train"
    val_img_dir = images_dir / "val"
    train_lbl_dir = labels_dir / "train"
    val_lbl_dir = labels_dir / "val"

    # Clean and create directories
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Copy files to split directories
    for img_path in train_images:
        shutil.copy(img_path, train_img_dir / img_path.name)
        label_path = labels_dir / (img_path.stem + ".txt")
        shutil.copy(label_path, train_lbl_dir / label_path.name)

    for img_path in val_images:
        shutil.copy(img_path, val_img_dir / img_path.name)
        label_path = labels_dir / (img_path.stem + ".txt")
        shutil.copy(label_path, val_lbl_dir / label_path.name)

    print(f"Created train/val split")
    return len(train_images), len(val_images)


def update_data_yaml(yolo_dir: Path):
    """Update data.yaml with train/val split paths."""
    yaml_path = yolo_dir / "data.yaml"

    content = f"""# YOLOv8 Detection Dataset Configuration for Baseball
# Auto-generated: {datetime.now().isoformat()}

path: {yolo_dir}
train: images/train
val: images/val

names:
  0: baseball
"""

    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"Updated {yaml_path}")
    return yaml_path


def train(epochs: int = 100, batch: int = 8, imgsz: int = 640,
          model_size: str = "n", skip_split: bool = False):
    """Train YOLOv8 for baseball detection."""
    from ultralytics import YOLO

    data_yaml = DATA_DIR / "data.yaml"

    # Create train/val split
    if not skip_split:
        print("\n[1/3] Creating train/val split...")
        create_train_val_split(DATA_DIR)
        print("\n[2/3] Updating data.yaml...")
        update_data_yaml(DATA_DIR)
    else:
        print("Skipping split (using existing)")

    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found.")
        print("Run baseball_labeler.py first, label some frames, and save.")
        return

    # Load YOLOv8 model
    base_model = f"yolov8{model_size}.pt"
    print(f"\n[3/3] Training with {base_model}...")
    model = YOLO(base_model)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        verbose=True,
        # Augmentation
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        degrees=10.0,
        scale=0.3,
    )

    print(f"\nTraining complete!")
    print(f"Best model saved to: {OUTPUT_DIR}/train/weights/best.pt")


def test_model():
    """Test the trained model on sample frames."""
    import cv2
    from ultralytics import YOLO

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
    parser = argparse.ArgumentParser(description="Train YOLOv8 baseball detector")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model-size", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"], help="Model size")
    parser.add_argument("--skip-split", action="store_true", help="Skip train/val split")
    args = parser.parse_args()

    if args.test:
        test_model()
    else:
        train(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            model_size=args.model_size,
            skip_split=args.skip_split
        )
