#!/usr/bin/env python
"""Training script for YOLOv8-pose bat barrel detection model.

This script:
1. Creates train/val split from labeled data
2. Sets up the dataset in YOLO format
3. Trains YOLOv8-pose with custom 3-keypoint skeleton
4. Saves model and training artifacts

Usage:
    python tools/train_yolo_bat_barrel.py [options]

Examples:
    # Quick test run
    python tools/train_yolo_bat_barrel.py --epochs 10 --batch 4

    # Full training
    python tools/train_yolo_bat_barrel.py --epochs 100 --batch 8 --imgsz 640
"""

import argparse
import shutil
import random
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
LABELS_DIR = PROJECT_ROOT / "data/labels/bat_barrel"
MODEL_DIR = PROJECT_ROOT / "models/yolo_bat_barrel"


def create_train_val_split(labels_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    """Create train/val split from labeled images."""
    images_dir = labels_dir / "images"
    labels_subdir = labels_dir / "labels"

    # Get all labeled images (those with corresponding .txt files)
    all_images = list(images_dir.glob("*.jpg"))
    labeled_images = []

    for img_path in all_images:
        label_path = labels_subdir / (img_path.stem + ".txt")
        if label_path.exists():
            labeled_images.append(img_path)

    if not labeled_images:
        raise ValueError(f"No labeled images found in {images_dir}")

    print(f"Found {len(labeled_images)} labeled images")

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
    train_lbl_dir = labels_subdir / "train"
    val_lbl_dir = labels_subdir / "val"

    # Clean and create directories
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Copy files to split directories
    for img_path in train_images:
        shutil.copy(img_path, train_img_dir / img_path.name)
        label_path = labels_subdir / (img_path.stem + ".txt")
        shutil.copy(label_path, train_lbl_dir / label_path.name)

    for img_path in val_images:
        shutil.copy(img_path, val_img_dir / img_path.name)
        label_path = labels_subdir / (img_path.stem + ".txt")
        shutil.copy(label_path, val_lbl_dir / label_path.name)

    print(f"Created train/val split in {labels_dir}")
    return len(train_images), len(val_images)


def update_data_yaml(labels_dir: Path, model_dir: Path):
    """Update data.yaml with correct paths."""
    yaml_path = model_dir / "data.yaml"

    content = f"""# YOLOv8 Pose Dataset Configuration for Bat Barrel Detection
# 3-keypoint skeleton: barrel_cap -> barrel_middle -> barrel_beginning
# Auto-generated: {datetime.now().isoformat()}

# Dataset paths
path: {labels_dir}
train: images/train
val: images/val

# Class names
names:
  0: bat

# Keypoint configuration
kpt_shape: [3, 3]  # [num_keypoints, dimensions] - 3 keypoints with (x, y, visibility)

# Keypoint names (for reference)
# 0: barrel_cap - end of bat
# 1: barrel_middle - middle of barrel
# 2: barrel_beginning - where barrel meets handle

# Skeleton connections for visualization
skeleton:
  - [0, 1]  # cap to middle
  - [1, 2]  # middle to beginning
"""

    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"Updated {yaml_path}")
    return yaml_path


def train_model(
    data_yaml: Path,
    model_dir: Path,
    epochs: int = 100,
    batch: int = 8,
    imgsz: int = 640,
    model_size: str = "n",
    device: str = "0",
    resume: bool = False,
    patience: int = 20,
):
    """Train YOLOv8-pose model."""
    from ultralytics import YOLO

    # Use pretrained pose model as base
    base_model = f"yolov8{model_size}-pose.pt"
    print(f"\nLoading base model: {base_model}")

    model = YOLO(base_model)

    # Training arguments
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "project": str(model_dir),
        "name": "train",
        "exist_ok": True,
        "patience": patience,
        "save": True,
        "plots": True,
        "verbose": True,
        # Pose-specific
        "pose": 12.0,  # pose loss gain
        "kobj": 2.0,   # keypoint obj loss gain
        # Data augmentation
        "flipud": 0.0,  # no vertical flip (bat orientation matters)
        "fliplr": 0.5,  # horizontal flip ok
        "mosaic": 0.5,  # reduced mosaic (preserve bat context)
        "mixup": 0.1,
        "degrees": 15.0,  # rotation augmentation
        "scale": 0.3,
        "translate": 0.1,
    }

    if resume:
        last_weights = model_dir / "train/weights/last.pt"
        if last_weights.exists():
            print(f"Resuming from {last_weights}")
            model = YOLO(str(last_weights))
            train_args["resume"] = True

    print(f"\nStarting training with {epochs} epochs, batch {batch}, imgsz {imgsz}")
    print(f"Output: {model_dir}/train/")
    print("-" * 60)

    results = model.train(**train_args)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best weights: {model_dir}/train/weights/best.pt")
    print(f"Results: {model_dir}/train/")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-pose bat barrel model")

    # Data arguments
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR,
                        help="Directory with labeled data")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="Output directory for model")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--model-size", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLO model size (n=nano, s=small, etc.)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (0, 1, cpu)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")

    # Utility arguments
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip train/val split (use existing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Setup only, don't train")

    args = parser.parse_args()

    print("="*60)
    print("BAT BARREL YOLO POSE TRAINING")
    print("="*60)
    print(f"Labels dir: {args.labels_dir}")
    print(f"Model dir: {args.model_dir}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    print("="*60)

    # Check for labeled data
    images_dir = args.labels_dir / "images"
    if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
        print(f"\nERROR: No images found in {images_dir}")
        print("Please run the labeler first:")
        print("  python tools/barrel_keypoint_labeler.py")
        return

    # Create train/val split
    if not args.skip_split:
        print("\n[1/3] Creating train/val split...")
        train_count, val_count = create_train_val_split(
            args.labels_dir,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
    else:
        print("\n[1/3] Skipping split (using existing)")
        train_count = len(list((images_dir / "train").glob("*.jpg")))
        val_count = len(list((images_dir / "val").glob("*.jpg")))

    # Update data.yaml
    print("\n[2/3] Updating data.yaml...")
    data_yaml = update_data_yaml(args.labels_dir, args.model_dir)

    if args.dry_run:
        print("\n[3/3] Dry run - skipping training")
        print(f"\nTo train, run:")
        print(f"  python tools/train_yolo_bat_barrel.py --epochs {args.epochs}")
        return

    # Train
    print("\n[3/3] Training model...")
    train_model(
        data_yaml=data_yaml,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        model_size=args.model_size,
        device=args.device,
        resume=args.resume,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
