#!/usr/bin/env python
"""Organize labeled player crops into ImageFolder structure for training.

Reads pitcher_labels.json, splits by video (not by crop) to prevent leakage,
and copies labeled crops into train/test ImageFolder directories.

Binary mode (default):
    data/labels/pitcher/frames/
        train/pitcher/       (~80% of videos' crops)
              not_pitcher/
        test/pitcher/        (~20% of videos' crops)
             not_pitcher/

Multiclass mode (--multiclass):
    data/labels/pitcher/frames/
        train/{pitcher,catcher,batter,other}/
        test/{pitcher,catcher,batter,other}/
    Skips legacy "not_pitcher" labels — only includes explicit 4-class labels.

Usage:
    python tools/prepare_pitcher_training.py
    python tools/prepare_pitcher_training.py --multiclass --clean
    python tools/prepare_pitcher_training.py --split 0.8 --seed 42
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DEFAULT_CROPS_DIR = PROJECT_ROOT / "data/labels/pitcher/crops"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data/labels/pitcher/pitcher_labels.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data/labels/pitcher/crop_metadata.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/labels/pitcher/frames"

BINARY_CLASSES = ["pitcher", "not_pitcher"]
MULTI_CLASSES = ["pitcher", "catcher", "batter", "other"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare player training data")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train fraction (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing output directory before copying")
    parser.add_argument("--multiclass", action="store_true",
                        help="4-class mode (pitcher/catcher/batter/other) instead of binary")
    args = parser.parse_args()

    classes = MULTI_CLASSES if args.multiclass else BINARY_CLASSES
    valid_labels = set(classes)
    mode_name = "multiclass (4-class)" if args.multiclass else "binary"

    logger.info(f"Mode: {mode_name} — classes: {classes}")

    # Load labels
    if not args.labels.exists():
        logger.error(f"Labels file not found: {args.labels}")
        logger.error("Run label_pitcher_crops.py first!")
        return

    with open(args.labels, "r") as f:
        label_data = json.load(f)
    labels = label_data.get("labels", {})
    logger.info(f"Loaded {len(labels)} labels")

    # Load metadata (to get video info for split-by-video)
    if not args.metadata.exists():
        logger.error(f"Metadata file not found: {args.metadata}")
        return

    with open(args.metadata, "r") as f:
        meta_data = json.load(f)
    crop_metadata = meta_data.get("crops", {})

    # Group labeled crops by source video (only include valid labels for this mode)
    video_to_crops = {}  # video_path -> [(crop_name, label), ...]
    skipped_no_meta = 0
    skipped_wrong_label = 0

    for crop_name, label in labels.items():
        if label not in valid_labels:
            skipped_wrong_label += 1
            continue

        info = crop_metadata.get(crop_name)
        if info is None:
            logger.warning(f"No metadata for {crop_name}, skipping")
            skipped_no_meta += 1
            continue

        video = info["video"]
        if video not in video_to_crops:
            video_to_crops[video] = []
        video_to_crops[video].append((crop_name, label))

    included = sum(len(v) for v in video_to_crops.values())
    logger.info(f"Included {included} crops from {len(video_to_crops)} videos")
    if skipped_wrong_label:
        logger.info(f"  Skipped {skipped_wrong_label} crops with labels not in {classes}")
    if skipped_no_meta:
        logger.info(f"  Skipped {skipped_no_meta} crops with no metadata")

    # Split by video
    rng = random.Random(args.seed)
    video_list = sorted(video_to_crops.keys())
    rng.shuffle(video_list)

    n_train = int(len(video_list) * args.split)
    train_videos = set(video_list[:n_train])
    test_videos = set(video_list[n_train:])

    logger.info(f"Split: {len(train_videos)} train videos, {len(test_videos)} test videos")

    # Clean output if requested
    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
        logger.info(f"Cleaned {args.output_dir}")

    # Create output dirs
    for split in ["train", "test"]:
        for cls in classes:
            (args.output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Copy crops
    counts = {split: {cls: 0 for cls in classes} for split in ["train", "test"]}

    for video, crops in video_to_crops.items():
        split = "train" if video in train_videos else "test"

        for crop_name, label in crops:
            src = args.crops_dir / crop_name
            dst = args.output_dir / split / label / crop_name

            if not src.exists():
                logger.warning(f"Crop not found: {src}")
                continue

            shutil.copy2(str(src), str(dst))
            counts[split][label] += 1

    # Summary
    logger.info(f"\nImageFolder structure created at: {args.output_dir}")
    for split in ["train", "test"]:
        parts = [f"{cls}={counts[split][cls]}" for cls in classes]
        logger.info(f"  {split.capitalize()}: {', '.join(parts)}")
    logger.info(f"  Total: {sum(sum(v.values()) for v in counts.values())}")

    # Check for class imbalance
    train_total = sum(counts['train'].values())
    if train_total > 0:
        for cls in classes:
            pct = counts['train'][cls] / train_total * 100
            if pct < 5:
                logger.warning(f"  Low representation: {cls} is only {pct:.1f}% of train set")


if __name__ == "__main__":
    main()
