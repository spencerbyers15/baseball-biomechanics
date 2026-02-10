#!/usr/bin/env python
"""
Extract frames from labeled video segments into train/test directories.

Reads scene_cut_labels.json, samples frames from each segment, and saves them
into an ImageFolder structure for torchvision training:

    data/labels/scene_cuts/frames/
        train/
            main_angle/
            other/
        test/
            main_angle/
            other/

Splits by VIDEO (not by frame) to prevent data leakage — frames from the
same video never appear in both train and test.

Usage:
    python tools/extract_segment_frames.py
    python tools/extract_segment_frames.py --sample-rate 15 --test-ratio 0.2
"""

import cv2
import json
import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
LABELS_PATH = PROJECT_ROOT / "data/labels/scene_cuts/scene_cut_labels.json"
OUTPUT_DIR = PROJECT_ROOT / "data/labels/scene_cuts/frames"


def main():
    parser = argparse.ArgumentParser(description="Extract segment frames for training")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=20,
                        help="Extract 1 frame every N frames (default: 20, ~3 per second at 60fps)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of videos for test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.labels) as f:
        data = json.load(f)

    # Collect labeled videos
    labeled = {k: v for k, v in data["videos"].items()
               if v.get("status") == "labeled"}
    print(f"Labeled videos: {len(labeled)}")

    # Split by video — shuffle and take test_ratio for test
    video_paths = list(labeled.keys())
    random.shuffle(video_paths)
    n_test = max(1, int(len(video_paths) * args.test_ratio))
    test_videos = set(video_paths[:n_test])
    train_videos = set(video_paths[n_test:])
    print(f"Train videos: {len(train_videos)}, Test videos: {len(test_videos)}")

    # Create output directories
    for split in ["train", "test"]:
        for cls in ["main_angle", "other"]:
            (args.output / split / cls).mkdir(parents=True, exist_ok=True)

    # Extract frames
    counts = defaultdict(int)
    total_videos = len(labeled)

    for vi, (vpath, vdata) in enumerate(labeled.items()):
        split = "test" if vpath in test_videos else "train"
        full_path = str(PROJECT_ROOT / vpath)

        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print(f"  [{vi+1}/{total_videos}] SKIP (cannot open): {vpath}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cuts = vdata.get("cut_frames", [])
        seg_labels = vdata.get("segment_labels", [])
        boundaries = [0] + cuts + [total_frames]
        segments = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

        stadium = vdata.get("stadium", "unknown")
        season = vdata.get("season", "unknown")
        video_name = Path(vpath).stem

        n_extracted = 0
        for si, (seg_start, seg_end) in enumerate(segments):
            if si >= len(seg_labels) or seg_labels[si] is None:
                continue

            label = seg_labels[si]
            seg_len = seg_end - seg_start

            # Sample frames
            if seg_len <= args.sample_rate:
                # Short segment — take middle frame
                frame_indices = [seg_start + seg_len // 2]
            else:
                frame_indices = list(range(seg_start, seg_end, args.sample_rate))

            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue

                fname = f"{stadium}_{season}_{video_name}_f{fi:05d}.jpg"
                out_path = args.output / split / label / fname
                cv2.imwrite(str(out_path), frame)
                counts[f"{split}/{label}"] += 1
                n_extracted += 1

        cap.release()
        print(f"  [{vi+1}/{total_videos}] {split:5s} | {stadium}/{season} | "
              f"{n_extracted} frames | {len(segments)} segments")

    # Summary
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    for key in sorted(counts.keys()):
        print(f"  {key}: {counts[key]}")
    print(f"  TOTAL: {sum(counts.values())}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
