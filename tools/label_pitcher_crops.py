#!/usr/bin/env python
"""Labeling UI for pitcher/not_pitcher person crops.

Shows each person crop alongside the full video frame with bbox context.
Labels are saved to pitcher_labels.json with resume support.

Display layout:
    Left:  Person crop (enlarged to ~400px height)
    Right: Full frame with this person's bbox in green, others in gray

Controls:
    P       - Label as pitcher
    N       - Label as not_pitcher
    S       - Skip
    B       - Go back
    Q/ESC   - Save & quit

Usage:
    python tools/label_pitcher_crops.py
    python tools/label_pitcher_crops.py --limit 500
    python tools/label_pitcher_crops.py --random
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DEFAULT_CROPS_DIR = PROJECT_ROOT / "data/labels/pitcher/crops"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data/labels/pitcher/crop_metadata.json"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data/labels/pitcher/pitcher_labels.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_labels(path: Path) -> dict:
    """Load existing labels for resume support."""
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("labels", {})
    return {}


def save_labels(path: Path, labels: dict):
    """Save labels to JSON with metadata summary."""
    pitcher_count = sum(1 for v in labels.values() if v == "pitcher")
    not_pitcher_count = sum(1 for v in labels.values() if v == "not_pitcher")

    data = {
        "metadata": {
            "total_labeled": len(labels),
            "pitcher_count": pitcher_count,
            "not_pitcher_count": not_pitcher_count,
            "last_saved": datetime.now().isoformat(),
        },
        "labels": labels,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(labels)} labels ({pitcher_count} pitcher, {not_pitcher_count} not_pitcher)")


def load_metadata(path: Path) -> dict:
    """Load crop metadata JSON."""
    if not path.exists():
        logger.error(f"Metadata file not found: {path}")
        logger.error("Run extract_pitcher_crops.py first!")
        return {}

    with open(path, "r") as f:
        data = json.load(f)
    return data.get("crops", {})


def get_frame_from_video(video_path: Path, frame_num: int) -> np.ndarray:
    """Read a single frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def build_display(
    crop_img: np.ndarray,
    full_frame: np.ndarray,
    bbox: list,
    all_bboxes: list,
    crop_name: str,
    stadium: str,
    labels: dict,
    idx: int,
    total: int,
) -> np.ndarray:
    """Build side-by-side display: enlarged crop (left) + annotated frame (right).

    Args:
        crop_img: The person crop image.
        full_frame: Full video frame.
        bbox: [x1, y1, x2, y2] of this person.
        all_bboxes: List of all person bboxes in this frame.
        crop_name: Crop filename.
        stadium: Stadium name.
        labels: Current label dict for counts.
        idx: Current crop index.
        total: Total crops to label.
    """
    target_h = 500

    # Left panel: enlarge crop to target height
    ch, cw = crop_img.shape[:2]
    scale = target_h / ch if ch > 0 else 1
    crop_display = cv2.resize(crop_img, (int(cw * scale), target_h), interpolation=cv2.INTER_LINEAR)

    # Right panel: full frame with bboxes
    fh, fw = full_frame.shape[:2]
    frame_scale = target_h / fh if fh > 0 else 1
    frame_display = cv2.resize(full_frame, (int(fw * frame_scale), target_h), interpolation=cv2.INTER_LINEAR)

    # Draw all bboxes on frame in gray
    for other_bbox in all_bboxes:
        ox1, oy1, ox2, oy2 = [int(v * frame_scale) for v in other_bbox]
        cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (128, 128, 128), 1)

    # Highlight current person's bbox in green
    bx1, by1, bx2, by2 = [int(v * frame_scale) for v in bbox]
    cv2.rectangle(frame_display, (bx1, by1), (bx2, by2), (0, 255, 0), 3)

    # Combine side by side with a divider
    divider = np.zeros((target_h, 3, 3), dtype=np.uint8)
    divider[:, :] = (100, 100, 100)

    display = np.hstack([crop_display, divider, frame_display])

    # Info overlay bar at top
    dh, dw = display.shape[:2]
    bar_h = 90
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (dw, bar_h), (0, 0, 0), -1)
    display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)

    # Counts
    pitcher_count = sum(1 for v in labels.values() if v == "pitcher")
    not_pitcher_count = sum(1 for v in labels.values() if v == "not_pitcher")

    # Text
    cv2.putText(display, f"Crop {idx + 1}/{total}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Pitcher: {pitcher_count}  Not: {not_pitcher_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(display, "[P]itch  [N]ot  [S]kip  [B]ack  [Q]uit", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Stadium + crop name on right
    cv2.putText(display, f"Stadium: {stadium.replace('_', ' ')}", (dw - 400, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, crop_name[:50], (dw - 400, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return display


def main():
    parser = argparse.ArgumentParser(description="Label pitcher crops")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max crops to label in this session")
    parser.add_argument("--random", action="store_true",
                        help="Randomize crop order")
    args = parser.parse_args()

    # Load existing labels and metadata
    labels = load_labels(args.output)
    crop_metadata = load_metadata(args.metadata)

    if not crop_metadata:
        return

    # Get unlabeled crops
    all_crops = sorted(crop_metadata.keys())
    unlabeled = [c for c in all_crops if c not in labels]

    logger.info(f"Total crops: {len(all_crops)}, already labeled: {len(labels)}, remaining: {len(unlabeled)}")

    if not unlabeled:
        logger.info("All crops already labeled!")
        return

    if args.random:
        import random
        random.shuffle(unlabeled)

    if args.limit:
        unlabeled = unlabeled[:args.limit]
        logger.info(f"Limiting to {args.limit} crops this session")

    # Group crops by (video, frame_num) for context display
    # Pre-compute: for each frame, collect all bboxes
    frame_bboxes = {}  # (video, frame_num) -> [bbox, ...]
    for cname, cinfo in crop_metadata.items():
        key = (cinfo["video"], cinfo["frame_num"])
        if key not in frame_bboxes:
            frame_bboxes[key] = []
        frame_bboxes[key].append(cinfo["bbox"])

    # Cache for video frames (avoid re-reading same frame)
    frame_cache = {}
    cache_max = 20

    # Labeling loop
    cv2.namedWindow("Pitcher Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pitcher Labeler", 1400, 600)

    history = []
    idx = 0

    logger.info("Starting labeling session...")
    logger.info("Controls: [P]itch  [N]ot  [S]kip  [B]ack  [Q]uit")

    while idx < len(unlabeled):
        crop_name = unlabeled[idx]
        info = crop_metadata.get(crop_name, {})
        stadium = info.get("stadium", "unknown")
        video_rel = info.get("video", "")
        frame_num = info.get("frame_num", 0)
        bbox = info.get("bbox", [0, 0, 0, 0])

        # Load crop image
        crop_path = args.crops_dir / crop_name
        crop_img = cv2.imread(str(crop_path))

        if crop_img is None:
            logger.warning(f"Could not load crop: {crop_path}")
            idx += 1
            continue

        # Load full frame (with caching)
        cache_key = (video_rel, frame_num)
        if cache_key in frame_cache:
            full_frame = frame_cache[cache_key]
        else:
            video_path = PROJECT_ROOT / video_rel
            full_frame = get_frame_from_video(video_path, frame_num)

            if full_frame is not None:
                # Manage cache size
                if len(frame_cache) >= cache_max:
                    frame_cache.pop(next(iter(frame_cache)))
                frame_cache[cache_key] = full_frame

        # Get all bboxes for this frame
        all_bboxes = frame_bboxes.get(cache_key, [bbox])

        if full_frame is None:
            # Fallback: show just the crop
            full_frame = crop_img.copy()
            all_bboxes = []

        # Build display
        display = build_display(
            crop_img, full_frame, bbox, all_bboxes,
            crop_name, stadium, labels, idx, len(unlabeled),
        )

        cv2.imshow("Pitcher Labeler", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('p'):
            labels[crop_name] = "pitcher"
            history.append((crop_name, idx))
            print(f"[{idx+1}/{len(unlabeled)}] PITCHER: {crop_name}")
            idx += 1

        elif key == ord('n'):
            labels[crop_name] = "not_pitcher"
            history.append((crop_name, idx))
            print(f"[{idx+1}/{len(unlabeled)}] NOT_PITCHER: {crop_name}")
            idx += 1

        elif key == ord('s'):
            print(f"[{idx+1}/{len(unlabeled)}] SKIP: {crop_name}")
            idx += 1

        elif key == ord('b'):
            if history:
                last_name, last_idx = history.pop()
                if last_name in labels:
                    del labels[last_name]
                idx = last_idx
                print(f"Going back to crop {idx + 1}")
            else:
                print("No history to go back to")

        elif key == ord('q') or key == 27:  # q or ESC
            print("\nQuitting...")
            break

    cv2.destroyAllWindows()

    # Save
    save_labels(args.output, labels)

    pitcher_count = sum(1 for v in labels.values() if v == "pitcher")
    not_pitcher_count = sum(1 for v in labels.values() if v == "not_pitcher")
    print(f"\nSession complete!")
    print(f"  Pitcher: {pitcher_count}")
    print(f"  Not pitcher: {not_pitcher_count}")
    print(f"  Total labeled: {len(labels)}")


if __name__ == "__main__":
    main()
