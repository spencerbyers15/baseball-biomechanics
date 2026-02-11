#!/usr/bin/env python
"""Labeling UI for player role classification (pitcher/catcher/batter/other).

Shows each person crop alongside the full video frame with bbox context.
Labels are saved to pitcher_labels.json with resume support.

Display layout:
    Left:  Person crop (enlarged to ~400px height)
    Right: Full frame with this person's bbox in green, others in gray

Controls:
    P       - Label as pitcher
    C       - Label as catcher
    R       - Label as batter
    O       - Label as other
    S       - Skip
    B       - Go back
    Q/ESC   - Save & quit

Usage:
    python tools/label_pitcher_crops.py
    python tools/label_pitcher_crops.py --filter not_pitcher
    python tools/label_pitcher_crops.py --filter all --limit 500
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

VALID_LABELS = {"pitcher", "catcher", "batter", "other", "not_pitcher"}

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
    catcher_count = sum(1 for v in labels.values() if v == "catcher")
    batter_count = sum(1 for v in labels.values() if v == "batter")
    other_count = sum(1 for v in labels.values() if v == "other")
    not_pitcher_count = sum(1 for v in labels.values() if v == "not_pitcher")

    data = {
        "metadata": {
            "total_labeled": len(labels),
            "pitcher_count": pitcher_count,
            "catcher_count": catcher_count,
            "batter_count": batter_count,
            "other_count": other_count,
            "not_pitcher_count": not_pitcher_count,
            "last_saved": datetime.now().isoformat(),
        },
        "labels": labels,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        f"Saved {len(labels)} labels "
        f"(P={pitcher_count} C={catcher_count} B={batter_count} O={other_count}"
        f"{f' legacy_not_pitcher={not_pitcher_count}' if not_pitcher_count else ''})"
    )


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
    current_label: str = None,
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
        current_label: Existing label for this crop (if relabeling).
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
    catcher_count = sum(1 for v in labels.values() if v == "catcher")
    batter_count = sum(1 for v in labels.values() if v == "batter")
    other_count = sum(1 for v in labels.values() if v == "other")

    # Text
    cv2.putText(display, f"Crop {idx + 1}/{total}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Pitcher: {pitcher_count}  Catcher: {catcher_count}  Batter: {batter_count}  Other: {other_count}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(display, "[P]itch  [C]atch  Batte[R]  [O]ther  [S]kip  [B]ack  [Q]uit", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Stadium + crop name on right
    cv2.putText(display, f"Stadium: {stadium.replace('_', ' ')}", (dw - 400, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    crop_label_text = crop_name[:50]
    if current_label:
        crop_label_text += f"  (was: {current_label})"
    cv2.putText(display, crop_label_text, (dw - 400, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return display


def main():
    parser = argparse.ArgumentParser(description="Label player crops (pitcher/catcher/batter/other)")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max crops to label in this session")
    parser.add_argument("--random", action="store_true",
                        help="Randomize crop order")
    parser.add_argument("--filter", choices=["unlabeled", "not_pitcher", "all"],
                        default="unlabeled",
                        help="Which crops to show: unlabeled (default), not_pitcher (relabel), all")
    args = parser.parse_args()

    # Load existing labels and metadata
    labels = load_labels(args.output)
    crop_metadata = load_metadata(args.metadata)

    if not crop_metadata:
        return

    # Filter crops based on --filter flag
    all_crops = sorted(crop_metadata.keys())

    if args.filter == "unlabeled":
        work_list = [c for c in all_crops if c not in labels]
    elif args.filter == "not_pitcher":
        work_list = [c for c in all_crops if labels.get(c) == "not_pitcher"]
    elif args.filter == "all":
        work_list = list(all_crops)

    # Count current state
    n_pitcher = sum(1 for v in labels.values() if v == "pitcher")
    n_catcher = sum(1 for v in labels.values() if v == "catcher")
    n_batter = sum(1 for v in labels.values() if v == "batter")
    n_other = sum(1 for v in labels.values() if v == "other")
    n_not_pitcher = sum(1 for v in labels.values() if v == "not_pitcher")

    logger.info(f"Total crops: {len(all_crops)}, labeled: {len(labels)}")
    logger.info(f"  Pitcher={n_pitcher} Catcher={n_catcher} Batter={n_batter} Other={n_other} not_pitcher(legacy)={n_not_pitcher}")
    logger.info(f"Filter: --filter {args.filter} -> {len(work_list)} crops to show")

    if not work_list:
        logger.info("No crops match the filter!")
        return

    if args.random:
        import random
        random.shuffle(work_list)

    if args.limit:
        work_list = work_list[:args.limit]
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
    cv2.namedWindow("Player Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Player Labeler", 1400, 600)

    history = []
    idx = 0

    logger.info("Starting labeling session...")
    logger.info("Controls: [P]itch  [C]atch  Batte[R]  [O]ther  [S]kip  [B]ack  [Q]uit")

    while idx < len(work_list):
        crop_name = work_list[idx]
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

        # Current label (for relabeling display)
        current_label = labels.get(crop_name)

        # Build display
        display = build_display(
            crop_img, full_frame, bbox, all_bboxes,
            crop_name, stadium, labels, idx, len(work_list),
            current_label=current_label,
        )

        cv2.imshow("Player Labeler", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('p'):
            old_label = labels.get(crop_name)
            labels[crop_name] = "pitcher"
            history.append((crop_name, idx, old_label))
            print(f"[{idx+1}/{len(work_list)}] PITCHER: {crop_name}")
            idx += 1

        elif key == ord('c'):
            old_label = labels.get(crop_name)
            labels[crop_name] = "catcher"
            history.append((crop_name, idx, old_label))
            print(f"[{idx+1}/{len(work_list)}] CATCHER: {crop_name}")
            idx += 1

        elif key == ord('r'):
            old_label = labels.get(crop_name)
            labels[crop_name] = "batter"
            history.append((crop_name, idx, old_label))
            print(f"[{idx+1}/{len(work_list)}] BATTER: {crop_name}")
            idx += 1

        elif key == ord('o'):
            old_label = labels.get(crop_name)
            labels[crop_name] = "other"
            history.append((crop_name, idx, old_label))
            print(f"[{idx+1}/{len(work_list)}] OTHER: {crop_name}")
            idx += 1

        elif key == ord('s'):
            print(f"[{idx+1}/{len(work_list)}] SKIP: {crop_name}")
            idx += 1

        elif key == ord('b'):
            if history:
                last_name, last_idx, old_label = history.pop()
                # Restore previous label (or remove if there was none)
                if old_label is not None:
                    labels[last_name] = old_label
                elif last_name in labels:
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
    catcher_count = sum(1 for v in labels.values() if v == "catcher")
    batter_count = sum(1 for v in labels.values() if v == "batter")
    other_count = sum(1 for v in labels.values() if v == "other")
    not_pitcher_count = sum(1 for v in labels.values() if v == "not_pitcher")
    print(f"\nSession complete!")
    print(f"  Pitcher: {pitcher_count}")
    print(f"  Catcher: {catcher_count}")
    print(f"  Batter:  {batter_count}")
    print(f"  Other:   {other_count}")
    if not_pitcher_count:
        print(f"  not_pitcher (legacy): {not_pitcher_count}")
    print(f"  Total labeled: {len(labels)}")


if __name__ == "__main__":
    main()
