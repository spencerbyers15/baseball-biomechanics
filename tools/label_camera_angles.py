#!/usr/bin/env python
"""Quick labeling tool for camera angle frames.

Usage:
    python tools/label_camera_angles.py [--frames-dir PATH] [--output PATH] [--limit N]

Controls:
    m - Label as main_angle (standard broadcast view)
    o - Label as other (replay, close-up, etc.)
    s - Skip this frame
    q - Quit and save
    b - Go back to previous frame
"""

import cv2
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data/stratified_label_frames"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/camera_angle_labels.json"


def load_existing_labels(output_path: Path) -> dict:
    """Load existing labels for resume support."""
    if output_path.exists():
        with open(output_path, "r") as f:
            data = json.load(f)
            return data.get("labels", {})
    return {}


def save_labels(output_path: Path, labels: dict, metadata: dict):
    """Save labels to JSON."""
    data = {
        "metadata": metadata,
        "labels": labels,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(labels)} labels to {output_path}")


def get_unlabeled_frames(frames_dir: Path, existing_labels: dict, limit: int = None) -> list:
    """Get list of frames that haven't been labeled yet."""
    all_frames = sorted(frames_dir.rglob("*.jpg"))

    # Filter out already labeled
    unlabeled = [f for f in all_frames if str(f) not in existing_labels]

    print(f"Found {len(all_frames)} total frames, {len(existing_labels)} already labeled")
    print(f"Remaining unlabeled: {len(unlabeled)}")

    if limit:
        unlabeled = unlabeled[:limit]
        print(f"Limiting to {limit} frames")

    return unlabeled


def display_frame(frame_path: Path, labels: dict, current_idx: int, total: int):
    """Display a frame with info overlay."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    # Get counts
    main_count = sum(1 for v in labels.values() if v == "main_angle")
    other_count = sum(1 for v in labels.values() if v == "other")

    # Add info overlay
    h, w = img.shape[:2]

    # Semi-transparent bar at top
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # Text info
    cv2.putText(img, f"Frame {current_idx + 1}/{total}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Main: {main_count}  Other: {other_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(img, "[M]ain  [O]ther  [S]kip  [B]ack  [Q]uit", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Stadium info from path
    stadium = frame_path.parent.name
    cv2.putText(img, f"Stadium: {stadium}", (w - 300, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img


def main():
    parser = argparse.ArgumentParser(description="Label camera angles in frames")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR,
                        help="Directory containing extracted frames")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON file for labels")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max frames to label in this session")
    parser.add_argument("--random", action="store_true",
                        help="Randomize frame order")
    args = parser.parse_args()

    # Load existing labels
    labels = load_existing_labels(args.output)

    # Get unlabeled frames
    frames = get_unlabeled_frames(args.frames_dir, labels, args.limit)

    if not frames:
        print("No unlabeled frames found!")
        return

    if args.random:
        import random
        random.shuffle(frames)

    # Labeling loop
    history = []  # For going back
    idx = 0

    cv2.namedWindow("Camera Angle Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Angle Labeler", 1280, 720)

    print("\nStarting labeling session...")
    print("Controls: [M]ain angle, [O]ther, [S]kip, [B]ack, [Q]uit")
    print("-" * 50)

    while idx < len(frames):
        frame_path = frames[idx]
        img = display_frame(frame_path, labels, idx, len(frames))

        if img is None:
            print(f"Could not load: {frame_path}")
            idx += 1
            continue

        cv2.imshow("Camera Angle Labeler", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('m'):
            labels[str(frame_path)] = "main_angle"
            history.append((str(frame_path), idx))
            print(f"[{idx+1}/{len(frames)}] MAIN: {frame_path.name}")
            idx += 1

        elif key == ord('o'):
            labels[str(frame_path)] = "other"
            history.append((str(frame_path), idx))
            print(f"[{idx+1}/{len(frames)}] OTHER: {frame_path.name}")
            idx += 1

        elif key == ord('s'):
            print(f"[{idx+1}/{len(frames)}] SKIP: {frame_path.name}")
            idx += 1

        elif key == ord('b'):
            if history:
                last_path, last_idx = history.pop()
                # Remove the last label
                if last_path in labels:
                    del labels[last_path]
                idx = last_idx
                print(f"Going back to frame {idx + 1}")
            else:
                print("No history to go back to")

        elif key == ord('q') or key == 27:  # q or ESC
            print("\nQuitting...")
            break

    cv2.destroyAllWindows()

    # Save results
    metadata = {
        "created": datetime.now().isoformat(),
        "frames_dir": str(args.frames_dir),
        "total_labeled": len(labels),
        "main_angle_count": sum(1 for v in labels.values() if v == "main_angle"),
        "other_count": sum(1 for v in labels.values() if v == "other"),
    }
    save_labels(args.output, labels, metadata)

    print(f"\nSession complete!")
    print(f"  Main angle: {metadata['main_angle_count']}")
    print(f"  Other: {metadata['other_count']}")
    print(f"  Total: {metadata['total_labeled']}")


if __name__ == "__main__":
    main()
