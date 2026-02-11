#!/usr/bin/env python
"""Extract main broadcast angle reference frames from videos."""

import cv2
import random
from pathlib import Path

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
VIDEOS_DIR = PROJECT_ROOT / "data/videos/diverse_stadiums"
OUTPUT_DIR = PROJECT_ROOT / "data/reference_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get ~20 frames from different stadiums (first few frames are usually main angle)
FRAMES_PER_STADIUM = 1
TARGET_FRAME = 5  # Frame 5 is typically pre-pitch main angle


def extract_reference_frames():
    """Extract early frames (main broadcast angle) from each stadium."""
    stadiums = [d for d in VIDEOS_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(stadiums)} stadiums")

    extracted = 0
    for stadium_dir in stadiums:
        videos = list(stadium_dir.glob("*.mp4"))
        if not videos:
            continue

        # Pick random video from this stadium
        video_path = random.choice(videos)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            continue

        # Get early frame (main angle before any cuts)
        cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
        ret, frame = cap.read()

        if ret:
            filename = f"{stadium_dir.name}_ref.jpg"
            output_path = OUTPUT_DIR / filename
            cv2.imwrite(str(output_path), frame)
            extracted += 1
            print(f"Extracted: {filename}")

        cap.release()

    print(f"\nExtracted {extracted} reference frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_reference_frames()
