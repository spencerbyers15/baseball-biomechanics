#!/usr/bin/env python
"""Extract frames from diverse stadium videos for bat labeling."""

import cv2
import random
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
VIDEOS_DIR = PROJECT_ROOT / "data/videos/diverse_stadiums"
OUTPUT_DIR = PROJECT_ROOT / "data/bat_labeling/frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target frames per video - focus on swing portion
FRAMES_PER_VIDEO = 5


def extract_frames():
    """Extract frames from all stadium videos."""
    all_videos = list(VIDEOS_DIR.rglob("*.mp4"))
    print(f"Found {len(all_videos)} videos across stadiums")

    extracted = 0

    for video_path in tqdm(all_videos, desc="Extracting frames"):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 30:
            cap.release()
            continue

        stadium = video_path.parent.name
        video_id = video_path.stem

        # Sample frames from swing portion (30-70% of video)
        # This is where bat is most visible
        start_pct, end_pct = 0.30, 0.70
        frame_indices = [
            int(total_frames * (start_pct + (end_pct - start_pct) * i / (FRAMES_PER_VIDEO - 1)))
            for i in range(FRAMES_PER_VIDEO)
        ]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                filename = f"{stadium}_{video_id}_f{idx:04d}.jpg"
                output_path = OUTPUT_DIR / filename
                cv2.imwrite(str(output_path), frame)
                extracted += 1

        cap.release()

    print(f"\nExtracted {extracted} frames to {OUTPUT_DIR}")
    print(f"Ready for labeling with: python tools/bat_keypoint_labeler.py")


if __name__ == "__main__":
    extract_frames()
