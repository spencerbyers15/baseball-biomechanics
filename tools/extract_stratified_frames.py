#!/usr/bin/env python
"""Extract frames stratified by stadium for labeling.

Ensures exactly 5 frames per stadium, each from a different video.
"""

import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
FRAMES_DIR = PROJECT_ROOT / "data/all_frames_by_stadium"
OUTPUT_DIR = PROJECT_ROOT / "data/stratified_label_frames"

FRAMES_PER_STADIUM = 5


def main():
    # Clean output dir
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    stadiums = sorted([d for d in FRAMES_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(stadiums)} stadiums")

    total_copied = 0

    for stadium_dir in stadiums:
        # Group frames by video
        frames_by_video = {}
        for frame in stadium_dir.glob("*.jpg"):
            # Extract video name from frame name (e.g., "MIL_NYM_745931_17_1_f0043.jpg" -> "MIL_NYM_745931_17_1")
            video_name = "_".join(frame.stem.rsplit("_", 1)[0].rsplit("_", 0))
            # Actually: frame.stem = "MIL_NYM_745931_17_1_f0043", split on last "_f"
            parts = frame.stem.rsplit("_f", 1)
            if len(parts) == 2:
                video_name = parts[0]
            else:
                video_name = frame.stem

            if video_name not in frames_by_video:
                frames_by_video[video_name] = []
            frames_by_video[video_name].append(frame)

        # Get unique videos
        videos = list(frames_by_video.keys())

        if len(videos) < FRAMES_PER_STADIUM:
            print(f"  {stadium_dir.name}: only {len(videos)} videos, using all")
            selected_videos = videos
        else:
            selected_videos = random.sample(videos, FRAMES_PER_STADIUM)

        # Pick one random frame from each selected video
        for video_name in selected_videos:
            frames = frames_by_video[video_name]
            # Pick frame from middle of video (more likely to be game action)
            mid_idx = len(frames) // 2
            frame = frames[mid_idx]

            # Copy to output with stadium prefix
            dest = OUTPUT_DIR / f"{stadium_dir.name}__{frame.name}"
            shutil.copy(frame, dest)
            total_copied += 1

        print(f"  {stadium_dir.name}: {len(selected_videos)} frames from {len(videos)} videos")

    print(f"\nTotal: {total_copied} frames copied to {OUTPUT_DIR}")
    print(f"Expected: {len(stadiums) * FRAMES_PER_STADIUM} = {len(stadiums)} stadiums * {FRAMES_PER_STADIUM} frames")


if __name__ == "__main__":
    main()
