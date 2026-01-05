#!/usr/bin/env python
"""Test YOLO model on new frames and videos not in training set."""

import cv2
import random
from pathlib import Path
from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
MODEL_PATH = PROJECT_ROOT / "models/yolo_mitt_diverse/weights/best.pt"
VIDEOS_2024 = PROJECT_ROOT / "data/videos/2024"
OUTPUT_DIR = PROJECT_ROOT / "data/debug/yolo_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_test_videos(n_videos=25):
    """Get random videos from 2024 folder (not in training set)."""
    all_videos = list(VIDEOS_2024.rglob("*.mp4"))
    return random.sample(all_videos, min(n_videos, len(all_videos)))

def extract_and_infer_frames(model, videos, frames_per_video=2):
    """Extract frames from videos and run inference."""
    saved_count = 0

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 10:
            cap.release()
            continue

        # Sample frames from middle of video (where action is)
        frame_indices = random.sample(range(10, max(11, total_frames - 10)),
                                     min(frames_per_video, total_frames - 20))

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Run inference
            results = model(frame, verbose=False)[0]

            # Draw annotations
            annotated = results.plot()

            # Save annotated image
            video_name = video_path.stem
            output_path = OUTPUT_DIR / f"{video_name}_frame{frame_idx}.jpg"
            cv2.imwrite(str(output_path), annotated)
            saved_count += 1

            if saved_count >= 50:
                cap.release()
                return saved_count

        cap.release()

    return saved_count

def infer_on_video(model, video_path, output_name):
    """Run inference on full video and save as mp4."""
    cap = cv2.VideoCapture(str(video_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUTPUT_DIR / output_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, verbose=False)[0]
        annotated = results.plot()

        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()

    return frame_count, output_path

def main():
    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("\n=== Extracting and inferring on 50 test frames ===")
    test_videos = get_test_videos(30)  # Get 30 videos, sample 2 frames each
    saved = extract_and_infer_frames(model, test_videos, frames_per_video=2)
    print(f"Saved {saved} annotated frames to {OUTPUT_DIR}")

    print("\n=== Running inference on 2 full videos ===")
    # Pick 2 random videos for full inference
    full_test_videos = random.sample(list(VIDEOS_2024.rglob("*.mp4")), 2)

    for i, video_path in enumerate(full_test_videos, 1):
        print(f"Processing video {i}: {video_path.name}")
        output_name = f"yolo_test_video_{i}_{video_path.stem}.mp4"
        frames, out_path = infer_on_video(model, video_path, output_name)
        print(f"  -> Saved {frames} frames to {out_path.name}")

    print(f"\n=== Done! All outputs in {OUTPUT_DIR} ===")

if __name__ == "__main__":
    main()
