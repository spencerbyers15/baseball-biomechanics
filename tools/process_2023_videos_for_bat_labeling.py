#!/usr/bin/env python
"""
Process all 2023 videos through scene cropper and extract frames for bat labeling.

Steps:
1. Find all 2023 round 2 videos
2. Run scene cropper on each (keep longest main_angle segment)
3. Extract 10 random frames from each cropped video
4. Save frames for bat keypoint labeling
"""

import cv2
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filtering.scene_cropper import crop_to_main_angle

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
VIDEOS_DIR = PROJECT_ROOT / "data/videos/2023_round2"
CROPPED_DIR = PROJECT_ROOT / "data/videos/2023_cropped"
FRAMES_DIR = PROJECT_ROOT / "data/bat_frames_round2_filtered"

FRAMES_PER_VIDEO = 10


def find_all_videos():
    """Find all videos in the 2023 round 2 directory."""
    videos = []
    for stadium_dir in VIDEOS_DIR.iterdir():
        if stadium_dir.is_dir():
            for video_file in stadium_dir.glob("*.mp4"):
                videos.append({
                    "path": video_file,
                    "stadium": stadium_dir.name,
                    "video_id": video_file.stem,
                })
    return videos


def process_video(video_info, classifier=None):
    """Process a single video through scene cropper."""
    video_path = video_info["path"]
    stadium = video_info["stadium"]
    video_id = video_info["video_id"]

    # Create output directory for this stadium
    stadium_crop_dir = CROPPED_DIR / stadium
    stadium_crop_dir.mkdir(parents=True, exist_ok=True)

    output_path = stadium_crop_dir / f"{video_id}_cropped.mp4"

    # Skip if already processed
    if output_path.exists():
        return {
            "success": True,
            "skipped": True,
            "output_path": str(output_path),
            "video_id": video_id,
            "stadium": stadium,
        }

    try:
        result = crop_to_main_angle(
            str(video_path),
            str(output_path),
            keep_segments="longest",
            detection_method="histogram",
            samples_per_segment=3,
            classifier=classifier,
            show_progress=False,
        )

        return {
            "success": result["success"],
            "skipped": False,
            "output_path": str(output_path) if result["success"] else None,
            "video_id": video_id,
            "stadium": stadium,
            "num_cuts": result["detection"]["num_cuts"],
            "num_segments": len(result["segments"]),
            "main_segments": sum(1 for s in result["segments"] if s["label"] == "main_angle"),
            "other_segments": sum(1 for s in result["segments"] if s["label"] == "other"),
            "crop_duration": result["crop_regions"][0][1] - result["crop_regions"][0][0] if result["crop_regions"] else 0,
        }
    except Exception as e:
        return {
            "success": False,
            "skipped": False,
            "error": str(e),
            "video_id": video_id,
            "stadium": stadium,
        }


def extract_frames_from_cropped(cropped_path, output_dir, video_id, num_frames=10):
    """Extract random frames from cropped video."""
    cap = cv2.VideoCapture(str(cropped_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        # If video is short, take all frames evenly distributed
        frame_indices = list(range(0, total_frames, max(1, total_frames // num_frames)))[:num_frames]
    else:
        # Random frames, but avoid first/last 10%
        start = int(total_frames * 0.1)
        end = int(total_frames * 0.9)
        frame_indices = sorted(random.sample(range(start, end), min(num_frames, end - start)))

    saved_frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"{video_id}_f{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(str(frame_path))

    cap.release()
    return saved_frames


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cropping", action="store_true", help="Skip cropping, only extract frames")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip frame extraction")
    args = parser.parse_args()

    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Find all videos
    videos = find_all_videos()
    print(f"Found {len(videos)} videos across {len(set(v['stadium'] for v in videos))} stadiums")

    # Phase 1: Crop videos
    if not args.skip_cropping:
        print(f"\n{'='*60}")
        print("PHASE 1: Cropping videos to main_angle segments")
        print(f"{'='*60}")

        # Initialize classifier once (expensive to load)
        from src.filtering.camera_filter import CameraAngleFilter
        classifier = CameraAngleFilter()
        classifier.initialize()

        crop_results = []
        for i, video_info in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}] {video_info['stadium']}/{video_info['video_id']}")

            result = process_video(video_info, classifier)
            crop_results.append(result)

            if result["skipped"]:
                print(f"  Skipped (already exists)")
            elif result["success"]:
                print(f"  Cuts: {result['num_cuts']}, Segments: {result['num_segments']} "
                      f"({result['main_segments']} main, {result['other_segments']} other)")
                print(f"  Cropped duration: {result['crop_duration']:.2f}s")
            else:
                print(f"  FAILED: {result.get('error', 'Unknown error')}")

        # Summary
        successful = sum(1 for r in crop_results if r["success"])
        failed = sum(1 for r in crop_results if not r["success"])
        print(f"\nCropping complete: {successful} successful, {failed} failed")

        # Save results
        results_path = CROPPED_DIR / "crop_results.json"
        with open(results_path, "w") as f:
            json.dump(crop_results, f, indent=2)
        print(f"Results saved to: {results_path}")

    # Phase 2: Extract frames
    if not args.skip_extraction:
        print(f"\n{'='*60}")
        print("PHASE 2: Extracting frames from cropped videos")
        print(f"{'='*60}")

        # Find all cropped videos
        cropped_videos = list(CROPPED_DIR.rglob("*_cropped.mp4"))
        print(f"Found {len(cropped_videos)} cropped videos")

        all_frames = []
        frames_info = []

        for i, cropped_path in enumerate(cropped_videos):
            stadium = cropped_path.parent.name
            video_id = cropped_path.stem.replace("_cropped", "")

            print(f"[{i+1}/{len(cropped_videos)}] {stadium}/{video_id}")

            frames = extract_frames_from_cropped(
                cropped_path, FRAMES_DIR, f"{stadium}_{video_id}", FRAMES_PER_VIDEO
            )

            for frame_path in frames:
                frames_info.append({
                    "path": frame_path,
                    "stadium": stadium,
                    "video_id": video_id,
                    "source": "2023_filtered",
                })

            all_frames.extend(frames)
            print(f"  Extracted {len(frames)} frames")

        # Save frames info
        frames_info_path = FRAMES_DIR / "frames_info.json"
        with open(frames_info_path, "w") as f:
            json.dump(frames_info, f, indent=2)

        print(f"\n{'='*60}")
        print("COMPLETE!")
        print(f"{'='*60}")
        print(f"Total frames extracted: {len(all_frames)}")
        print(f"Frames saved to: {FRAMES_DIR}")
        print(f"Frames info: {frames_info_path}")
        print(f"\nNext step: Run bat_keypoint_labeler.py on {FRAMES_DIR}")


if __name__ == "__main__":
    random.seed(42)
    main()
