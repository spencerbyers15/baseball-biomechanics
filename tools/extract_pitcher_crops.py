#!/usr/bin/env python
"""Extract person crops from cropped pitcher calibration videos for labeling.

Samples frames across the 1744-video dataset, runs YOLO person detection on
each, and saves every detected person as a separate cropped image. These crops
are then labeled as pitcher/not_pitcher using label_pitcher_crops.py.

Sampling strategy:
    - 1 video per stadium per season (30 stadiums x 3 seasons = ~90 videos)
    - ~5 frames per video, evenly spaced (skip first/last 10% for transitions)
    - ~450 frames x ~5 people/frame = ~2000-3000 person crops

Usage:
    python tools/extract_pitcher_crops.py
    python tools/extract_pitcher_crops.py --per-stadium 2 --frames-per-video 8
    python tools/extract_pitcher_crops.py --video-dir path/to/videos
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data/videos/pitcher_calibration_cropped"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/labels/pitcher/crops"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data/labels/pitcher/crop_metadata.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def discover_videos(video_dir: Path, per_stadium: int = 1, seed: int = 42) -> list:
    """Discover videos with stadium/season structure, sample per_stadium per season.

    Expected structure: video_dir/{Stadium}/{Season}/*.mp4

    Returns list of (video_path, stadium, season) tuples.
    """
    rng = random.Random(seed)
    selected = []

    stadiums = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name != "no_main_angle_round3"])
    logger.info(f"Found {len(stadiums)} stadiums in {video_dir}")

    for stadium_dir in stadiums:
        stadium = stadium_dir.name
        seasons = sorted([d for d in stadium_dir.iterdir() if d.is_dir()])

        for season_dir in seasons:
            season = season_dir.name
            videos = sorted(season_dir.glob("*.mp4"))

            if not videos:
                continue

            # Sample per_stadium videos from this season
            n = min(per_stadium, len(videos))
            picks = rng.sample(videos, n)

            for v in picks:
                selected.append((v, stadium, season))

    logger.info(f"Selected {len(selected)} videos ({per_stadium}/stadium/season)")
    return selected


def sample_frame_indices(total_frames: int, n_frames: int = 5) -> list:
    """Pick evenly-spaced frame indices, skipping first/last 10%."""
    start = int(total_frames * 0.10)
    end = int(total_frames * 0.90)

    if end <= start:
        return [total_frames // 2]

    span = end - start
    if n_frames <= 1:
        return [start + span // 2]

    step = span / (n_frames - 1)
    return [int(start + i * step) for i in range(n_frames)]


def run_yolo_on_frame(yolo_model, frame: np.ndarray, conf: float = 0.3) -> list:
    """Run YOLO person detection, return list of bbox dicts."""
    h, w = frame.shape[:2]
    results = yolo_model(frame, classes=[0], conf=conf, verbose=False)

    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            det_conf = float(box.conf[0].cpu().numpy())
            cx_norm = (x1 + x2) / 2 / w
            cy_norm = (y1 + y2) / 2 / h

            detections.append({
                "idx": i,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cx_norm": round(cx_norm, 4),
                "cy_norm": round(cy_norm, 4),
                "conf": round(det_conf, 3),
            })

    return detections


def crop_person(frame: np.ndarray, bbox: list, padding: int = 10) -> np.ndarray:
    """Crop person from frame with small padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    cx1 = max(0, int(x1) - padding)
    cy1 = max(0, int(y1) - padding)
    cx2 = min(w, int(x2) + padding)
    cy2 = min(h, int(y2) + padding)

    return frame[cy1:cy2, cx1:cx2]


def make_crop_name(stadium: str, season: str, video_name: str, frame_num: int, det_idx: int) -> str:
    """Generate a unique crop filename."""
    vid_stem = Path(video_name).stem
    return f"{stadium}_{season}_{vid_stem}_f{frame_num:04d}_d{det_idx}.jpg"


def main():
    parser = argparse.ArgumentParser(description="Extract person crops for pitcher labeling")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--per-stadium", type=int, default=1,
                        help="Videos per stadium per season (default: 1)")
    parser.add_argument("--frames-per-video", type=int, default=5,
                        help="Frames to sample per video (default: 5)")
    parser.add_argument("--person-conf", type=float, default=0.3,
                        help="YOLO person detection confidence threshold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-crop-size", type=int, default=30,
                        help="Minimum crop width or height in pixels")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover and sample videos
    videos = discover_videos(args.video_dir, args.per_stadium, args.seed)
    if not videos:
        logger.error("No videos found!")
        sys.exit(1)

    # Load YOLO
    logger.info("Loading YOLO model...")
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    yolo.to("cuda")
    logger.info("YOLO loaded on GPU")

    # Process videos
    metadata = {"crops": {}}
    total_crops = 0
    total_frames_processed = 0

    for vid_i, (video_path, stadium, season) in enumerate(videos):
        logger.info(f"[{vid_i+1}/{len(videos)}] {stadium}/{season}/{video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"  Could not open {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            logger.warning(f"  Too few frames ({total_frames}), skipping")
            cap.release()
            continue

        frame_indices = sample_frame_indices(total_frames, args.frames_per_video)
        video_crops = 0

        for frame_num in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            total_frames_processed += 1
            detections = run_yolo_on_frame(yolo, frame, args.person_conf)

            for det in detections:
                crop = crop_person(frame, det["bbox"])
                ch, cw = crop.shape[:2]

                if cw < args.min_crop_size or ch < args.min_crop_size:
                    continue

                crop_name = make_crop_name(stadium, season, video_path.name, frame_num, det["idx"])
                crop_path = args.output_dir / crop_name

                cv2.imwrite(str(crop_path), crop)

                metadata["crops"][crop_name] = {
                    "video": str(video_path.relative_to(PROJECT_ROOT)),
                    "stadium": stadium,
                    "season": season,
                    "frame_num": frame_num,
                    "det_idx": det["idx"],
                    "bbox": det["bbox"],
                    "cx_norm": det["cx_norm"],
                    "cy_norm": det["cy_norm"],
                    "conf": det["conf"],
                }

                total_crops += 1
                video_crops += 1

        cap.release()
        logger.info(f"  {video_crops} crops from {len(frame_indices)} frames (total: {total_frames} frames)")

    # Save metadata
    metadata["summary"] = {
        "total_crops": total_crops,
        "total_frames": total_frames_processed,
        "total_videos": len(videos),
        "per_stadium": args.per_stadium,
        "frames_per_video": args.frames_per_video,
        "person_conf": args.person_conf,
        "seed": args.seed,
    }

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nDone! {total_crops} crops from {total_frames_processed} frames across {len(videos)} videos")
    logger.info(f"Crops saved to: {args.output_dir}")
    logger.info(f"Metadata saved to: {args.metadata}")

    # Stadium distribution
    stadium_counts = {}
    for info in metadata["crops"].values():
        s = info["stadium"]
        stadium_counts[s] = stadium_counts.get(s, 0) + 1

    logger.info(f"\nPer-stadium crop counts:")
    for s in sorted(stadium_counts):
        logger.info(f"  {s}: {stadium_counts[s]}")


if __name__ == "__main__":
    main()
