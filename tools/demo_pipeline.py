#!/usr/bin/env python
"""
End-to-end demo pipeline for baseball biomechanics analysis.

Pipeline steps:
1. Scrape a random pitch video from Statcast (2025 season)
2. Crop to main camera angle using scene detection
3. Detect bat barrel, baseball, and home plate on each frame
4. Write annotated video to data/debug/

Usage:
    python tools/demo_pipeline.py
    python tools/demo_pipeline.py --start-date 2025-04-01 --end-date 2025-04-07
    python tools/demo_pipeline.py --video path/to/existing/video.mp4
"""

import argparse
import cv2
import logging
import numpy as np
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scraper.savant import BaseballSavantScraper
from src.scraper.video_downloader import VideoDownloader
from src.detection.baseball_detector import BaseballDetector
from src.detection.home_plate_detector import HomePlateDetector
from src.filtering.scene_cropper import crop_to_main_angle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model paths
BAT_BARREL_WEIGHTS = PROJECT_ROOT / "models/yolo_bat_barrel/train/weights/best.pt"
BASEBALL_WEIGHTS = PROJECT_ROOT / "models/yolo_baseball/train/weights/best.pt"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data/debug/demo_pipeline"

# Visualization settings
COLORS = {
    'baseball': (0, 255, 255),      # Yellow
    'bat_cap': (0, 0, 255),         # Red
    'bat_middle': (0, 255, 0),      # Green
    'bat_beginning': (255, 0, 0),   # Blue
    'bat_skeleton': (255, 255, 0),  # Cyan
    'home_plate': (255, 0, 255),    # Magenta
}


class BatBarrelDetector:
    """Wrapper for bat barrel YOLO-pose model."""

    KEYPOINT_NAMES = ["cap", "middle", "beginning"]
    SKELETON = [[0, 1], [1, 2]]

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self.model = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        from ultralytics import YOLO
        logger.info(f"Loading bat barrel model: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        self._initialized = True

    def detect_frame(self, frame: np.ndarray):
        """Detect bat barrel keypoints in a frame."""
        if not self._initialized:
            self.initialize()

        results = self.model(frame, verbose=False)

        detections = []
        for result in results:
            if result.keypoints is None:
                continue

            keypoints = result.keypoints.data.cpu().numpy()
            boxes = result.boxes

            for i, kps in enumerate(keypoints):
                conf = boxes.conf[i].item() if boxes is not None and len(boxes.conf) > i else 0

                # Extract keypoints with visibility
                kp_data = []
                for idx, (x, y, v) in enumerate(kps):
                    if v > 0.5:
                        kp_data.append({
                            'name': self.KEYPOINT_NAMES[idx],
                            'x': float(x),
                            'y': float(y),
                            'visibility': float(v)
                        })

                if kp_data:
                    detections.append({
                        'keypoints': kp_data,
                        'confidence': conf,
                        'raw_keypoints': kps
                    })

        return detections

    def visualize(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bat barrel detections on frame."""
        vis = frame.copy()

        for det in detections:
            kps = det['raw_keypoints']
            conf = det['confidence']

            # Draw skeleton first
            for j, k in self.SKELETON:
                if kps[j][2] > 0.5 and kps[k][2] > 0.5:
                    pt1 = (int(kps[j][0]), int(kps[j][1]))
                    pt2 = (int(kps[k][0]), int(kps[k][1]))
                    cv2.line(vis, pt1, pt2, COLORS['bat_skeleton'], 3)

            # Draw keypoints
            for idx, (x, y, v) in enumerate(kps):
                if v > 0.5:
                    pt = (int(x), int(y))
                    color = COLORS[f'bat_{self.KEYPOINT_NAMES[idx]}']
                    cv2.circle(vis, pt, 8, color, -1)
                    cv2.circle(vis, pt, 10, (255, 255, 255), 2)

        return vis


class TrainedBaseballDetector:
    """Wrapper for trained YOLO baseball detection model."""

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self.model = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        from ultralytics import YOLO
        logger.info(f"Loading baseball model: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        self._initialized = True

    def detect_frame(self, frame: np.ndarray):
        """Detect baseball in a frame."""
        if not self._initialized:
            self.initialize()

        results = self.model(frame, verbose=False)

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append({
                    'centroid': (cx, cy),
                    'bbox': (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    'confidence': conf
                })

        # Sort by confidence, return best
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def visualize(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw baseball detections on frame."""
        vis = frame.copy()

        for det in detections[:1]:  # Only show best detection
            cx, cy = det['centroid']
            conf = det['confidence']
            x, y, w, h = det['bbox']

            # Draw box
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)),
                         COLORS['baseball'], 2)
            # Draw centroid
            cv2.circle(vis, (int(cx), int(cy)), 6, COLORS['baseball'], -1)
            cv2.circle(vis, (int(cx), int(cy)), 8, (255, 255, 255), 2)

            # Label
            label = f"ball {conf:.0%}"
            cv2.putText(vis, label, (int(x), int(y) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['baseball'], 2)

        return vis


def scrape_random_pitch(start_date: str, end_date: str) -> dict:
    """Scrape a random pitch from Statcast and get video URL."""
    logger.info(f"Searching Statcast for pitches between {start_date} and {end_date}")

    scraper = BaseballSavantScraper(request_delay=1.0)

    # Search for pitches
    df = scraper.search_statcast(
        start_date=start_date,
        end_date=end_date,
        player_type="pitcher"
    )

    if len(df) == 0:
        raise ValueError(f"No pitches found between {start_date} and {end_date}")

    logger.info(f"Found {len(df)} pitches, selecting random one...")

    # Try up to 10 random pitches to find one with a video
    for attempt in range(10):
        row = df.sample(1).iloc[0]

        logger.info(f"Attempt {attempt + 1}: Trying pitch from game {row['game_pk']}, "
                   f"AB {row['at_bat_number']}, pitch {row['pitch_number']}")

        # Get video URL
        video_url = scraper.get_video_url_from_statcast_row(row)

        if video_url:
            game_date = str(row.get('game_date', ''))[:10]

            pitch_info = {
                'game_pk': int(row['game_pk']),
                'at_bat_number': int(row['at_bat_number']),
                'pitch_number': int(row['pitch_number']),
                'game_date': game_date,
                'video_url': video_url,
                'pitcher_name': row.get('player_name', 'Unknown'),
                'pitch_type': row.get('pitch_type', 'Unknown'),
                'release_speed': row.get('release_speed', None),
            }

            logger.info(f"Found video: {pitch_info['pitcher_name']}, "
                       f"{pitch_info['pitch_type']} @ {pitch_info['release_speed']} mph")

            return pitch_info

    raise ValueError("Could not find a pitch with available video after 10 attempts")


def download_video(pitch_info: dict) -> Path:
    """Download video for the pitch."""
    downloader = VideoDownloader(
        download_dir=str(OUTPUT_DIR / "videos"),
        request_delay=0.5
    )

    local_path = downloader.download_video(
        url=pitch_info['video_url'],
        game_pk=pitch_info['game_pk'],
        at_bat_number=pitch_info['at_bat_number'],
        pitch_number=pitch_info['pitch_number'],
        game_date=pitch_info['game_date'],
        show_progress=True
    )

    if not local_path:
        raise ValueError("Failed to download video")

    return Path(local_path)


def crop_to_main_view(video_path: Path) -> Path:
    """Crop video to main camera angle."""
    output_path = video_path.parent / f"{video_path.stem}_cropped.mp4"

    logger.info("Detecting scene cuts and classifying segments...")

    result = crop_to_main_angle(
        video_path=str(video_path),
        output_path=str(output_path),
        keep_segments="longest",
        detection_method="histogram",
        min_segment_duration=0.5,
        samples_per_segment=3,
        show_progress=True
    )

    if not result['success']:
        logger.warning("Scene cropping failed, using original video")
        return video_path

    logger.info(f"Cropped to main angle: {result['crop_regions']}")
    return output_path


def process_video_with_detections(
    video_path: Path,
    output_path: Path,
    bat_detector: BatBarrelDetector,
    ball_detector: TrainedBaseballDetector,
    plate_detector: HomePlateDetector
) -> dict:
    """Process video and overlay all detections."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Detection counters
    stats = {
        'total_frames': total_frames,
        'bat_detections': 0,
        'ball_detections': 0,
        'plate_detections': 0
    }

    # Get home plate detection from first few frames (it's static)
    plate_detection = None
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        plate_detection = plate_detector.detect_frame(frame)
        if plate_detection:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    logger.info(f"Processing {total_frames} frames...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()

        # Detect bat barrel
        bat_dets = bat_detector.detect_frame(frame)
        if bat_dets:
            stats['bat_detections'] += 1
            vis = bat_detector.visualize(vis, bat_dets)

        # Detect baseball
        ball_dets = ball_detector.detect_frame(frame)
        if ball_dets:
            stats['ball_detections'] += 1
            vis = ball_detector.visualize(vis, ball_dets)

        # Draw home plate (use cached detection)
        if plate_detection:
            vis = plate_detector.visualize(vis, plate_detection)
            stats['plate_detections'] += 1

        # Add info overlay
        info_lines = [
            f"Frame: {frame_idx + 1}/{total_frames}",
            f"Bat: {'YES' if bat_dets else 'no'}",
            f"Ball: {'YES' if ball_dets else 'no'}",
            f"Plate: {'YES' if plate_detection else 'no'}"
        ]

        y_offset = height - 100
        for line in info_lines:
            cv2.putText(vis, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25

        out.write(vis)
        frame_idx += 1

        if frame_idx % 30 == 0:
            logger.info(f"  Processed frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    logger.info(f"Output saved to: {output_path}")
    logger.info(f"Stats: {stats}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="End-to-end demo pipeline")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date (YYYY-MM-DD), default: 7 days ago from 2025 season")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD), default: today or end of available data")
    parser.add_argument("--video", type=Path, default=None,
                        help="Use existing video instead of scraping")
    parser.add_argument("--skip-crop", action="store_true",
                        help="Skip temporal cropping step")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Custom output filename (without extension)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine dates for 2025 season
    # MLB 2025 season starts around late March
    if args.start_date is None:
        # Default to early 2025 season
        args.start_date = "2025-04-01"
    if args.end_date is None:
        args.end_date = "2025-04-15"

    # Check model weights exist
    if not BAT_BARREL_WEIGHTS.exists():
        logger.error(f"Bat barrel weights not found: {BAT_BARREL_WEIGHTS}")
        logger.error("Train with: python tools/train_yolo_bat_barrel.py")
        return 1

    if not BASEBALL_WEIGHTS.exists():
        logger.error(f"Baseball weights not found: {BASEBALL_WEIGHTS}")
        logger.error("Train with: python tools/train_yolo_baseball.py")
        return 1

    # Initialize detectors
    logger.info("Initializing detectors...")
    bat_detector = BatBarrelDetector(BAT_BARREL_WEIGHTS)
    bat_detector.initialize()

    ball_detector = TrainedBaseballDetector(BASEBALL_WEIGHTS)
    ball_detector.initialize()

    plate_detector = HomePlateDetector()
    plate_detector.initialize()

    # Get video
    if args.video:
        # Use provided video
        video_path = args.video
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return 1
        pitch_info = {'game_pk': 0, 'at_bat_number': 0, 'pitch_number': 0}
    else:
        # Scrape random pitch
        try:
            pitch_info = scrape_random_pitch(args.start_date, args.end_date)
        except Exception as e:
            logger.error(f"Failed to scrape pitch: {e}")
            return 1

        # Download video
        try:
            video_path = download_video(pitch_info)
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return 1

    # Crop to main angle
    if not args.skip_crop:
        try:
            video_path = crop_to_main_view(video_path)
        except Exception as e:
            logger.warning(f"Scene cropping failed: {e}, using original video")

    # Determine output filename
    if args.output_name:
        output_name = args.output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"demo_{pitch_info['game_pk']}_{pitch_info['at_bat_number']}_{pitch_info['pitch_number']}_{timestamp}"

    output_path = OUTPUT_DIR / f"{output_name}_annotated.mp4"

    # Process video with all detections
    logger.info("Processing video with detections...")
    try:
        stats = process_video_with_detections(
            video_path=video_path,
            output_path=output_path,
            bat_detector=bat_detector,
            ball_detector=ball_detector,
            plate_detector=plate_detector
        )
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print("\n" + "="*60)
    print("DEMO PIPELINE COMPLETE")
    print("="*60)
    print(f"Output video: {output_path}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Bat barrel detections: {stats['bat_detections']} ({100*stats['bat_detections']/stats['total_frames']:.1f}%)")
    print(f"Baseball detections: {stats['ball_detections']} ({100*stats['ball_detections']/stats['total_frames']:.1f}%)")
    print(f"Home plate detected: {'Yes' if stats['plate_detections'] > 0 else 'No'}")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
