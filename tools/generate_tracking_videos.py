"""
Generate tracking videos using SAM2 video predictor and MediaPipe pose estimation.

Uses labeled frame data to initialize tracking, then generates:
1. Pitcher skeleton video (white background)
2. Batter skeleton video (white background)
3. Catcher glove tracking video (white background)
4. Full overlay video (skeletons + glove on original)
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress verbose output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import cv2
import numpy as np

# Configure logging - only errors and final results
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class VideoTrackingPipeline:
    """Pipeline for tracking players and generating output videos."""

    SKELETON_CONNECTIONS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
    ]

    COLORS = {
        "pitcher": (255, 165, 0),   # Orange
        "batter": (0, 255, 0),      # Green
        "glove": (255, 0, 255),     # Magenta
    }

    def __init__(self, project_dir: str, cache_dir: str = "F:/hf_cache"):
        self.project_dir = Path(project_dir)
        self.cache_dir = cache_dir
        self.labels_dir = self.project_dir / "data" / "labels"
        self.output_dir = self.project_dir / "data" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sam_predictor = None
        self.pose_backend = None
        self.device = None

    def _init_sam(self):
        """Initialize SAM2 video predictor."""
        if self.sam_predictor is not None:
            return

        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try SAM2 from HuggingFace
        try:
            from sam2.build_sam import build_sam2_video_predictor

            # Use SAM2 hiera large for best quality
            sam2_checkpoint = Path(self.cache_dir) / "sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"

            if not sam2_checkpoint.exists():
                logger.info("Downloading SAM2 checkpoint...")
                import urllib.request
                url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
                sam2_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, sam2_checkpoint)

            self.sam_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
            logger.info(f"SAM2 video predictor loaded on {self.device}")
            return
        except ImportError:
            pass

        # Fallback to Ultralytics SAM
        try:
            from ultralytics import SAM
            self.sam_predictor = SAM("sam2.1_b.pt")
            logger.info("Loaded Ultralytics SAM2.1")
            return
        except ImportError:
            pass

        logger.warning("SAM2 not available - using bbox tracking fallback")

    def _init_pose(self):
        """Initialize MediaPipe pose backend."""
        if self.pose_backend is not None:
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Download model
            model_path = self._download_mediapipe_model()

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=3,  # Detect multiple poses
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose_backend = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as e:
            logger.error(f"MediaPipe init failed: {e}")
            raise

    def _download_mediapipe_model(self) -> str:
        """Download MediaPipe pose model."""
        import urllib.request

        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "pose_landmarker_full.task"

        if not model_path.exists():
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
            urllib.request.urlretrieve(url, model_path)

        return str(model_path)

    def load_labels(self) -> Tuple[Dict, List]:
        """Load labeled frame data."""
        labels_file = self.labels_dir / "labels.json"
        frames_info_file = self.labels_dir / "frames_info.json"

        with open(labels_file) as f:
            labels = json.load(f)

        with open(frames_info_file) as f:
            frames_info = json.load(f)

        return labels, frames_info

    def get_video_label_data(self, video_path: str, labels: Dict, frames_info: List) -> Optional[Dict]:
        """Get label data for a specific video."""
        video_name = Path(video_path).stem

        for info in frames_info:
            if video_name in info.get("source_video", ""):
                frame_name = Path(info["frame_path"]).name
                if frame_name in labels:
                    frame_labels = labels[frame_name]
                    if frame_labels:  # Non-empty labels
                        return {
                            "frame_idx": info["source_frame_idx"],
                            "labels": frame_labels,
                            "total_frames": info["total_frames"],
                        }
        return None

    def track_objects_bbox(
        self,
        video_path: str,
        init_points: Dict[str, Tuple[int, int]],
        init_frame_idx: int,
    ) -> Dict[str, List[Optional[Tuple[int, int, int, int]]]]:
        """
        Simple bbox tracking using optical flow or template matching.
        Returns dict mapping label -> list of bboxes per frame.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize trackers
        trackers = {}
        tracks = {label: [None] * total_frames for label in init_points.keys()}

        # Seek to init frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_idx)
        ret, init_frame = cap.read()
        if not ret:
            cap.release()
            return tracks

        # Initialize CSRT trackers for each object
        for label, point in init_points.items():
            x, y = point
            # Create initial bbox around point
            bbox_size = 80 if label == "glove" else 150
            x1 = max(0, x - bbox_size // 2)
            y1 = max(0, y - bbox_size // 2)
            bbox = (x1, y1, bbox_size, bbox_size)

            # Use TrackerMIL which is available in opencv-contrib
            try:
                tracker = cv2.TrackerVit_create()
            except:
                tracker = cv2.TrackerMIL_create()
            tracker.init(init_frame, bbox)
            trackers[label] = tracker
            tracks[label][init_frame_idx] = bbox

        # Track forward
        for frame_idx in range(init_frame_idx + 1, total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            for label, tracker in trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    tracks[label][frame_idx] = tuple(map(int, bbox))

        # Reinitialize and track backward
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_idx)
        ret, init_frame = cap.read()

        for label, point in init_points.items():
            x, y = point
            bbox_size = 80 if label == "glove" else 150
            x1 = max(0, x - bbox_size // 2)
            y1 = max(0, y - bbox_size // 2)
            bbox = (x1, y1, bbox_size, bbox_size)

            try:
                tracker = cv2.TrackerVit_create()
            except:
                tracker = cv2.TrackerMIL_create()
            tracker.init(init_frame, bbox)
            trackers[label] = tracker

        for frame_idx in range(init_frame_idx - 1, -1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            for label, tracker in trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    tracks[label][frame_idx] = tuple(map(int, bbox))

        cap.release()
        return tracks

    def estimate_pose_in_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Estimate pose for person in bbox.
        Returns dict mapping landmark name -> (x, y, confidence).
        """
        self._init_pose()

        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        # Add padding
        pad = 30
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_frame, x + w + pad)
        y2 = min(h_frame, y + h + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Convert to RGB for MediaPipe
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=crop_rgb)

        result = self.pose_backend.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = result.pose_landmarks[0]
        crop_h, crop_w = crop.shape[:2]

        pose = {}
        landmark_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index",
        ]

        for idx, lm in enumerate(landmarks):
            # Convert back to full frame coordinates
            px = lm.x * crop_w + x1
            py = lm.y * crop_h + y1
            conf = lm.visibility if hasattr(lm, 'visibility') else 1.0
            pose[landmark_names[idx]] = (px, py, conf)

        return pose

    def draw_skeleton(
        self,
        canvas: np.ndarray,
        pose: Dict[str, Tuple[float, float, float]],
        color: Tuple[int, int, int],
        thickness: int = 2,
        min_confidence: float = 0.3,
    ) -> np.ndarray:
        """Draw skeleton on canvas."""
        if pose is None:
            return canvas

        # Draw connections
        for start_name, end_name in self.SKELETON_CONNECTIONS:
            if start_name in pose and end_name in pose:
                start = pose[start_name]
                end = pose[end_name]

                if start[2] >= min_confidence and end[2] >= min_confidence:
                    pt1 = (int(start[0]), int(start[1]))
                    pt2 = (int(end[0]), int(end[1]))
                    cv2.line(canvas, pt1, pt2, color, thickness)

        # Draw keypoints
        for name, (x, y, conf) in pose.items():
            if conf >= min_confidence:
                cv2.circle(canvas, (int(x), int(y)), 4, color, -1)

        return canvas

    def draw_glove_marker(
        self,
        canvas: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (255, 0, 255),
    ) -> np.ndarray:
        """Draw glove tracking marker."""
        if bbox is None:
            return canvas

        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2

        # Draw crosshair
        size = 20
        cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, 2)
        cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, 2)
        cv2.circle(canvas, (cx, cy), 8, color, 2)

        # Draw bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)

        return canvas

    def process_video(
        self,
        video_path: str,
        label_data: Dict,
        output_prefix: str,
    ) -> Dict[str, str]:
        """
        Process a single video and generate all 4 output videos.

        Returns dict with paths to generated videos.
        """
        logger.info(f"Processing: {Path(video_path).name}")

        # Extract init points from labels
        init_points = {}
        for label, data in label_data["labels"].items():
            if "point" in data:
                init_points[label] = tuple(data["point"])

        if not init_points:
            logger.warning("No valid init points found")
            return {}

        # Track objects through video
        logger.info("  Tracking objects...")
        tracks = self.track_objects_bbox(
            video_path,
            init_points,
            label_data["frame_idx"],
        )

        # Open video for reading
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_paths = {
            "pitcher": self.output_dir / f"{output_prefix}_pitcher_skeleton.mp4",
            "batter": self.output_dir / f"{output_prefix}_batter_skeleton.mp4",
            "glove": self.output_dir / f"{output_prefix}_glove_tracking.mp4",
            "overlay": self.output_dir / f"{output_prefix}_full_overlay.mp4",
        }

        writers = {
            "pitcher": cv2.VideoWriter(str(output_paths["pitcher"]), fourcc, fps, (width, height)),
            "batter": cv2.VideoWriter(str(output_paths["batter"]), fourcc, fps, (width, height)),
            "glove": cv2.VideoWriter(str(output_paths["glove"]), fourcc, fps, (width, height)),
            "overlay": cv2.VideoWriter(str(output_paths["overlay"]), fourcc, fps, (width, height)),
        }

        self._init_pose()

        logger.info("  Generating output videos...")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # White background canvases
            white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
            pitcher_canvas = white_bg.copy()
            batter_canvas = white_bg.copy()
            glove_canvas = white_bg.copy()
            overlay_canvas = frame.copy()

            # Process pitcher
            if "pitcher" in tracks and tracks["pitcher"][frame_idx]:
                bbox = tracks["pitcher"][frame_idx]
                pose = self.estimate_pose_in_bbox(frame, bbox)
                pitcher_canvas = self.draw_skeleton(pitcher_canvas, pose, self.COLORS["pitcher"], 3)
                overlay_canvas = self.draw_skeleton(overlay_canvas, pose, self.COLORS["pitcher"], 2)

            # Process batter
            if "batter" in tracks and tracks["batter"][frame_idx]:
                bbox = tracks["batter"][frame_idx]
                pose = self.estimate_pose_in_bbox(frame, bbox)
                batter_canvas = self.draw_skeleton(batter_canvas, pose, self.COLORS["batter"], 3)
                overlay_canvas = self.draw_skeleton(overlay_canvas, pose, self.COLORS["batter"], 2)

            # Process glove
            if "glove" in tracks and tracks["glove"][frame_idx]:
                bbox = tracks["glove"][frame_idx]
                glove_canvas = self.draw_glove_marker(glove_canvas, bbox, self.COLORS["glove"])
                overlay_canvas = self.draw_glove_marker(overlay_canvas, bbox, self.COLORS["glove"])

            # Write frames
            writers["pitcher"].write(pitcher_canvas)
            writers["batter"].write(batter_canvas)
            writers["glove"].write(glove_canvas)
            writers["overlay"].write(overlay_canvas)

            frame_idx += 1

            # Progress indicator every 100 frames
            if frame_idx % 100 == 0:
                logger.info(f"    Frame {frame_idx}/{total_frames}")

        # Cleanup
        cap.release()
        for w in writers.values():
            w.release()

        return {k: str(v) for k, v in output_paths.items()}

    def run(self, max_videos: int = 1):
        """Run the full pipeline."""
        logger.info("=" * 50)
        logger.info("Baseball Video Tracking Pipeline")
        logger.info("=" * 50)

        # Load labels
        labels, frames_info = self.load_labels()

        # Group by unique source videos
        video_labels = {}
        for info in frames_info:
            video_path = info.get("source_video")
            if video_path and Path(video_path).exists():
                frame_name = Path(info["frame_path"]).name
                if frame_name in labels and labels[frame_name]:
                    video_labels[video_path] = {
                        "frame_idx": info["source_frame_idx"],
                        "labels": labels[frame_name],
                        "total_frames": info["total_frames"],
                    }

        logger.info(f"Found {len(video_labels)} videos with labels")

        # Process videos
        results = []
        for i, (video_path, label_data) in enumerate(video_labels.items()):
            if i >= max_videos:
                break

            video_name = Path(video_path).stem
            output_paths = self.process_video(video_path, label_data, video_name)

            if output_paths:
                results.append({
                    "source": video_path,
                    "outputs": output_paths,
                })

                logger.info(f"\nGenerated videos for {video_name}:")
                for name, path in output_paths.items():
                    logger.info(f"  {name}: {path}")

        logger.info("\n" + "=" * 50)
        logger.info(f"Pipeline complete. Processed {len(results)} video(s).")
        logger.info("=" * 50)

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate tracking videos")
    parser.add_argument("--project-dir", default="F:/Claude_Projects/baseball-biomechanics")
    parser.add_argument("--max-videos", type=int, default=1, help="Max videos to process")
    parser.add_argument("--cache-dir", default="F:/hf_cache")
    args = parser.parse_args()

    pipeline = VideoTrackingPipeline(args.project_dir, args.cache_dir)
    results = pipeline.run(max_videos=args.max_videos)

    return results


if __name__ == "__main__":
    main()
