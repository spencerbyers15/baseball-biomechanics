"""MediaPipe Pose estimation backend using the new Tasks API."""

import logging
from typing import List, Optional

import numpy as np

from src.pose.base import KeypointData, PoseBackend, PoseResult

logger = logging.getLogger(__name__)


class MediaPipeBackend(PoseBackend):
    """
    MediaPipe Pose estimation backend using the Tasks API.

    Uses Google's MediaPipe Pose Landmarker which provides 33 body landmarks.
    """

    # MediaPipe Pose Landmarker landmark names (33 landmarks)
    MEDIAPIPE_LANDMARKS = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        num_poses: int = 1,
    ):
        """
        Initialize the MediaPipe backend.

        Args:
            min_detection_confidence: Minimum confidence for pose detection.
            min_tracking_confidence: Minimum confidence for pose tracking.
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy).
            num_poses: Maximum number of poses to detect.
        """
        super().__init__(min_detection_confidence, min_tracking_confidence)
        self.model_complexity = model_complexity
        self.num_poses = num_poses
        self._landmarker = None
        self._mp = None

    @property
    def name(self) -> str:
        """Name of the pose estimation backend."""
        return "mediapipe"

    @property
    def keypoint_names(self) -> List[str]:
        """List of keypoint names produced by this backend."""
        return self.MEDIAPIPE_LANDMARKS.copy()

    @property
    def supports_3d(self) -> bool:
        """MediaPipe supports 3D pose estimation."""
        return True

    def initialize(self) -> None:
        """Initialize the MediaPipe Pose Landmarker."""
        if self._is_initialized:
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self._mp = mp

            # Get model path - download if needed
            model_path = self._get_model_path()

            # Create pose landmarker options
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=self.num_poses,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
            )

            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._is_initialized = True
            logger.info("MediaPipe Pose Landmarker initialized successfully")

        except ImportError as e:
            raise ImportError(
                f"mediapipe package error: {e}. "
                "Install with: pip install mediapipe"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise

    def _get_model_path(self) -> str:
        """Download and return path to the pose landmarker model."""
        import os
        import urllib.request
        from pathlib import Path

        # Model URLs for different complexities
        model_urls = {
            0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        }

        model_names = {
            0: "pose_landmarker_lite.task",
            1: "pose_landmarker_full.task",
            2: "pose_landmarker_heavy.task",
        }

        # Get cache directory
        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = model_names.get(self.model_complexity, model_names[1])
        model_path = cache_dir / model_name

        if not model_path.exists():
            url = model_urls.get(self.model_complexity, model_urls[1])
            logger.info(f"Downloading MediaPipe model from {url}...")
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model downloaded to {model_path}")

        return str(model_path)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
    ) -> PoseResult:
        """
        Process a single frame and return pose estimation results.

        Args:
            frame: Input frame (BGR format from OpenCV).
            frame_number: Frame index for result tracking.
            timestamp_ms: Timestamp in milliseconds.

        Returns:
            PoseResult containing detected keypoints.
        """
        if not self._is_initialized:
            self.initialize()

        import cv2

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Detect poses
        detection_result = self._landmarker.detect(mp_image)

        # Create PoseResult
        pose_result = PoseResult(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            model_name=self.name,
            raw_output=detection_result,
        )

        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            pose_result.is_valid = False
            logger.debug(f"No pose detected in frame {frame_number}")
            return pose_result

        # Get first detected pose
        landmarks = detection_result.pose_landmarks[0]
        world_landmarks = detection_result.pose_world_landmarks[0] if detection_result.pose_world_landmarks else None

        h, w = frame.shape[:2]

        for idx, landmark in enumerate(landmarks):
            # Get world coordinates if available
            z_coord = None
            if world_landmarks and idx < len(world_landmarks):
                z_coord = world_landmarks[idx].z

            keypoint = KeypointData(
                name=self.MEDIAPIPE_LANDMARKS[idx],
                x=landmark.x * w,  # Convert to pixel coordinates
                y=landmark.y * h,
                z=z_coord,
                confidence=landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                is_occluded=(landmark.visibility < self.min_detection_confidence) if hasattr(landmark, 'visibility') else False,
            )
            pose_result.keypoints.append(keypoint)

        return pose_result

    def draw_pose(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        draw_landmarks: bool = True,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Draw pose landmarks on a frame.

        Args:
            frame: Input frame (BGR format).
            pose_result: PoseResult to visualize.
            draw_landmarks: Whether to draw landmark points.
            draw_connections: Whether to draw skeleton connections.

        Returns:
            Frame with pose visualization drawn.
        """
        import cv2

        output = frame.copy()

        if not pose_result.is_valid or not pose_result.keypoints:
            return output

        # Define connections (skeleton)
        connections = [
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
        ]

        # Build keypoint map
        kp_map = {kp.name: (int(kp.x), int(kp.y)) for kp in pose_result.keypoints}

        # Draw connections
        if draw_connections:
            for start_name, end_name in connections:
                if start_name in kp_map and end_name in kp_map:
                    cv2.line(output, kp_map[start_name], kp_map[end_name], (0, 255, 0), 2)

        # Draw landmarks
        if draw_landmarks:
            for kp in pose_result.keypoints:
                x, y = int(kp.x), int(kp.y)
                color = (0, 0, 255) if kp.is_occluded else (255, 0, 0)
                cv2.circle(output, (x, y), 4, color, -1)

        return output

    def cleanup(self) -> None:
        """Clean up MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._is_initialized = False
        logger.debug("MediaPipe Pose Landmarker resources cleaned up")
