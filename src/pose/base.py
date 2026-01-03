"""Abstract base class for pose estimation backends."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KeypointData:
    """
    Data for a single keypoint/landmark.

    Attributes:
        name: Name of the keypoint (e.g., "left_shoulder").
        x: X coordinate (normalized 0-1 or pixel value).
        y: Y coordinate (normalized 0-1 or pixel value).
        z: Z coordinate (depth, optional for 2D models).
        confidence: Detection confidence (0-1).
        is_occluded: Whether the keypoint is occluded/estimated.
    """
    name: str
    x: float
    y: float
    z: Optional[float] = None
    confidence: float = 1.0
    is_occluded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for database storage."""
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "is_occluded": self.is_occluded,
        }

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z, confidence]."""
        return np.array([
            self.x,
            self.y,
            self.z if self.z is not None else 0.0,
            self.confidence,
        ])


@dataclass
class PoseResult:
    """
    Result of pose estimation for a single frame.

    Attributes:
        frame_number: Frame index in the video.
        timestamp_ms: Timestamp in milliseconds.
        keypoints: List of detected keypoints.
        is_valid: Whether pose detection was successful.
        model_name: Name of the model that produced this result.
        raw_output: Optional raw model output for debugging.
    """
    frame_number: int
    timestamp_ms: float
    keypoints: List[KeypointData] = field(default_factory=list)
    is_valid: bool = True
    model_name: str = ""
    raw_output: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "is_valid": self.is_valid,
            "model_name": self.model_name,
        }

    def get_keypoint(self, name: str) -> Optional[KeypointData]:
        """Get a keypoint by name."""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None

    def get_keypoints_array(self) -> np.ndarray:
        """
        Get all keypoints as a numpy array.

        Returns:
            Array of shape (N, 4) with [x, y, z, confidence] for each keypoint.
        """
        if not self.keypoints:
            return np.array([])
        return np.stack([kp.to_array() for kp in self.keypoints])

    def get_bone_vector(
        self,
        start_keypoint: str,
        end_keypoint: str,
    ) -> Optional[np.ndarray]:
        """
        Calculate vector between two keypoints (bone direction).

        Args:
            start_keypoint: Name of the start keypoint.
            end_keypoint: Name of the end keypoint.

        Returns:
            3D vector from start to end, or None if keypoints not found.
        """
        start = self.get_keypoint(start_keypoint)
        end = self.get_keypoint(end_keypoint)

        if start is None or end is None:
            return None

        return np.array([
            end.x - start.x,
            end.y - start.y,
            (end.z or 0) - (start.z or 0),
        ])


class PoseBackend(ABC):
    """
    Abstract base class for pose estimation backends.

    All pose estimation implementations should inherit from this class
    and implement the required abstract methods.
    """

    # Standard keypoint names that all backends should map to
    STANDARD_KEYPOINTS = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the pose backend.

        Args:
            min_detection_confidence: Minimum confidence for pose detection.
            min_tracking_confidence: Minimum confidence for pose tracking.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._is_initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the pose estimation backend."""
        pass

    @property
    @abstractmethod
    def keypoint_names(self) -> List[str]:
        """List of keypoint names produced by this backend."""
        pass

    @property
    def supports_3d(self) -> bool:
        """Whether this backend supports 3D pose estimation."""
        return False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the pose estimation model.

        This should be called before processing any frames.
        """
        pass

    @abstractmethod
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
        pass

    def process_video(
        self,
        video_path: str,
        show_progress: bool = True,
    ) -> List[PoseResult]:
        """
        Process all frames in a video.

        Args:
            video_path: Path to the video file.
            show_progress: Whether to show progress bar.

        Returns:
            List of PoseResult for each frame.
        """
        import cv2
        from tqdm import tqdm

        if not self._is_initialized:
            self.initialize()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results = []
        frame_iter = range(total_frames)
        if show_progress:
            frame_iter = tqdm(frame_iter, desc=f"Pose estimation ({self.name})")

        frame_number = 0
        for _ in frame_iter:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_number / fps) * 1000.0
            result = self.process_frame(frame, frame_number, timestamp_ms)
            results.append(result)
            frame_number += 1

        cap.release()
        logger.info(f"Processed {len(results)} frames with {self.name}")

        return results

    def process_frames(
        self,
        frames: List[Tuple[int, np.ndarray]],
        fps: float = 30.0,
        show_progress: bool = True,
    ) -> List[PoseResult]:
        """
        Process a list of frames.

        Args:
            frames: List of (frame_number, frame) tuples.
            fps: Frames per second for timestamp calculation.
            show_progress: Whether to show progress bar.

        Returns:
            List of PoseResult for each frame.
        """
        from tqdm import tqdm

        if not self._is_initialized:
            self.initialize()

        results = []
        frame_iter = frames
        if show_progress:
            frame_iter = tqdm(frames, desc=f"Pose estimation ({self.name})")

        for frame_number, frame in frame_iter:
            timestamp_ms = (frame_number / fps) * 1000.0
            result = self.process_frame(frame, frame_number, timestamp_ms)
            results.append(result)

        return results

    def cleanup(self) -> None:
        """
        Clean up resources.

        Override this method to release any resources held by the backend.
        """
        self._is_initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def map_to_standard_keypoints(
        self,
        keypoints: List[KeypointData],
    ) -> List[KeypointData]:
        """
        Map backend-specific keypoints to standard keypoint names.

        Override this method to provide custom mapping.

        Args:
            keypoints: List of backend-specific keypoints.

        Returns:
            List of keypoints with standardized names.
        """
        # Default implementation: return as-is
        return keypoints

    def calculate_joint_angle(
        self,
        pose_result: PoseResult,
        point_a: str,
        point_b: str,
        point_c: str,
    ) -> Optional[float]:
        """
        Calculate the angle at point_b formed by points A-B-C.

        Args:
            pose_result: Pose estimation result.
            point_a: First point name.
            point_b: Center point name (vertex of angle).
            point_c: Third point name.

        Returns:
            Angle in degrees, or None if points not found.
        """
        a = pose_result.get_keypoint(point_a)
        b = pose_result.get_keypoint(point_b)
        c = pose_result.get_keypoint(point_c)

        if any(p is None for p in [a, b, c]):
            return None

        # Calculate vectors
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])

        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))

        return float(angle)

    def calculate_biomechanics(
        self,
        pose_result: PoseResult,
    ) -> Dict[str, float]:
        """
        Calculate common biomechanical metrics from pose.

        Args:
            pose_result: Pose estimation result.

        Returns:
            Dictionary of biomechanical measurements.
        """
        metrics = {}

        # Elbow angles
        left_elbow_angle = self.calculate_joint_angle(
            pose_result, "left_shoulder", "left_elbow", "left_wrist"
        )
        right_elbow_angle = self.calculate_joint_angle(
            pose_result, "right_shoulder", "right_elbow", "right_wrist"
        )

        if left_elbow_angle is not None:
            metrics["left_elbow_angle"] = left_elbow_angle
        if right_elbow_angle is not None:
            metrics["right_elbow_angle"] = right_elbow_angle

        # Knee angles
        left_knee_angle = self.calculate_joint_angle(
            pose_result, "left_hip", "left_knee", "left_ankle"
        )
        right_knee_angle = self.calculate_joint_angle(
            pose_result, "right_hip", "right_knee", "right_ankle"
        )

        if left_knee_angle is not None:
            metrics["left_knee_angle"] = left_knee_angle
        if right_knee_angle is not None:
            metrics["right_knee_angle"] = right_knee_angle

        # Shoulder angles
        left_shoulder_angle = self.calculate_joint_angle(
            pose_result, "left_elbow", "left_shoulder", "left_hip"
        )
        right_shoulder_angle = self.calculate_joint_angle(
            pose_result, "right_elbow", "right_shoulder", "right_hip"
        )

        if left_shoulder_angle is not None:
            metrics["left_shoulder_angle"] = left_shoulder_angle
        if right_shoulder_angle is not None:
            metrics["right_shoulder_angle"] = right_shoulder_angle

        # Hip angles
        left_hip_angle = self.calculate_joint_angle(
            pose_result, "left_shoulder", "left_hip", "left_knee"
        )
        right_hip_angle = self.calculate_joint_angle(
            pose_result, "right_shoulder", "right_hip", "right_knee"
        )

        if left_hip_angle is not None:
            metrics["left_hip_angle"] = left_hip_angle
        if right_hip_angle is not None:
            metrics["right_hip_angle"] = right_hip_angle

        return metrics
