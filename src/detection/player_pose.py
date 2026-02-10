"""Player detection and pose estimation using YOLO + MediaPipe.

Detects players by role (pitcher, batter) using YOLO person detection
with spatial heuristics, then extracts 33-landmark poses via MediaPipe.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.pose.base import PoseResult

logger = logging.getLogger(__name__)


class PlayerPoseDetector:
    """Detect players by role and extract poses.

    Uses YOLOv8n for person detection with spatial heuristics to identify
    the pitcher (and eventually batter), then runs MediaPipe Pose on the
    cropped region for accurate 33-landmark pose estimation.
    """

    # Skeleton connections for visualization (orange)
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
    ]

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        person_conf: float = 0.3,
        crop_padding: int = 40,
        min_crop_w: int = 50,
        min_crop_h: int = 30,
    ):
        """Initialize the detector.

        Args:
            yolo_model: YOLO model name/path for person detection.
            person_conf: Minimum confidence for YOLO person detections.
            crop_padding: Pixels of padding around person bbox for pose crop.
            min_crop_w: Minimum crop width to attempt pose estimation.
            min_crop_h: Minimum crop height to attempt pose estimation.
        """
        self.yolo_model_name = yolo_model
        self.person_conf = person_conf
        self.crop_padding = crop_padding
        self.min_crop_w = min_crop_w
        self.min_crop_h = min_crop_h

        self._yolo = None
        self._pose_backend = None

    def _load_yolo(self):
        """Lazy-load YOLO model."""
        if self._yolo is not None:
            return
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {self.yolo_model_name}")
        self._yolo = YOLO(self.yolo_model_name)

    def _load_pose(self):
        """Lazy-load MediaPipe pose backend."""
        if self._pose_backend is not None:
            return
        from src.pose.mediapipe_backend import MediaPipeBackend
        self._pose_backend = MediaPipeBackend(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            num_poses=1,
        )
        self._pose_backend.initialize()
        logger.info("MediaPipe pose backend initialized")

    def _detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO person detection and compute normalized coords.

        Returns list of candidate dicts with keys:
            idx, bbox (x1,y1,x2,y2), cx_norm, cy_norm, area, conf
        """
        self._load_yolo()
        h, w = frame.shape[:2]

        results = self._yolo(frame, classes=[0], conf=self.person_conf, verbose=False)
        candidates = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                conf = float(box.conf[0].cpu().numpy())
                cx_norm = (x1 + x2) / 2 / w
                cy_norm = (y1 + y2) / 2 / h
                area = (x2 - x1) * (y2 - y1)

                candidates.append({
                    "idx": i,
                    "bbox": (x1, y1, x2, y2),
                    "cx_norm": cx_norm,
                    "cy_norm": cy_norm,
                    "area": area,
                    "conf": conf,
                })

        return candidates

    def _find_pitcher(
        self, candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the pitcher from detected persons using spatial heuristics.

        Pitcher heuristic (for standard center-field camera angle):
        - Horizontally centered: cx_norm between 0.30 and 0.70
        - Lower half of frame: cy_norm >= 0.50
        - Select the person lowest in frame (max cy_norm) = on the mound
        - Filter out fans in upper stands: cy_norm < 0.20
        """
        # Filter fans in stands
        field = [c for c in candidates if c["cy_norm"] >= 0.20]

        # Pitcher zone: center, lower half
        pitcher_zone = [
            c for c in field
            if 0.30 <= c["cx_norm"] <= 0.70
            and c["cy_norm"] >= 0.50
        ]

        if not pitcher_zone:
            return None

        # Lowest person in zone = on mound
        pitcher_zone.sort(key=lambda c: c["cy_norm"], reverse=True)
        return pitcher_zone[0]

    def _crop_person(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """Crop person from frame with padding.

        Returns (crop, crop_x1, crop_y1) or None if crop too small.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad = self.crop_padding

        cx1 = max(0, int(x1) - pad)
        cy1 = max(0, int(y1) - pad)
        cx2 = min(w, int(x2) + pad)
        cy2 = min(h, int(y2) + pad)

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        if crop_w < self.min_crop_w or crop_h < self.min_crop_h:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        return crop, cx1, cy1

    def _run_pose_on_crop(
        self,
        crop: np.ndarray,
        crop_x1: int,
        crop_y1: int,
        frame_number: int,
    ) -> PoseResult:
        """Run MediaPipe on crop and transform coordinates back to frame space."""
        self._load_pose()

        pose_result = self._pose_backend.process_frame(crop, frame_number)

        # Transform keypoint coordinates from crop-local to frame-global
        for kp in pose_result.keypoints:
            kp.x += crop_x1
            kp.y += crop_y1

        return pose_result

    def detect_frame(
        self, frame: np.ndarray, frame_number: int = 0
    ) -> Dict[str, Any]:
        """Detect pitcher and extract pose from a single frame.

        Args:
            frame: BGR frame from OpenCV.
            frame_number: Frame index.

        Returns:
            Dict with 'pitcher' key mapping to result dict or None.
            Result dict has keys: pose (PoseResult), bbox (x1,y1,x2,y2), conf (float).
        """
        result = {"pitcher": None}

        candidates = self._detect_persons(frame)
        if not candidates:
            return result

        pitcher = self._find_pitcher(candidates)
        if pitcher is None:
            return result

        crop_result = self._crop_person(frame, pitcher["bbox"])
        if crop_result is None:
            return result

        crop, cx1, cy1 = crop_result
        pose = self._run_pose_on_crop(crop, cx1, cy1, frame_number)

        result["pitcher"] = {
            "pose": pose,
            "bbox": pitcher["bbox"],
            "conf": pitcher["conf"],
        }

        return result

    def visualize(
        self,
        frame: np.ndarray,
        results: Dict[str, Any],
        color: Tuple[int, int, int] = (0, 140, 255),  # orange in BGR
    ) -> np.ndarray:
        """Draw pitcher detection + pose skeleton on frame.

        Args:
            frame: BGR frame.
            results: Output from detect_frame().
            color: BGR color for skeleton and label.

        Returns:
            Annotated frame copy.
        """
        output = frame.copy()

        pitcher = results.get("pitcher")
        if pitcher is None:
            return output

        bbox = pitcher["bbox"]
        conf = pitcher["conf"]
        pose = pitcher["pose"]

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Subtle bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)

        # Label
        label = f"PITCHER {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            output, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        # Draw skeleton if pose is valid
        if pose and pose.is_valid and pose.keypoints:
            kp_map = {}
            for kp in pose.keypoints:
                if kp.confidence > 0.3:
                    kp_map[kp.name] = (int(kp.x), int(kp.y))

            # Draw connections
            for start_name, end_name in self.SKELETON_CONNECTIONS:
                if start_name in kp_map and end_name in kp_map:
                    cv2.line(output, kp_map[start_name], kp_map[end_name], color, 2)

            # Draw keypoints
            for name, pt in kp_map.items():
                cv2.circle(output, pt, 3, (255, 255, 255), -1)
                cv2.circle(output, pt, 3, color, 1)

        return output

    def cleanup(self):
        """Release resources."""
        if self._pose_backend is not None:
            self._pose_backend.cleanup()
            self._pose_backend = None
        self._yolo = None
