"""Player detection and pose estimation using YOLO + MediaPipe.

Detects players by role (pitcher, batter) using YOLO person detection
with spatial heuristics, then extracts 33-landmark poses via MediaPipe.

Supports per-stadium calibrated pitcher zones (from pitcher_zones.json)
for improved detection, with distance-based scoring and temporal smoothing.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.pose.base import PoseResult

logger = logging.getLogger(__name__)

# Default hard-coded pitcher zone (used when no calibration data available)
DEFAULT_PITCHER_ZONE = {
    "cx_min": 0.30,
    "cx_max": 0.70,
    "cy_min": 0.50,
    "cy_max": 1.00,
    "center_cx": 0.50,
    "center_cy": 0.65,
}


class PlayerPoseDetector:
    """Detect players by role and extract poses.

    Uses YOLOv8n for person detection with spatial heuristics to identify
    the pitcher (and eventually batter), then runs MediaPipe Pose on the
    cropped region for accurate 33-landmark pose estimation.

    Supports per-stadium calibrated zones loaded from pitcher_zones.json
    for improved pitcher identification. Falls back to hard-coded defaults
    when no calibration data is available.
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
        pitcher_zones_path: Optional[str] = None,
        stadium: Optional[str] = None,
    ):
        """Initialize the detector.

        Args:
            yolo_model: YOLO model name/path for person detection.
            person_conf: Minimum confidence for YOLO person detections.
            crop_padding: Pixels of padding around person bbox for pose crop.
            min_crop_w: Minimum crop width to attempt pose estimation.
            min_crop_h: Minimum crop height to attempt pose estimation.
            pitcher_zones_path: Path to pitcher_zones.json. If None, looks
                for data/pitcher_zones.json relative to project root.
            stadium: Stadium name for zone lookup (e.g. "Dodger Stadium"
                or "Dodger_Stadium"). If None, uses default hard-coded zone.
        """
        self.yolo_model_name = yolo_model
        self.person_conf = person_conf
        self.crop_padding = crop_padding
        self.min_crop_w = min_crop_w
        self.min_crop_h = min_crop_h

        self._yolo = None
        self._pose_backend = None

        # Calibrated zone support
        self._zones: Dict[str, dict] = {}
        self._active_zone: Optional[dict] = None
        self._stadium = stadium

        # Temporal smoothing state
        self._prev_pitcher_cx: Optional[float] = None
        self._prev_pitcher_cy: Optional[float] = None
        self._frames_without_pitcher: int = 0
        self._temporal_reset_threshold: int = 10  # Reset after N consecutive misses

        # Load zones
        self._load_zones(pitcher_zones_path)
        if stadium:
            self.set_stadium(stadium)

    def _load_zones(self, path: Optional[str] = None):
        """Load pitcher zones from JSON file."""
        if path is None:
            # Try default location relative to this file
            project_root = Path(__file__).resolve().parent.parent.parent
            path = str(project_root / "data" / "pitcher_zones.json")

        zones_path = Path(path)
        if zones_path.exists():
            try:
                with open(zones_path, "r") as f:
                    self._zones = json.load(f)
                logger.info(f"Loaded pitcher zones for {len(self._zones)} stadiums")
            except Exception as e:
                logger.warning(f"Failed to load pitcher zones: {e}")
                self._zones = {}
        else:
            logger.debug(f"No pitcher zones file at {path}")

    def set_stadium(self, stadium: str):
        """Set the active stadium for zone lookup.

        Args:
            stadium: Stadium name (spaces or underscores, e.g. "Dodger Stadium").
        """
        self._stadium = stadium
        stadium_key = stadium.replace(" ", "_")

        if stadium_key in self._zones:
            zone_data = self._zones[stadium_key]
            self._active_zone = zone_data
            logger.info(
                f"Using calibrated zone for {stadium}: "
                f"cx={zone_data['mean_cx']:.3f}±{zone_data['std_cx']:.3f}, "
                f"cy={zone_data['mean_cy']:.3f}±{zone_data['std_cy']:.3f}"
            )
        else:
            self._active_zone = None
            logger.info(f"No calibrated zone for {stadium}, using defaults")

        # Reset temporal state when switching stadiums
        self.reset_temporal()

    def reset_temporal(self):
        """Reset temporal smoothing state."""
        self._prev_pitcher_cx = None
        self._prev_pitcher_cy = None
        self._frames_without_pitcher = 0

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
            idx, bbox (x1,y1,x2,y2), cx_norm, cy_norm, bbox_w_norm, bbox_h_norm, area, conf
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
                bbox_w_norm = (x2 - x1) / w
                bbox_h_norm = (y2 - y1) / h
                area = (x2 - x1) * (y2 - y1)

                candidates.append({
                    "idx": i,
                    "bbox": (x1, y1, x2, y2),
                    "cx_norm": cx_norm,
                    "cy_norm": cy_norm,
                    "bbox_w_norm": bbox_w_norm,
                    "bbox_h_norm": bbox_h_norm,
                    "area": area,
                    "conf": conf,
                })

        return candidates

    def _find_pitcher(
        self, candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the pitcher from detected persons.

        Uses calibrated zone + distance scoring if available,
        otherwise falls back to the original hard-coded heuristic.
        """
        if self._active_zone:
            return self._find_pitcher_calibrated(candidates)
        return self._find_pitcher_default(candidates)

    def _find_pitcher_default(
        self, candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Original hard-coded pitcher heuristic (backward compatible).

        - Horizontally centered: cx_norm between 0.30 and 0.70
        - Lower half of frame: cy_norm >= 0.50
        - Select the person lowest in frame (max cy_norm) = on the mound
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

    def _find_pitcher_calibrated(
        self, candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select pitcher using calibrated zone + distance-based scoring.

        Uses the per-stadium zone center and std dev to define a search region,
        then scores candidates by distance to the zone center. Also filters
        by bbox size to reject distant fielders or close-up artifacts.
        """
        zone = self._active_zone

        mean_cx = zone["mean_cx"]
        mean_cy = zone["mean_cy"]
        std_cx = zone["std_cx"]
        std_cy = zone["std_cy"]

        # Define search region: mean ± 2.5 * std (generous)
        cx_min = mean_cx - 2.5 * std_cx
        cx_max = mean_cx + 2.5 * std_cx
        cy_min = mean_cy - 2.5 * std_cy
        cy_max = min(1.0, mean_cy + 2.5 * std_cy)

        # Ensure minimum zone size
        if cx_max - cx_min < 0.20:
            cx_min = mean_cx - 0.10
            cx_max = mean_cx + 0.10
        if cy_max - cy_min < 0.15:
            cy_min = mean_cy - 0.075
            cy_max = mean_cy + 0.075

        # Filter fans in stands
        field = [c for c in candidates if c["cy_norm"] >= 0.20]

        # Filter to calibrated zone
        in_zone = [
            c for c in field
            if cx_min <= c["cx_norm"] <= cx_max
            and cy_min <= c["cy_norm"] <= cy_max
        ]

        if not in_zone:
            return None

        # Bbox size filtering: reject candidates with bboxes much smaller or
        # larger than expected pitcher size
        mean_bbox_w = zone.get("mean_bbox_w", 0.08)
        mean_bbox_h = zone.get("mean_bbox_h", 0.22)
        std_bbox_w = zone.get("std_bbox_w", 0.03)
        std_bbox_h = zone.get("std_bbox_h", 0.08)

        filtered = []
        for c in in_zone:
            bw = c.get("bbox_w_norm", 0)
            bh = c.get("bbox_h_norm", 0)

            # Reject if bbox is way too small (distant fielder) or too large
            if bw > 0 and bh > 0:
                w_dev = abs(bw - mean_bbox_w) / max(std_bbox_w, 0.01)
                h_dev = abs(bh - mean_bbox_h) / max(std_bbox_h, 0.01)
                # Allow up to 3 std devs from mean bbox size
                if w_dev > 3.0 and h_dev > 3.0:
                    continue
            filtered.append(c)

        if not filtered:
            # Fall back to zone-filtered without bbox size check
            filtered = in_zone

        # Score by distance to zone center (closer = better)
        # Normalize distances by std so that dimensions are comparable
        for c in filtered:
            dx = (c["cx_norm"] - mean_cx) / max(std_cx, 0.01)
            dy = (c["cy_norm"] - mean_cy) / max(std_cy, 0.01)
            c["_distance"] = math.sqrt(dx * dx + dy * dy)

        filtered.sort(key=lambda c: c["_distance"])
        return filtered[0]

    def _find_pitcher_with_temporal(
        self, candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select pitcher with temporal smoothing.

        If a pitcher was found in the previous frame, strongly prefer
        a candidate near the same position (temporal continuity).
        Falls back to standard _find_pitcher if no prior position.
        """
        # Try spatial selection first
        spatial_pick = self._find_pitcher(candidates)

        # If no previous position, just use spatial result
        if self._prev_pitcher_cx is None:
            if spatial_pick:
                self._prev_pitcher_cx = spatial_pick["cx_norm"]
                self._prev_pitcher_cy = spatial_pick["cy_norm"]
                self._frames_without_pitcher = 0
            else:
                self._frames_without_pitcher += 1
                if self._frames_without_pitcher >= self._temporal_reset_threshold:
                    self.reset_temporal()
            return spatial_pick

        # We have a previous pitcher position — check for temporal continuity
        # Look for any candidate near the previous position
        temporal_radius = 0.08  # Normalized distance threshold

        # Filter fans in stands
        field = [c for c in candidates if c["cy_norm"] >= 0.20]

        temporal_candidates = []
        for c in field:
            dx = c["cx_norm"] - self._prev_pitcher_cx
            dy = c["cy_norm"] - self._prev_pitcher_cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < temporal_radius:
                c["_temporal_dist"] = dist
                temporal_candidates.append(c)

        if temporal_candidates:
            # Pick closest to previous position
            temporal_candidates.sort(key=lambda c: c["_temporal_dist"])
            pick = temporal_candidates[0]
            self._prev_pitcher_cx = pick["cx_norm"]
            self._prev_pitcher_cy = pick["cy_norm"]
            self._frames_without_pitcher = 0
            return pick

        # No temporal match — use spatial pick and update position
        if spatial_pick:
            self._prev_pitcher_cx = spatial_pick["cx_norm"]
            self._prev_pitcher_cy = spatial_pick["cy_norm"]
            self._frames_without_pitcher = 0
        else:
            self._frames_without_pitcher += 1
            if self._frames_without_pitcher >= self._temporal_reset_threshold:
                self.reset_temporal()

        return spatial_pick

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
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        use_temporal: bool = True,
    ) -> Dict[str, Any]:
        """Detect pitcher and extract pose from a single frame.

        Args:
            frame: BGR frame from OpenCV.
            frame_number: Frame index.
            use_temporal: If True, use temporal smoothing for pitcher selection.

        Returns:
            Dict with 'pitcher' key mapping to result dict or None.
            Result dict has keys: pose (PoseResult), bbox (x1,y1,x2,y2), conf (float).
        """
        result = {"pitcher": None}

        candidates = self._detect_persons(frame)
        if not candidates:
            if use_temporal:
                self._frames_without_pitcher += 1
                if self._frames_without_pitcher >= self._temporal_reset_threshold:
                    self.reset_temporal()
            return result

        if use_temporal:
            pitcher = self._find_pitcher_with_temporal(candidates)
        else:
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
        self.reset_temporal()
