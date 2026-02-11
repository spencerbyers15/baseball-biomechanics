"""Player detection and pose estimation using YOLO + RTMPose.

Detects players by role (pitcher, batter) using YOLO person detection
with an EfficientNet-B0 pitcher classifier (preferred) or spatial heuristics
(fallback), then extracts 17-landmark poses via RTMPose-X (GPU-accelerated
ONNX through rtmlib).

Supports per-stadium calibrated pitcher zones (from pitcher_zones.json)
as fallback when no classifier model is available.
"""

import json
import logging
import math
import time
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
    the pitcher (and eventually batter), then runs RTMPose-X on the
    cropped region for accurate 17-landmark pose estimation (GPU).

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
        pitcher_classifier_path: Optional[str] = None,
        use_pitcher_classifier: bool = True,
        detect_interval: int = 1,
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
            pitcher_classifier_path: Path to pitcher classifier model. If None,
                looks for models/pitcher_classifier/best.pt relative to project root.
            use_pitcher_classifier: If True (default), try to load the pitcher
                classifier. Falls back to zone heuristics if model not found.
            detect_interval: Run YOLO+classifier every N frames (default 1 = every
                frame). Intermediate frames get linearly interpolated bboxes between
                the previous and next keyframe detections. Set to 3 to skip detection
                on 2/3 of frames. Call flush_buffer() after the last video frame.
        """
        self.yolo_model_name = yolo_model
        self.person_conf = person_conf
        self.crop_padding = crop_padding
        self.min_crop_w = min_crop_w
        self.min_crop_h = min_crop_h
        self.detect_interval = detect_interval

        self._yolo = None
        self._pose_backend = None
        self._pitcher_classifier = None
        self._use_pitcher_classifier = use_pitcher_classifier
        self._pitcher_classifier_path = pitcher_classifier_path

        # Calibrated zone support
        self._zones: Dict[str, dict] = {}
        self._active_zone: Optional[dict] = None
        self._stadium = stadium

        # Temporal smoothing state
        self._prev_pitcher_cx: Optional[float] = None
        self._prev_pitcher_cy: Optional[float] = None
        self._prev_bbox: Optional[Tuple[float, float, float, float]] = None
        self._frames_without_pitcher: int = 0
        self._temporal_reset_threshold: int = 10  # Reset after N consecutive misses
        self._temporal_radius: float = 0.08  # Primary match radius
        self._max_jump_radius: float = 0.15  # Max allowed jump before rejection
        self._bbox_ema_alpha: float = 0.7  # EMA blend: 0.7 new + 0.3 old

        # Bbox interpolation buffer (used when detect_interval > 1)
        self._frame_buffer: List[Tuple[np.ndarray, int]] = []  # (frame, frame_number)
        self._last_keyframe_bbox: Optional[Tuple[float, float, float, float]] = None

        # Load zones
        self._load_zones(pitcher_zones_path)
        if stadium:
            self.set_stadium(stadium)

        # Try to load pitcher classifier
        if use_pitcher_classifier:
            self._load_pitcher_classifier(pitcher_classifier_path)

    def _load_pitcher_classifier(self, path: Optional[str] = None):
        """Try to load the pitcher classifier model.

        Falls back silently to zone heuristics if model file not found.
        """
        if path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            path = str(project_root / "models" / "pitcher_classifier" / "best.pt")

        classifier_path = Path(path)
        if not classifier_path.exists():
            logger.debug(f"No pitcher classifier at {path}, using zone heuristics")
            return

        try:
            from src.filtering.pitcher_classifier import PitcherClassifier
            self._pitcher_classifier = PitcherClassifier(
                model_path=str(classifier_path),
                device="cuda",
            )
            self._pitcher_classifier.initialize()
            logger.info("Pitcher classifier loaded — using classifier-based selection")
        except Exception as e:
            logger.warning(f"Failed to load pitcher classifier: {e}")
            self._pitcher_classifier = None

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
        self._prev_bbox = None
        self._frames_without_pitcher = 0
        self._frame_buffer.clear()
        self._last_keyframe_bbox = None

    def _load_yolo(self):
        """Lazy-load YOLO model on GPU."""
        if self._yolo is not None:
            return
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {self.yolo_model_name}")
        self._yolo = YOLO(self.yolo_model_name)
        self._yolo.to("cuda")

    def _load_pose(self):
        """Lazy-load RTMPose-X pose backend (GPU)."""
        if self._pose_backend is not None:
            return
        from src.pose.rtmpose_backend import RTMPoseBackend
        self._pose_backend = RTMPoseBackend(
            min_detection_confidence=0.3,
            device="cuda",
        )
        self._pose_backend.initialize()
        logger.info("RTMPose-X pose backend initialized")

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
        self, candidates: List[Dict[str, Any]], frame: np.ndarray = None,
    ) -> Optional[Dict[str, Any]]:
        """Select the pitcher from detected persons.

        Priority: classifier (if available) > calibrated zone > default heuristic.
        """
        if self._pitcher_classifier is not None and frame is not None:
            result = self._find_pitcher_classified(candidates, frame)
            if result is not None:
                return result
            # Classifier found no pitcher — fall through to heuristics

        if self._active_zone:
            return self._find_pitcher_calibrated(candidates)
        return self._find_pitcher_default(candidates)

    def _find_pitcher_classified(
        self, candidates: List[Dict[str, Any]], frame: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Select pitcher using the trained EfficientNet-B0 classifier.

        Crops each candidate, classifies all crops in one batch, and picks
        the highest-confidence pitcher prediction.
        """
        # Crop each candidate
        crops = []
        valid_indices = []
        for i, c in enumerate(candidates):
            crop_result = self._crop_person(frame, c["bbox"])
            if crop_result is not None:
                crops.append(crop_result[0])  # just the image
                valid_indices.append(i)

        if not crops:
            return None

        # Classify all crops in one batch
        results = self._pitcher_classifier.classify_crops_batch(crops)

        # Pick highest-confidence pitcher
        best = None
        best_conf = 0.0
        for idx, (label, conf) in zip(valid_indices, results):
            if label == "pitcher" and conf > best_conf:
                best = candidates[idx]
                best_conf = conf

        if best is not None:
            logger.debug(f"Classifier picked pitcher with conf={best_conf:.3f}")

        return best

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

    def _centroid_distance(self, candidate: Dict[str, Any]) -> float:
        """Normalized distance from candidate centroid to previous pitcher position."""
        dx = candidate["cx_norm"] - self._prev_pitcher_cx
        dy = candidate["cy_norm"] - self._prev_pitcher_cy
        return math.sqrt(dx * dx + dy * dy)

    def _update_temporal(self, pick: Dict[str, Any]):
        """Update temporal tracking state with accepted candidate."""
        self._prev_pitcher_cx = pick["cx_norm"]
        self._prev_pitcher_cy = pick["cy_norm"]
        self._frames_without_pitcher = 0

    def _handle_miss(self):
        """Handle a frame where no pitcher was accepted."""
        self._frames_without_pitcher += 1
        if self._frames_without_pitcher >= self._temporal_reset_threshold:
            self.reset_temporal()

    def _find_pitcher_with_temporal(
        self, candidates: List[Dict[str, Any]], frame: np.ndarray = None,
    ) -> Optional[Dict[str, Any]]:
        """Select pitcher with temporal smoothing and jump rejection.

        If a pitcher was found in the previous frame, strongly prefer
        a candidate near the same position (temporal continuity).
        Rejects candidates that jump too far from the previous position
        to prevent the bounding box from snapping to a different person.
        Falls back to standard _find_pitcher if no prior position.
        """
        # Try spatial selection first (classifier or zone)
        spatial_pick = self._find_pitcher(candidates, frame)

        # If no previous position, just use spatial result
        if self._prev_pitcher_cx is None:
            if spatial_pick:
                self._update_temporal(spatial_pick)
            else:
                self._handle_miss()
            return spatial_pick

        # We have a previous pitcher position — enforce temporal continuity
        # Filter fans in stands
        field = [c for c in candidates if c["cy_norm"] >= 0.20]

        # Look for any candidate near the previous position (tight radius)
        temporal_candidates = []
        for c in field:
            dist = self._centroid_distance(c)
            if dist < self._temporal_radius:
                c["_temporal_dist"] = dist
                temporal_candidates.append(c)

        if temporal_candidates:
            # Pick closest to previous position
            temporal_candidates.sort(key=lambda c: c["_temporal_dist"])
            pick = temporal_candidates[0]
            self._update_temporal(pick)
            return pick

        # No temporal match — check if spatial pick is within max jump radius
        if spatial_pick:
            dist = self._centroid_distance(spatial_pick)
            if dist < self._max_jump_radius:
                self._update_temporal(spatial_pick)
                return spatial_pick
            else:
                # Spatial pick is too far — likely a different person, reject
                logger.debug(
                    f"Rejected pitcher jump: dist={dist:.3f} > max={self._max_jump_radius}"
                )
                self._handle_miss()
                return None

        self._handle_miss()
        return None

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
        """Run pose estimation on crop and transform coordinates back to frame space."""
        self._load_pose()

        pose_result = self._pose_backend.process_frame(crop, frame_number)

        # Transform keypoint coordinates from crop-local to frame-global
        for kp in pose_result.keypoints:
            kp.x += crop_x1
            kp.y += crop_y1

        return pose_result

    @staticmethod
    def _interpolate_bbox(
        bbox_a: Tuple[float, float, float, float],
        bbox_b: Tuple[float, float, float, float],
        t: float,
    ) -> Tuple[float, float, float, float]:
        """Linearly interpolate between two bboxes. t in [0, 1]."""
        return tuple(a + t * (b - a) for a, b in zip(bbox_a, bbox_b))

    def _detect_frame_full(
        self,
        frame: np.ndarray,
        frame_number: int,
        use_temporal: bool,
        t0: float,
    ) -> Dict[str, Any]:
        """Run full YOLO + classifier + pose pipeline on a single frame.

        This is the core detection logic, called for every frame when
        detect_interval=1, or only on keyframes when detect_interval>1.
        """
        result = {"pitcher": None, "timing": {}}

        t1_start = time.perf_counter()
        candidates = self._detect_persons(frame)
        t1 = time.perf_counter()
        result["timing"]["yolo_detect_ms"] = (t1 - t1_start) * 1000

        if not candidates:
            if use_temporal:
                self._frames_without_pitcher += 1
                if self._frames_without_pitcher >= self._temporal_reset_threshold:
                    self.reset_temporal()
            result["timing"]["total_ms"] = (time.perf_counter() - t0) * 1000
            return result

        t2 = time.perf_counter()
        if use_temporal:
            pitcher = self._find_pitcher_with_temporal(candidates, frame)
        else:
            pitcher = self._find_pitcher(candidates, frame)
        t3 = time.perf_counter()
        result["timing"]["find_pitcher_ms"] = (t3 - t2) * 1000

        if pitcher is None:
            result["timing"]["total_ms"] = (time.perf_counter() - t0) * 1000
            return result

        # EMA-smooth the bounding box to prevent jitter
        raw_bbox = pitcher["bbox"]
        if use_temporal and self._prev_bbox is not None:
            a = self._bbox_ema_alpha
            smoothed = tuple(
                a * new + (1 - a) * old
                for new, old in zip(raw_bbox, self._prev_bbox)
            )
            self._prev_bbox = smoothed
        else:
            smoothed = raw_bbox
            self._prev_bbox = raw_bbox

        crop_result = self._crop_person(frame, smoothed)
        if crop_result is None:
            result["timing"]["total_ms"] = (time.perf_counter() - t0) * 1000
            return result

        crop, cx1, cy1 = crop_result

        t4 = time.perf_counter()
        pose = self._run_pose_on_crop(crop, cx1, cy1, frame_number)
        t5 = time.perf_counter()
        result["timing"]["pose_ms"] = (t5 - t4) * 1000
        result["timing"]["total_ms"] = (t5 - t0) * 1000

        result["pitcher"] = {
            "pose": pose,
            "bbox": smoothed,
            "conf": pitcher["conf"],
        }

        return result

    def _process_buffered_frames(
        self, target_bbox: Optional[Tuple[float, float, float, float]],
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Process buffered intermediate frames with linearly interpolated bboxes.

        Interpolates between _last_keyframe_bbox and target_bbox, runs pose on
        each buffered frame with the interpolated bbox.

        Args:
            target_bbox: The bbox detected on the current keyframe.

        Returns:
            List of (frame_number, result_dict) tuples for each buffered frame.
        """
        if not self._frame_buffer or self._last_keyframe_bbox is None:
            return []

        prev_bbox = self._last_keyframe_bbox
        # If current keyframe has no detection, extrapolate with previous bbox
        end_bbox = target_bbox if target_bbox is not None else prev_bbox
        n_buffered = len(self._frame_buffer)
        total_steps = n_buffered + 1  # intervals from prev keyframe to current

        deferred = []
        for i, (buf_frame, buf_frame_num) in enumerate(self._frame_buffer):
            t = (i + 1) / total_steps
            interp_bbox = self._interpolate_bbox(prev_bbox, end_bbox, t)

            buf_result = {"pitcher": None, "timing": {}}
            tb0 = time.perf_counter()
            crop_result = self._crop_person(buf_frame, interp_bbox)
            if crop_result is not None:
                crop, cx1, cy1 = crop_result
                tb1 = time.perf_counter()
                pose = self._run_pose_on_crop(crop, cx1, cy1, buf_frame_num)
                tb2 = time.perf_counter()
                buf_result["pitcher"] = {
                    "pose": pose,
                    "bbox": interp_bbox,
                    "conf": -1.0,
                }
                buf_result["timing"] = {
                    "yolo_detect_ms": 0.0,
                    "find_pitcher_ms": 0.0,
                    "pose_ms": (tb2 - tb1) * 1000,
                    "total_ms": (tb2 - tb0) * 1000,
                    "skipped_detect": True,
                    "interpolated": True,
                }
            deferred.append((buf_frame_num, buf_result))

        return deferred

    def flush_buffer(self) -> List[Tuple[int, Dict[str, Any]]]:
        """Process any remaining buffered frames at end of video.

        Uses the last keyframe bbox (no interpolation target available).
        Call this after the last frame of a video to ensure all frames are processed.

        Returns:
            List of (frame_number, result_dict) tuples for each buffered frame.
        """
        if not self._frame_buffer:
            return []

        # No next keyframe — use last keyframe bbox for all remaining frames
        deferred = self._process_buffered_frames(self._last_keyframe_bbox)
        self._frame_buffer.clear()
        return deferred

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        use_temporal: bool = True,
    ) -> Dict[str, Any]:
        """Detect pitcher and extract pose from a single frame.

        When detect_interval > 1, intermediate frames are buffered and processed
        with linearly interpolated bboxes when the next keyframe arrives. The
        result dict includes a 'deferred_results' key containing a list of
        (frame_number, result_dict) tuples for the processed buffered frames.

        Call flush_buffer() after the last frame to process remaining buffered frames.

        Args:
            frame: BGR frame from OpenCV.
            frame_number: Frame index.
            use_temporal: If True, use temporal smoothing for pitcher selection.

        Returns:
            Dict with keys:
                - 'pitcher': result dict or None (pose, bbox, conf)
                - 'timing': timing breakdown dict
                - '_buffered': True if frame was buffered (intermediate frame)
                - 'deferred_results': list of (frame_number, result) for processed
                  buffered frames (only present on keyframe results)
        """
        t0 = time.perf_counter()

        # No subsampling — process every frame directly
        if self.detect_interval <= 1:
            return self._detect_frame_full(frame, frame_number, use_temporal, t0)

        is_keyframe = (frame_number % self.detect_interval == 0)

        if not is_keyframe:
            # Intermediate frame: buffer for later interpolation
            self._frame_buffer.append((frame.copy(), frame_number))
            return {"pitcher": None, "timing": {}, "_buffered": True}

        # --- Keyframe: run full detection ---
        keyframe_result = self._detect_frame_full(
            frame, frame_number, use_temporal, t0,
        )

        # Get the keyframe's detected bbox
        current_bbox = None
        if keyframe_result["pitcher"] is not None:
            current_bbox = keyframe_result["pitcher"]["bbox"]

        # Process buffered intermediate frames with interpolated bboxes
        deferred = self._process_buffered_frames(current_bbox)
        self._frame_buffer.clear()

        # Update last keyframe bbox for next interval
        if current_bbox is not None:
            self._last_keyframe_bbox = current_bbox

        keyframe_result["deferred_results"] = deferred
        return keyframe_result

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
        self._pitcher_classifier = None
        self.reset_temporal()
