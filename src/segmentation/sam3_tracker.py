"""SAM 3 segmentation and tracking for baseball players."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PlayerSegmentationResult:
    """
    Result of player segmentation for a single frame.

    Attributes:
        frame_number: Frame index in the video.
        player_role: Role of the player (pitcher/batter/catcher).
        mask: Binary segmentation mask.
        bbox: Bounding box (x, y, width, height).
        confidence: Segmentation confidence score.
        text_prompt: Text prompt used for segmentation.
    """
    frame_number: int
    player_role: str
    mask: np.ndarray
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    confidence: float
    text_prompt: str
    cropped_frame: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "frame_number": self.frame_number,
            "player_role": self.player_role,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
            "confidence": self.confidence,
            "text_prompt": self.text_prompt,
        }


@dataclass
class VideoSegmentationResult:
    """
    Complete segmentation results for a video.

    Attributes:
        video_path: Path to the source video.
        fps: Frames per second of the video.
        total_frames: Total number of frames processed.
        player_results: Dictionary mapping player role to list of frame results.
    """
    video_path: str
    fps: float
    total_frames: int
    player_results: Dict[str, List[PlayerSegmentationResult]] = field(default_factory=dict)


class SAM3Tracker:
    """
    SAM 3 based player segmentation and tracking.

    Uses text prompts to identify and segment baseball players
    (pitcher, batter, catcher) and tracks them through video frames.
    """

    DEFAULT_PROMPTS = {
        "pitcher": "pitcher on mound throwing baseball",
        "batter": "batter at plate with bat",
        "catcher": "catcher in crouch behind plate",
    }

    def __init__(
        self,
        model_name: str = "sam2.1_b.pt",
        output_dir: str = "data/masks",
        mask_format: str = "png",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the SAM 3 tracker.

        Args:
            model_name: SAM model name (sam3, sam2, etc.).
            output_dir: Directory to save mask outputs.
            mask_format: Format for saving masks (png, rle).
            confidence_threshold: Minimum confidence for valid segmentation.
            device: Device to run model on (cuda, cpu, or auto).
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mask_format = mask_format
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._prompts = self.DEFAULT_PROMPTS.copy()

    def load_model(self) -> None:
        """
        Load the SAM 3 model.

        Imports ultralytics and loads the model on first use.
        """
        if self.model is not None:
            return

        try:
            from ultralytics import SAM

            logger.info(f"Loading SAM model: {self.model_name}")
            self.model = SAM(self.model_name)
            logger.info("SAM model loaded successfully")

        except ImportError:
            raise ImportError(
                "ultralytics package is required for SAM 3. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise

    def set_custom_prompts(self, prompts: Dict[str, str]) -> None:
        """
        Set custom text prompts for player segmentation.

        Args:
            prompts: Dictionary mapping role to text prompt.
        """
        self._prompts.update(prompts)
        logger.debug(f"Updated prompts: {self._prompts}")

    def get_prompt(self, role: str) -> str:
        """Get the text prompt for a player role."""
        return self._prompts.get(role, self.DEFAULT_PROMPTS.get(role, role))

    def segment_all_players(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        use_yolo: bool = True,
    ) -> Dict[str, Optional[PlayerSegmentationResult]]:
        """
        Segment all players (pitcher, batter, catcher) in a single frame.

        Uses YOLO for person detection, then classifies by position:
        - Pitcher: Center of frame, lower half (on mound)
        - Batter: Left-center of frame (at plate, from camera's view)
        - Catcher: Center-right of frame (behind plate)

        Args:
            frame: Input frame (BGR format from OpenCV).
            frame_number: Frame index for result tracking.
            use_yolo: If True, use YOLO for person detection (recommended).

        Returns:
            Dict mapping role -> PlayerSegmentationResult (or None if not found)
        """
        self.load_model()

        results_dict: Dict[str, Optional[PlayerSegmentationResult]] = {
            "pitcher": None,
            "batter": None,
            "catcher": None,
        }

        h_frame, w_frame = frame.shape[:2]

        try:
            if use_yolo:
                # Use YOLO for reliable person detection
                candidates = self._detect_persons_yolo(frame)
            else:
                # Fallback to SAM automatic (less reliable)
                candidates = self._detect_persons_sam(frame)

            if not candidates:
                logger.debug(f"No valid player candidates in frame {frame_number}")
                return results_dict

            # Filter out fans in stands (upper portion of frame)
            field_candidates = [c for c in candidates if c["cy_norm"] >= 0.20]

            if not field_candidates:
                return results_dict

            assigned = set()

            # Find pitcher - center-lower of frame (on mound)
            # Pitcher is the person LOWEST in frame (largest cy_norm) in center area
            # Must be lower than umpire/catcher area (cy >= 0.50)
            pitcher_candidates = [c for c in field_candidates
                                 if 0.30 <= c["cx_norm"] <= 0.70
                                 and c["cy_norm"] >= 0.50]
            if pitcher_candidates:
                # Take the one lowest in frame (largest cy) - that's the pitcher on mound
                pitcher_candidates.sort(key=lambda c: c["cy_norm"], reverse=True)
                results_dict["pitcher"] = self._create_result_from_bbox(
                    frame, pitcher_candidates[0], "pitcher", frame_number
                )
                assigned.add(pitcher_candidates[0]["idx"])

            # Find batter - left-center of frame
            batter_candidates = [c for c in field_candidates
                                if c["idx"] not in assigned
                                and c["cx_norm"] <= 0.50
                                and c["cy_norm"] >= 0.20]
            if batter_candidates:
                # Take the largest one
                batter_candidates.sort(key=lambda c: c["area"], reverse=True)
                results_dict["batter"] = self._create_result_from_bbox(
                    frame, batter_candidates[0], "batter", frame_number
                )
                assigned.add(batter_candidates[0]["idx"])

            # Find catcher - center-right, not the pitcher
            catcher_candidates = [c for c in field_candidates
                                 if c["idx"] not in assigned
                                 and c["cx_norm"] >= 0.40
                                 and c["cy_norm"] >= 0.25]
            if catcher_candidates:
                # Take the largest one
                catcher_candidates.sort(key=lambda c: c["area"], reverse=True)
                results_dict["catcher"] = self._create_result_from_bbox(
                    frame, catcher_candidates[0], "catcher", frame_number
                )

            return results_dict

        except Exception as e:
            logger.error(f"Multi-player segmentation error: {e}")
            return results_dict

    def _detect_persons_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons using YOLO."""
        try:
            from ultralytics import YOLO

            # Load YOLO model (cached after first load)
            if not hasattr(self, "_yolo_model"):
                self._yolo_model = YOLO("yolov8n.pt")

            h_frame, w_frame = frame.shape[:2]
            results = self._yolo_model(frame, classes=[0], verbose=False)  # class 0 = person

            candidates = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    w = x2 - x1
                    h = y2 - y1
                    cx_norm = (x1 + x2) / 2 / w_frame
                    cy_norm = (y1 + y2) / 2 / h_frame
                    area = w * h

                    candidates.append({
                        "idx": i,
                        "bbox": (float(x1), float(y1), float(w), float(h)),
                        "cx_norm": cx_norm,
                        "cy_norm": cy_norm,
                        "area": area,
                        "confidence": conf,
                    })

            return candidates

        except ImportError:
            logger.warning("YOLO not available, falling back to SAM")
            return self._detect_persons_sam(frame)

    def _detect_persons_sam(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons using SAM automatic segmentation (fallback)."""
        results = self.model(frame)

        if not results or len(results) == 0:
            return []

        result = results[0]
        if not hasattr(result, "masks") or result.masks is None:
            return []

        masks = result.masks.data.cpu().numpy()
        if len(masks) == 0:
            return []

        confs = None
        if hasattr(result, "boxes") and result.boxes is not None:
            if hasattr(result.boxes, "conf"):
                confs = result.boxes.conf.cpu().numpy()

        h_frame, w_frame = frame.shape[:2]
        min_area = h_frame * w_frame * 0.005  # 0.5% of frame

        candidates = []
        for i, mask in enumerate(masks):
            bbox = self._mask_to_bbox(mask)
            x, y, w, h = bbox

            if w < 40 or h < 60:
                continue

            area = w * h
            if area < min_area:
                continue

            aspect_ratio = h / max(w, 1)
            if aspect_ratio < 0.6:
                continue

            cx_norm = (x + w / 2) / w_frame
            cy_norm = (y + h / 2) / h_frame
            confidence = confs[i] if confs is not None and i < len(confs) else 0.5

            candidates.append({
                "idx": i,
                "mask": mask,
                "bbox": bbox,
                "cx_norm": cx_norm,
                "cy_norm": cy_norm,
                "area": area,
                "confidence": confidence,
            })

        return candidates

    def _create_result_from_bbox(
        self,
        frame: np.ndarray,
        candidate: Dict[str, Any],
        role: str,
        frame_number: int,
    ) -> PlayerSegmentationResult:
        """Create a PlayerSegmentationResult from a YOLO/SAM detection."""
        x, y, w, h = candidate["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Add padding
        padding = 20
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_frame, x + w + padding)
        y2 = min(h_frame, y + h + padding)

        cropped = frame[y1:y2, x1:x2].copy()

        # Get mask if available (from SAM), otherwise create from bbox
        mask = candidate.get("mask")
        if mask is None:
            # Create simple bbox mask
            mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 1

        return PlayerSegmentationResult(
            frame_number=frame_number,
            player_role=role,
            mask=mask,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            confidence=candidate["confidence"],
            text_prompt=f"yolo-detected:{role}",
            cropped_frame=cropped,
        )

    def _create_result(
        self,
        frame: np.ndarray,
        candidate: Dict[str, Any],
        role: str,
        frame_number: int,
    ) -> PlayerSegmentationResult:
        """Create a PlayerSegmentationResult from a candidate mask."""
        x, y, w, h = candidate["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Add padding
        padding = 15
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_frame, x + w + padding)
        y2 = min(h_frame, y + h + padding)

        cropped = frame[y1:y2, x1:x2].copy()

        return PlayerSegmentationResult(
            frame_number=frame_number,
            player_role=role,
            mask=candidate["mask"],
            bbox=(x1, y1, x2 - x1, y2 - y1),  # Updated bbox with padding
            confidence=candidate["confidence"],
            text_prompt=f"position-based:{role}",
            cropped_frame=cropped,
        )

    def segment_video_all_players(
        self,
        video_path: str,
        show_progress: bool = True,
    ) -> Dict[str, List[PlayerSegmentationResult]]:
        """
        Segment all players (pitcher, batter, catcher) across all frames.

        Args:
            video_path: Path to the video file.
            show_progress: Whether to show progress bar.

        Returns:
            Dict mapping role -> list of PlayerSegmentationResult per frame
        """
        self.load_model()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Segmenting all players in: {video_path.name} "
            f"({total_frames} frames, {fps:.1f} fps)"
        )

        results: Dict[str, List[PlayerSegmentationResult]] = {
            "pitcher": [],
            "batter": [],
            "catcher": [],
        }

        frame_iter = range(total_frames)
        if show_progress:
            frame_iter = tqdm(frame_iter, desc="Segmenting players")

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Segment all players in this frame
            frame_results = self.segment_all_players(frame, frame_number)

            # Append results for each role
            for role in ["pitcher", "batter", "catcher"]:
                if frame_results[role] is not None:
                    results[role].append(frame_results[role])

            frame_number += 1
            if show_progress and hasattr(frame_iter, '__iter__'):
                pass  # tqdm updates automatically

        cap.release()

        # Log summary
        for role, role_results in results.items():
            logger.info(
                f"Segmented {len(role_results)}/{total_frames} frames for {role}"
            )

        return results

    def segment_frame(
        self,
        frame: np.ndarray,
        text_prompt: str,
        player_role: str,
        frame_number: int = 0,
    ) -> Optional[PlayerSegmentationResult]:
        """
        Segment a single frame using automatic detection.

        Note: SAM 2.1 via Ultralytics doesn't support text prompts directly.
        Instead, we use automatic segmentation and select the largest person-like
        region based on aspect ratio heuristics.

        Args:
            frame: Input frame (BGR format from OpenCV).
            text_prompt: Text prompt (used for logging, not for segmentation).
            player_role: Role identifier (pitcher/batter/catcher).
            frame_number: Frame index for result tracking.

        Returns:
            PlayerSegmentationResult if successful, None otherwise.
        """
        self.load_model()

        try:
            # Run SAM with automatic segmentation (no prompts)
            # This returns all detected segments in the frame
            results = self.model(frame)

            if not results or len(results) == 0:
                logger.debug(f"No segmentation results for frame {frame_number}")
                return None

            result = results[0]

            # Extract masks
            if not hasattr(result, "masks") or result.masks is None:
                return None

            masks = result.masks.data.cpu().numpy()
            if len(masks) == 0:
                return None

            # Get boxes for aspect ratio filtering
            boxes = None
            confs = None
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else None

            # Find the best person-like mask based on:
            # 1. Size (larger is better for main subjects)
            # 2. Aspect ratio (height > width for standing person)
            # 3. Position (center-ish of frame for baseball)
            best_idx = None
            best_score = 0
            h_frame, w_frame = frame.shape[:2]

            for i, mask in enumerate(masks):
                bbox = self._mask_to_bbox(mask)
                x, y, w, h = bbox

                if w < 20 or h < 20:  # Too small
                    continue

                # Score based on size
                area = w * h
                size_score = area / (w_frame * h_frame)

                # Score based on aspect ratio (prefer tall shapes for people)
                aspect_ratio = h / max(w, 1)
                aspect_score = min(aspect_ratio / 2.0, 1.0)  # Normalize

                # Score based on position (prefer center)
                cx = x + w / 2
                cy = y + h / 2
                center_dist = ((cx - w_frame/2)**2 + (cy - h_frame/2)**2) ** 0.5
                center_score = 1.0 - min(center_dist / (w_frame/2), 1.0)

                # Confidence score if available
                conf_score = confs[i] if confs is not None and i < len(confs) else 0.5

                # Combined score
                score = (size_score * 0.3 + aspect_score * 0.3 +
                        center_score * 0.2 + conf_score * 0.2)

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                return None

            mask = masks[best_idx]
            bbox = self._mask_to_bbox(mask)
            confidence = confs[best_idx] if confs is not None else best_score

            if confidence < self.confidence_threshold:
                logger.debug(f"Low confidence ({confidence:.2f}) for frame {frame_number}")
                return None

            # Crop the frame using the bounding box
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            cropped = frame[y:y+h, x:x+w].copy() if w > 0 and h > 0 else None

            return PlayerSegmentationResult(
                frame_number=frame_number,
                player_role=player_role,
                mask=mask,
                bbox=bbox,
                confidence=confidence,
                text_prompt=text_prompt,
                cropped_frame=cropped,
            )

        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            return None

    def segment_frame_with_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        player_role: str,
        frame_number: int = 0,
    ) -> Optional[PlayerSegmentationResult]:
        """
        Segment a frame using an exemplar bounding box.

        Args:
            frame: Input frame (BGR format).
            bbox: Bounding box (x1, y1, x2, y2) around the player.
            player_role: Role identifier.
            frame_number: Frame index.

        Returns:
            PlayerSegmentationResult if successful, None otherwise.
        """
        self.load_model()

        try:
            # Run SAM with bounding box prompt
            x1, y1, w, h = bbox
            box_xyxy = [[x1, y1, x1 + w, y1 + h]]
            results = self.model(frame, bboxes=box_xyxy)

            if not results or len(results) == 0:
                return None

            result = results[0]

            if hasattr(result, "masks") and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                if len(masks) == 0:
                    return None

                mask = masks[0]

                # Get confidence
                if hasattr(result, "boxes") and result.boxes is not None:
                    conf = result.boxes.conf.cpu().numpy()
                    confidence = float(conf[0]) if len(conf) > 0 else 1.0
                else:
                    confidence = 1.0

                # Crop frame
                x, y, w, h = int(x1), int(y1), int(w), int(h)
                cropped = frame[y:y+h, x:x+w].copy() if w > 0 and h > 0 else None

                return PlayerSegmentationResult(
                    frame_number=frame_number,
                    player_role=player_role,
                    mask=mask,
                    bbox=bbox,
                    confidence=confidence,
                    text_prompt=f"bbox:{bbox}",
                    cropped_frame=cropped,
                )

            return None

        except Exception as e:
            logger.error(f"Bbox segmentation error: {e}")
            return None

    def segment_video(
        self,
        video_path: str,
        roles: Optional[List[str]] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
        show_progress: bool = True,
        save_masks: bool = True,
    ) -> VideoSegmentationResult:
        """
        Segment all players in a video.

        Args:
            video_path: Path to the video file.
            roles: List of player roles to segment (default: all).
            custom_prompts: Custom prompts for specific roles.
            show_progress: Whether to show progress bar.
            save_masks: Whether to save mask files.

        Returns:
            VideoSegmentationResult containing all frame results.
        """
        self.load_model()

        if roles is None:
            roles = ["pitcher", "batter", "catcher"]

        if custom_prompts:
            self.set_custom_prompts(custom_prompts)

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Processing video: {video_path.name} ({total_frames} frames, {fps:.1f} fps)"
        )

        result = VideoSegmentationResult(
            video_path=str(video_path),
            fps=fps,
            total_frames=total_frames,
        )

        for role in roles:
            result.player_results[role] = []

        # Create output directory for this video
        if save_masks:
            video_mask_dir = self.output_dir / video_path.stem
            video_mask_dir.mkdir(parents=True, exist_ok=True)

        frame_iter = range(total_frames)
        if show_progress:
            frame_iter = tqdm(frame_iter, desc="Segmenting frames")

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for role in roles:
                prompt = self.get_prompt(role)
                seg_result = self.segment_frame(
                    frame, prompt, role, frame_number
                )

                if seg_result:
                    result.player_results[role].append(seg_result)

                    # Save mask if requested
                    if save_masks:
                        mask_path = (
                            video_mask_dir / f"{role}_frame_{frame_number:06d}.png"
                        )
                        self._save_mask(seg_result.mask, mask_path)
                        logger.debug(f"Saved mask: {mask_path}")

            frame_number += 1
            if show_progress:
                frame_iter.update(1) if hasattr(frame_iter, 'update') else None

        cap.release()

        # Log summary
        for role, results_list in result.player_results.items():
            logger.info(
                f"Segmented {len(results_list)}/{total_frames} frames for {role}"
            )

        return result

    def track_player_through_video(
        self,
        video_path: str,
        initial_bbox: Tuple[float, float, float, float],
        player_role: str,
        show_progress: bool = True,
    ) -> List[PlayerSegmentationResult]:
        """
        Track a player through video using initial bounding box.

        Uses SAM's tracking capability to follow the player
        through subsequent frames after initial detection.

        Args:
            video_path: Path to the video file.
            initial_bbox: Initial bounding box (x, y, width, height).
            player_role: Role of the player being tracked.
            show_progress: Whether to show progress bar.

        Returns:
            List of segmentation results for each frame.
        """
        self.load_model()

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []

        # Process first frame with initial bbox
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return results

        current_bbox = initial_bbox
        frame_iter = range(total_frames)
        if show_progress:
            frame_iter = tqdm(frame_iter, desc=f"Tracking {player_role}")

        frame_number = 0
        while ret:
            seg_result = self.segment_frame_with_bbox(
                frame, current_bbox, player_role, frame_number
            )

            if seg_result:
                results.append(seg_result)
                # Update bbox for next frame
                current_bbox = seg_result.bbox

            ret, frame = cap.read()
            frame_number += 1

        cap.release()
        logger.info(f"Tracked {len(results)}/{total_frames} frames for {player_role}")

        return results

    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculate bounding box from binary mask.

        Args:
            mask: Binary mask array.

        Returns:
            Bounding box (x, y, width, height).
        """
        if mask.sum() == 0:
            return (0.0, 0.0, 0.0, 0.0)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (
            float(x_min),
            float(y_min),
            float(x_max - x_min),
            float(y_max - y_min),
        )

    def _save_mask(self, mask: np.ndarray, path: Path) -> None:
        """
        Save a segmentation mask to file.

        Args:
            mask: Binary mask array.
            path: Output file path.
        """
        # Convert to uint8 image
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(path), mask_img)

    def load_mask(self, path: str) -> np.ndarray:
        """
        Load a saved mask from file.

        Args:
            path: Path to the mask file.

        Returns:
            Binary mask as numpy array.
        """
        mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (mask_img > 127).astype(np.uint8)

    def apply_mask_to_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Apply a mask to a frame, removing background.

        Args:
            frame: Input frame (BGR).
            mask: Binary mask.
            background_color: Color for masked out regions.

        Returns:
            Masked frame.
        """
        # Resize mask if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Create 3-channel mask
        mask_3ch = np.stack([mask] * 3, axis=-1)

        # Apply mask
        result = np.where(mask_3ch, frame, background_color)
        return result.astype(np.uint8)

    def get_cropped_frames(
        self,
        video_path: str,
        segmentation_results: List[PlayerSegmentationResult],
        padding: int = 20,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract cropped frames based on segmentation results.

        Args:
            video_path: Path to the video file.
            segmentation_results: List of segmentation results.
            padding: Padding around bounding box.

        Returns:
            List of (frame_number, cropped_frame) tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Build frame number to result mapping
        result_map = {r.frame_number: r for r in segmentation_results}

        cropped_frames = []
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number in result_map:
                seg_result = result_map[frame_number]
                x, y, w, h = seg_result.bbox

                # Apply padding
                h_frame, w_frame = frame.shape[:2]
                x1 = max(0, int(x) - padding)
                y1 = max(0, int(y) - padding)
                x2 = min(w_frame, int(x + w) + padding)
                y2 = min(h_frame, int(y + h) + padding)

                cropped = frame[y1:y2, x1:x2].copy()
                cropped_frames.append((frame_number, cropped))

            frame_number += 1

        cap.release()
        return cropped_frames
