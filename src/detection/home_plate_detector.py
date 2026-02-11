"""Home plate detection using SAM3 with text prompts."""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class HomePlateDetection:
    """Home plate detection result."""
    frame_number: int
    centroid: Tuple[float, float]
    corners: List[Tuple[int, int]]  # Polygon corners
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    confidence: float
    white_ratio: float
    mask: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "centroid_x": self.centroid[0],
            "centroid_y": self.centroid[1],
            "corners": self.corners,
            "bbox_x": self.bbox[0],
            "bbox_y": self.bbox[1],
            "bbox_w": self.bbox[2],
            "bbox_h": self.bbox[3],
            "confidence": self.confidence,
            "white_ratio": self.white_ratio,
        }


class HomePlateDetector:
    """Detect home plate using SAM3 with text prompt 'home plate'."""

    def __init__(
        self,
        cache_dir: str = "F:/hf_cache",
        text_prompt: str = "home plate",
        confidence_threshold: float = 0.3,
    ):
        self.cache_dir = cache_dir
        self.text_prompt = text_prompt
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False

    def initialize(self) -> None:
        """Load the SAM3 model with text encoder."""
        if self._initialized:
            return

        from transformers import Sam3Model, Sam3Processor

        logger.info("Loading SAM3 with text encoder...")
        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir
        )
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir, torch_dtype=torch.bfloat16
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"SAM3 loaded on {self.device}")
        self._initialized = True

    def _extract_corners(self, contour: np.ndarray) -> List[Tuple[int, int]]:
        """Extract corner points from contour."""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return [[int(p[0][0]), int(p[0][1])] for p in approx]

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
    ) -> Optional[HomePlateDetection]:
        """Detect home plate in a single frame using SAM3 text prompt."""
        if not self._initialized:
            self.initialize()

        h, w = frame.shape[:2]

        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Run SAM3 with text prompt
        inputs = self.processor(
            images=pil_image,
            text=self.text_prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get masks
        target_sizes = inputs.get("original_sizes")
        if target_sizes is not None:
            target_sizes = target_sizes.tolist()
        else:
            target_sizes = [(h, w)]

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.1,
            mask_threshold=0.5,
            target_sizes=target_sizes
        )[0]

        masks = results.get("masks", [])
        if not hasattr(masks, '__len__') or len(masks) == 0:
            return None

        # Find best detection
        best_detection = None
        best_score = 0

        for mask, box, score in zip(
            results["masks"], results["boxes"], results["scores"]
        ):
            # Convert tensors to numpy
            if hasattr(box, 'cpu'):
                box = box.float().cpu().numpy()
            if hasattr(score, 'cpu'):
                score = float(score.float().cpu())
            if hasattr(mask, 'cpu'):
                mask = mask.float().cpu().numpy()

            if score < self.confidence_threshold:
                continue

            if score > best_score:
                best_score = score
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Extract contour for corners
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                corners = []
                bw, bh = x2 - x1, y2 - y1
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    corners = self._extract_corners(largest)
                    x, y, bw, bh = cv2.boundingRect(largest)

                best_detection = HomePlateDetection(
                    frame_number=frame_number,
                    centroid=(float(cx), float(cy)),
                    corners=corners,
                    bbox=(float(x1), float(y1), float(bw), float(bh)),
                    confidence=score,
                    white_ratio=score,  # Using score as proxy
                    mask=mask.astype(np.uint8),
                )

        return best_detection

    def detect_video_first_frame(
        self,
        video_path: str,
        sample_frames: int = 5,
    ) -> Optional[HomePlateDetection]:
        """Detect home plate from video (tries first N frames)."""
        if not self._initialized:
            self.initialize()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        best_detection = None

        for i in range(sample_frames):
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.detect_frame(frame, i)
            if detection:
                if best_detection is None or detection.confidence > best_detection.confidence:
                    best_detection = detection

        cap.release()
        return best_detection

    def visualize(
        self,
        frame: np.ndarray,
        detection: Optional[HomePlateDetection],
    ) -> np.ndarray:
        """Draw detection on frame."""
        vis = frame.copy()

        if detection is not None:
            # Overlay mask
            if detection.mask is not None:
                mask_color = np.zeros_like(frame)
                mask_color[:, :, 1] = (detection.mask * 255).astype(np.uint8)
                vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

            # Draw corners
            if detection.corners:
                corners = np.array(detection.corners, dtype=np.int32)
                cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
                for corner in detection.corners:
                    cv2.circle(vis, tuple(corner), 4, (0, 0, 255), -1)

            # Draw centroid
            cx, cy = detection.centroid
            cv2.circle(vis, (int(cx), int(cy)), 6, (255, 0, 255), -1)

            # Info
            info = f"Home Plate: {detection.confidence:.2f}"
            cv2.putText(vis, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No home plate detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return vis

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
