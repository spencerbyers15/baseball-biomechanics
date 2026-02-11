"""Baseball detection using YOLO-World with text prompts."""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from ultralytics import YOLOWorld

logger = logging.getLogger(__name__)


@dataclass
class BallDetection:
    """Single baseball detection result."""
    frame_number: int
    timestamp_ms: float
    centroid: Tuple[float, float]  # (x, y)
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    confidence: float

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "x": self.centroid[0],
            "y": self.centroid[1],
            "bbox_x": self.bbox[0],
            "bbox_y": self.bbox[1],
            "bbox_w": self.bbox[2],
            "bbox_h": self.bbox[3],
            "confidence": self.confidence,
        }


class BaseballDetector:
    """Detect baseballs in video frames using YOLO-World."""

    def __init__(
        self,
        model_name: str = "yolov8s-world.pt",
        confidence_threshold: float = 0.1,
        classes: List[str] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.classes = classes or ["baseball", "ball"]
        self.model = None
        self._initialized = False

    def initialize(self) -> None:
        """Load the YOLO-World model."""
        if self._initialized:
            return

        logger.info(f"Loading YOLO-World model: {self.model_name}")
        self.model = YOLOWorld(self.model_name)
        self.model.set_classes(self.classes)
        self._initialized = True
        logger.info(f"Model initialized with classes: {self.classes}")

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
    ) -> List[BallDetection]:
        """Detect baseballs in a single frame."""
        if not self._initialized:
            self.initialize()

        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    detections.append(BallDetection(
                        frame_number=frame_number,
                        timestamp_ms=timestamp_ms,
                        centroid=(cx, cy),
                        bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                        confidence=conf,
                    ))

        # Sort by confidence, return best detection(s)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_video(
        self,
        video_path: str,
        max_frames: int = None,
        show_progress: bool = True,
    ) -> List[BallDetection]:
        """Detect baseballs throughout a video."""
        if not self._initialized:
            self.initialize()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        all_detections = []
        frame_num = 0

        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Detecting baseball")

        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_num / fps) * 1000
            detections = self.detect_frame(frame, frame_num, timestamp_ms)

            # Keep best detection per frame
            if detections:
                all_detections.append(detections[0])

            frame_num += 1
            if show_progress:
                pbar.update(1)

        cap.release()
        if show_progress:
            pbar.close()

        logger.info(f"Detected baseball in {len(all_detections)}/{total_frames} frames")
        return all_detections

    def get_best_detection(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
    ) -> Optional[BallDetection]:
        """Get single best detection for a frame."""
        detections = self.detect_frame(frame, frame_number, timestamp_ms)
        return detections[0] if detections else None

    def visualize(
        self,
        frame: np.ndarray,
        detections: List[BallDetection],
    ) -> np.ndarray:
        """Draw detections on frame."""
        vis = frame.copy()

        for det in detections:
            x, y, w, h = det.bbox
            cx, cy = det.centroid
            conf = det.confidence

            # Color by confidence
            if conf > 0.3:
                color = (0, 255, 0)  # Green
            elif conf > 0.1:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red

            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (255, 0, 255), -1)

            label = f"ball {conf:.0%}"
            cv2.putText(vis, label, (int(x), int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
