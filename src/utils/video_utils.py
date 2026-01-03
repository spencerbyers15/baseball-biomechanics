"""Video utilities for frame extraction and manipulation."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """
    Information about a video file.

    Attributes:
        path: Path to the video file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        total_frames: Total number of frames.
        duration_seconds: Duration in seconds.
        codec: Video codec fourcc code.
    """
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration_seconds,
            "codec": self.codec,
        }


def get_video_info(video_path: str) -> Optional[VideoInfo]:
    """
    Get information about a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoInfo object or None if video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])

        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec,
        )
    finally:
        cap.release()


def extract_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    format: str = "jpg",
    show_progress: bool = True,
) -> List[Tuple[int, str]]:
    """
    Extract frames from a video to image files.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save frames (creates temp dir if None).
        start_frame: Starting frame number.
        end_frame: Ending frame number (None for all frames).
        step: Extract every Nth frame.
        format: Output image format (jpg, png).
        show_progress: Whether to show progress bar.

    Returns:
        List of (frame_number, file_path) tuples.
    """
    video_path = Path(video_path)

    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames

    end_frame = min(end_frame, total_frames)
    frames_to_extract = range(start_frame, end_frame, step)

    extracted = []
    iterator = tqdm(frames_to_extract, desc="Extracting frames") if show_progress else frames_to_extract

    for frame_num in iterator:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"Could not read frame {frame_num}")
            continue

        output_path = output_dir / f"frame_{frame_num:06d}.{format}"
        cv2.imwrite(str(output_path), frame)
        extracted.append((frame_num, str(output_path)))

    cap.release()
    logger.info(f"Extracted {len(extracted)} frames to {output_dir}")

    return extracted


class VideoProcessor:
    """
    Video processing utility for frame-by-frame operations.

    Provides an iterator interface for processing video frames
    with optional resizing, cropping, and color conversion.
    """

    def __init__(
        self,
        video_path: str,
        resize: Optional[Tuple[int, int]] = None,
        color_mode: str = "bgr",
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ):
        """
        Initialize the video processor.

        Args:
            video_path: Path to the video file.
            resize: Optional (width, height) to resize frames.
            color_mode: Output color mode (bgr, rgb, gray).
            start_frame: Starting frame number.
            end_frame: Ending frame number (None for all frames).
        """
        self.video_path = Path(video_path)
        self.resize = resize
        self.color_mode = color_mode.lower()
        self.start_frame = start_frame
        self.end_frame = end_frame

        self._cap = None
        self._info = None

    @property
    def info(self) -> Optional[VideoInfo]:
        """Get video information."""
        if self._info is None:
            self._info = get_video_info(str(self.video_path))
        return self._info

    def open(self) -> None:
        """Open the video file."""
        if self._cap is not None:
            return

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        if self.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        logger.debug(f"Opened video: {self.video_path}")

    def close(self) -> None:
        """Close the video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug(f"Closed video: {self.video_path}")

    def read_frame(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """
        Read the next frame.

        Returns:
            Tuple of (frame_number, timestamp_ms, frame) or None if no more frames.
        """
        if self._cap is None:
            self.open()

        frame_num = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

        if self.end_frame is not None and frame_num >= self.end_frame:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Get timestamp
        timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)

        # Apply transformations
        frame = self._process_frame(frame)

        return frame_num, timestamp_ms, frame

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply processing transformations to frame."""
        # Resize if specified
        if self.resize is not None:
            frame = cv2.resize(frame, self.resize)

        # Color conversion
        if self.color_mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.color_mode == "gray":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def iterate_frames(
        self,
        show_progress: bool = True,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Iterate over all frames in the video.

        Args:
            show_progress: Whether to show progress bar.

        Yields:
            Tuple of (frame_number, timestamp_ms, frame).
        """
        self.open()

        total = self.info.total_frames if self.info else 0
        if self.end_frame:
            total = min(total, self.end_frame - self.start_frame)

        pbar = tqdm(total=total, desc="Processing video") if show_progress else None

        try:
            while True:
                result = self.read_frame()
                if result is None:
                    break

                yield result

                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()
            self.close()

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.

        Args:
            frame_number: Frame number to retrieve.

        Returns:
            Frame as numpy array or None if not found.
        """
        self.open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()

        if not ret:
            return None

        return self._process_frame(frame)

    def get_frames_at(self, frame_numbers: List[int]) -> List[Tuple[int, np.ndarray]]:
        """
        Get multiple specific frames.

        Args:
            frame_numbers: List of frame numbers to retrieve.

        Returns:
            List of (frame_number, frame) tuples.
        """
        self.open()
        frames = []

        for frame_num in sorted(frame_numbers):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self._cap.read()

            if ret:
                frames.append((frame_num, self._process_frame(frame)))

        return frames

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __iter__(self):
        """Iterate over frames."""
        return self.iterate_frames(show_progress=False)


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> None:
    """
    Create a video file from a list of frames.

    Args:
        frames: List of frames as numpy arrays.
        output_path: Path for output video file.
        fps: Frames per second.
        codec: Video codec fourcc code.
    """
    if not frames:
        raise ValueError("No frames provided")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
    finally:
        out.release()

    logger.info(f"Created video: {output_path} ({len(frames)} frames)")


def crop_frame(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding: int = 0,
) -> np.ndarray:
    """
    Crop a frame using a bounding box.

    Args:
        frame: Input frame.
        bbox: Bounding box (x, y, width, height).
        padding: Additional padding around the crop.

    Returns:
        Cropped frame.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox

    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min(w, int(x + bw) + padding)
    y2 = min(h, int(y + bh) + padding)

    return frame[y1:y2, x1:x2].copy()


def apply_mask_to_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply a binary mask to a frame.

    Args:
        frame: Input frame.
        mask: Binary mask (same dimensions as frame).
        background: Optional background to show through mask.

    Returns:
        Masked frame.
    """
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    if len(mask.shape) == 2:
        mask = np.stack([mask] * 3, axis=-1)

    if background is None:
        background = np.zeros_like(frame)

    return np.where(mask > 0, frame, background).astype(np.uint8)


def overlay_pose(
    frame: np.ndarray,
    keypoints: List[Dict],
    connections: Optional[List[Tuple[str, str]]] = None,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    line_color: Tuple[int, int, int] = (255, 0, 0),
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Overlay pose keypoints on a frame.

    Args:
        frame: Input frame.
        keypoints: List of keypoint dictionaries with x, y, name.
        connections: Optional list of (start_name, end_name) for bones.
        point_color: BGR color for keypoints.
        line_color: BGR color for connections.
        point_radius: Radius of keypoint circles.
        line_thickness: Thickness of connection lines.

    Returns:
        Frame with pose overlay.
    """
    output = frame.copy()

    # Create name to position mapping
    kp_map = {kp["name"]: (int(kp["x"]), int(kp["y"])) for kp in keypoints}

    # Draw connections first (under points)
    if connections:
        for start_name, end_name in connections:
            if start_name in kp_map and end_name in kp_map:
                cv2.line(
                    output,
                    kp_map[start_name],
                    kp_map[end_name],
                    line_color,
                    line_thickness,
                )

    # Draw keypoints
    for kp in keypoints:
        x, y = int(kp["x"]), int(kp["y"])
        cv2.circle(output, (x, y), point_radius, point_color, -1)

    return output
