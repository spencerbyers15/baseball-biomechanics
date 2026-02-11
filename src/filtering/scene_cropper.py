"""
Video temporal cropping using scene cut detection + EfficientNet-B0 classification.

Pipeline:
1. Detect camera cuts via histogram diff (threshold 0.25)
2. Classify each segment by sampling frames and running EfficientNet-B0
3. Pick the longest main_angle segment
4. Crop with ffmpeg (stream copy for speed)
"""

import cv2
import numpy as np
import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A video segment between two cuts."""
    start_frame: int
    end_frame: int
    fps: float

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_time(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


@dataclass
class ClassifiedSegment(Segment):
    """A segment with classification results."""
    label: str = ""          # "main_angle" or "other"
    avg_confidence: float = 0.0
    main_angle_score: float = 0.0  # average P(main_angle) across sampled frames


def compute_histogram_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute normalized histogram difference between two frames.

    Uses color histogram comparison. Returns value 0-1 where higher = more different.
    """
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    hist_size = [50, 60]
    h_ranges = [0, 180]
    s_ranges = [0, 256]

    hist1 = cv2.calcHist([hsv1], [0, 1], None, hist_size, h_ranges + s_ranges)
    hist2 = cv2.calcHist([hsv2], [0, 1], None, hist_size, h_ranges + s_ranges)

    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return 1.0 - max(0.0, similarity)


def compute_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute structural similarity between two frames.

    Returns value 0-1 where higher = more different (inverted from standard SSIM).
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    scale = 0.25
    gray1 = cv2.resize(gray1, None, fx=scale, fy=scale)
    gray2 = cv2.resize(gray2, None, fx=scale, fy=scale)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    gray1 = gray1.astype(np.float64)
    gray2 = gray2.astype(np.float64)

    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim = np.mean(ssim_map)
    return 1.0 - ssim


def detect_scene_cuts(
    video_path: str,
    threshold: float = 0.08,
    min_segment_duration: float = 0.5,
    subsample: int = 4,
    show_progress: bool = False,
) -> Tuple[List[int], float, int]:
    """
    Detect scene cuts using histogram diff with frame subsampling.

    Compares every Nth frame (default 4) for speed. When a cut is detected
    between sampled frames, the cut is placed at the midpoint.

    Args:
        video_path: Path to video file
        threshold: Cut detection threshold (default 0.25)
        min_segment_duration: Minimum segment length in seconds
        subsample: Compare every Nth frame (default 4, ~4x speedup)
        show_progress: Show progress bar

    Returns:
        (cut_frames, fps, total_frames)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_segment_frames = int(min_segment_duration * fps)

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")

    prev_frame_num = 0
    cut_frames = []

    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="Detecting cuts")
        pbar.update(1)

    frame_num = 0
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if show_progress:
            pbar.update(1)

        # Only compare every Nth frame
        if frame_num % subsample != 0:
            continue

        score = compute_histogram_diff(prev_frame, curr_frame)

        if score > threshold:
            # Place cut at midpoint between the two sampled frames
            cut_at = (prev_frame_num + frame_num) // 2
            if len(cut_frames) == 0 or (cut_at - cut_frames[-1]) >= min_segment_frames:
                cut_frames.append(cut_at)

        prev_frame = curr_frame
        prev_frame_num = frame_num

    cap.release()
    if show_progress:
        pbar.close()

    logger.info(f"Detected {len(cut_frames)} cuts in {video_path}")
    return cut_frames, fps, total_frames


def classify_segments(
    video_path: str,
    segments: List[Segment],
    classifier=None,
    samples_per_segment: int = 5,
) -> List[ClassifiedSegment]:
    """
    Classify each segment by sampling frames and running EfficientNet-B0.

    Args:
        video_path: Path to video file
        segments: List of Segment objects from cut detection
        classifier: CameraAngleClassifier instance (created if None)
        samples_per_segment: Number of frames to sample per segment

    Returns:
        List of ClassifiedSegment with labels and scores
    """
    from .camera_filter import CameraAngleClassifier

    if classifier is None:
        classifier = CameraAngleClassifier()
        classifier.initialize()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    classified = []

    for seg in segments:
        seg_len = seg.end_frame - seg.start_frame

        # Choose sample positions evenly distributed within segment
        if seg_len <= 0:
            classified.append(ClassifiedSegment(
                start_frame=seg.start_frame, end_frame=seg.end_frame,
                fps=seg.fps, label="other", avg_confidence=0.5,
                main_angle_score=0.0))
            continue

        n_samples = min(samples_per_segment, max(1, seg_len // 10))
        if n_samples == 1:
            positions = [seg.start_frame + seg_len // 2]
        else:
            step = seg_len // (n_samples + 1)
            positions = [seg.start_frame + step * (i + 1) for i in range(n_samples)]

        # Read and classify sampled frames
        frames = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        if not frames:
            classified.append(ClassifiedSegment(
                start_frame=seg.start_frame, end_frame=seg.end_frame,
                fps=seg.fps, label="other", avg_confidence=0.5,
                main_angle_score=0.0))
            continue

        # Batch classify
        results = classifier.classify_frames_batch(frames)

        # Compute main_angle_score: average P(main_angle) across samples
        # For frames labeled main_angle, confidence IS P(main_angle)
        # For frames labeled other, P(main_angle) = 1 - confidence
        main_scores = []
        for label, conf in results:
            if label == "main_angle":
                main_scores.append(conf)
            else:
                main_scores.append(1.0 - conf)

        main_angle_score = np.mean(main_scores)
        avg_confidence = np.mean([conf for _, conf in results])

        # Label based on majority
        label = "main_angle" if main_angle_score > 0.5 else "other"

        classified.append(ClassifiedSegment(
            start_frame=seg.start_frame,
            end_frame=seg.end_frame,
            fps=seg.fps,
            label=label,
            avg_confidence=avg_confidence,
            main_angle_score=main_angle_score,
        ))

    cap.release()

    n_main = sum(1 for s in classified if s.label == "main_angle")
    logger.info(f"Classified {len(classified)} segments: {n_main} main_angle, "
                f"{len(classified) - n_main} other")

    return classified


def select_main_segment(segments: List[ClassifiedSegment]) -> Optional[ClassifiedSegment]:
    """
    Select the single best main_angle segment.

    If multiple segments are main_angle, returns the longest one.
    Returns None if no main_angle segments exist.
    """
    main_segments = [s for s in segments if s.label == "main_angle"]

    if not main_segments:
        logger.warning("No main_angle segments found")
        return None

    best = max(main_segments, key=lambda s: s.duration)
    logger.info(f"Selected main segment: {best.start_time:.2f}s - {best.end_time:.2f}s "
                f"({best.duration:.2f}s, score={best.main_angle_score:.3f})")
    return best


def crop_video_ffmpeg(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> bool:
    """
    Crop video to a single time region using ffmpeg stream copy.

    Args:
        input_path: Input video path
        output_path: Output video path
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        True if successful
    """
    duration = end_time - start_time
    if duration <= 0:
        logger.warning("Invalid crop region")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-c", "copy",
        str(output_path),
    ]

    logger.info(f"Cropping {input_path} -> {output_path} "
                f"({start_time:.2f}s - {end_time:.2f}s, {duration:.2f}s)")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg error: {result.stderr}")
        return False

    return True


def crop_to_main_angle(
    video_path: str,
    output_path: str,
    cut_threshold: float = 0.08,
    min_segment_duration: float = 0.5,
    samples_per_segment: int = 5,
    classifier=None,
    show_progress: bool = False,
) -> dict:
    """
    Full pipeline: detect cuts, classify segments, crop to longest main_angle.

    Args:
        video_path: Input video path
        output_path: Output video path
        cut_threshold: Histogram diff threshold for cut detection
        min_segment_duration: Minimum segment length in seconds
        samples_per_segment: Frames to sample per segment for classification
        classifier: CameraAngleClassifier instance (created if None)
        show_progress: Show progress bars

    Returns:
        Dict with processing results and diagnostics
    """
    logger.info(f"Processing {video_path}")

    # Step 1: Detect cuts
    cut_frames, fps, total_frames = detect_scene_cuts(
        video_path,
        threshold=cut_threshold,
        min_segment_duration=min_segment_duration,
        show_progress=show_progress,
    )

    # Build segments from cuts
    boundaries = [0] + cut_frames + [total_frames]
    segments = []
    for i in range(len(boundaries) - 1):
        segments.append(Segment(
            start_frame=boundaries[i],
            end_frame=boundaries[i + 1],
            fps=fps,
        ))

    # Step 2: Classify segments
    classified = classify_segments(
        video_path, segments,
        classifier=classifier,
        samples_per_segment=samples_per_segment,
    )

    # Step 3: Select longest main_angle segment
    main_seg = select_main_segment(classified)

    # Step 4: Crop
    success = False
    if main_seg:
        success = crop_video_ffmpeg(
            video_path, output_path,
            main_seg.start_time, main_seg.end_time,
        )

    return {
        "success": success,
        "input_path": video_path,
        "output_path": output_path if success else None,
        "cut_threshold": cut_threshold,
        "num_cuts": len(cut_frames),
        "cut_frames": cut_frames,
        "segments": [
            {
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
                "label": s.label,
                "main_angle_score": s.main_angle_score,
            }
            for s in classified
        ],
        "main_segment": {
            "start_time": main_seg.start_time,
            "end_time": main_seg.end_time,
            "duration": main_seg.duration,
            "main_angle_score": main_seg.main_angle_score,
        } if main_seg else None,
        "fps": fps,
        "total_frames": total_frames,
    }
