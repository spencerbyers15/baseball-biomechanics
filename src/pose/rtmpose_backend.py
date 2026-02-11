"""RTMPose-X pose estimation backend via rtmlib (GPU-accelerated ONNX).

Uses RTMPose-X (384x288, 78.8 AP on COCO) through rtmlib's ONNX Runtime
backend. Supports CUDA GPU acceleration for ~3-5ms/frame inference.

Requires: onnxruntime-gpu, rtmlib (installed with --no-deps)
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import List
from urllib import request

import numpy as np


def _ensure_cudnn_on_path():
    """Add nvidia-cudnn-cu12 DLLs to PATH if installed via pip.

    onnxruntime-gpu looks for cudnn64_9.dll on PATH, but pip installs
    it under site-packages/nvidia/cudnn/bin/. This adds that directory
    to PATH so CUDA execution provider can load.
    """
    try:
        import nvidia.cudnn
        cudnn_bin = Path(nvidia.cudnn.__path__[0]) / "bin"
        if cudnn_bin.is_dir() and str(cudnn_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(cudnn_bin) + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass


_ensure_cudnn_on_path()

from src.pose.base import KeypointData, PoseBackend, PoseResult

logger = logging.getLogger(__name__)

# RTMPose COCO body keypoints (17) — matches STANDARD_KEYPOINTS in base.py
RTMPOSE_KEYPOINTS = [
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

# RTMPose-X body model (384x288 input, body7 training)
MODEL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
)
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "rtmpose"


class RTMPoseBackend(PoseBackend):
    """RTMPose-X pose estimation backend using rtmlib (ONNX Runtime GPU).

    Produces 17 COCO keypoints in pixel coordinates. No z-coordinate
    (2D model only, z=None). Designed as a drop-in replacement for
    MediaPipeBackend in the player_pose pipeline.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        device: str = "cuda",
    ):
        """Initialize the RTMPose backend.

        Args:
            min_detection_confidence: Minimum keypoint confidence threshold.
            min_tracking_confidence: Unused, kept for API compatibility.
            device: Inference device ('cuda' or 'cpu').
        """
        super().__init__(min_detection_confidence, min_tracking_confidence)
        self.device = device
        self._pose = None

    @property
    def name(self) -> str:
        return "rtmpose-x"

    @property
    def keypoint_names(self) -> List[str]:
        return RTMPOSE_KEYPOINTS.copy()

    @property
    def supports_3d(self) -> bool:
        return False

    def initialize(self) -> None:
        """Initialize the RTMPose ONNX model via rtmlib."""
        if self._is_initialized:
            return

        model_path = self._ensure_model()

        try:
            from rtmlib import RTMPose

            self._pose = RTMPose(
                onnx_model=str(model_path),
                model_input_size=(288, 384),  # (width, height)
                backend="onnxruntime",
                device=self.device,
            )
            self._is_initialized = True
            logger.info(f"RTMPose-X initialized on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"rtmlib package error: {e}. "
                "Install with: pip install rtmlib --no-deps"
            )

    def _ensure_model(self) -> Path:
        """Download RTMPose-X ONNX model if not present.

        Downloads the zip from OpenMMLab, extracts the .onnx file to
        models/rtmpose/, and deletes the zip.

        Returns:
            Path to the .onnx model file.
        """
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Check for any existing .onnx file
        existing = list(MODEL_DIR.glob("*.onnx"))
        if existing:
            logger.info(f"Using existing model: {existing[0].name}")
            return existing[0]

        # Download
        zip_path = MODEL_DIR / "rtmpose-x.zip"
        logger.info(f"Downloading RTMPose-X model to {MODEL_DIR}...")
        request.urlretrieve(MODEL_URL, zip_path)

        # Extract .onnx file(s)
        logger.info("Extracting model...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".onnx"):
                    target = MODEL_DIR / Path(name).name
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    logger.info(f"Extracted: {target.name}")

        zip_path.unlink()
        logger.info("Model download complete")

        # Find the extracted model
        onnx_files = list(MODEL_DIR.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx file found in {MODEL_DIR} after extraction"
            )
        return onnx_files[0]

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
    ) -> PoseResult:
        """Run pose estimation on a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV).
            frame_number: Frame index for result tracking.
            timestamp_ms: Timestamp in milliseconds.

        Returns:
            PoseResult with 17 COCO keypoints in pixel coordinates.
        """
        if not self._is_initialized:
            self.initialize()

        # rtmlib RTMPose: bboxes=[] uses full image as single detection
        keypoints, scores = self._pose(frame)

        pose_result = PoseResult(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            model_name=self.name,
        )

        # keypoints shape: (N, 17, 2), scores shape: (N, 17)
        if keypoints is None or len(keypoints) == 0:
            pose_result.is_valid = False
            return pose_result

        # Take the first (and typically only) person
        kps = keypoints[0]   # (17, 2)
        scs = scores[0]      # (17,)

        for idx, kp_name in enumerate(RTMPOSE_KEYPOINTS):
            pose_result.keypoints.append(KeypointData(
                name=kp_name,
                x=float(kps[idx, 0]),
                y=float(kps[idx, 1]),
                z=None,  # RTMPose body is 2D only
                confidence=float(scs[idx]),
                is_occluded=scs[idx] < self.min_detection_confidence,
            ))

        return pose_result

    def cleanup(self) -> None:
        """Release RTMPose resources."""
        self._pose = None
        self._is_initialized = False
        logger.debug("RTMPose-X resources cleaned up")
