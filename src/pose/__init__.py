"""Pose estimation module with pluggable backends."""

from src.pose.base import PoseBackend, PoseResult, KeypointData
from src.pose.mediapipe_backend import MediaPipeBackend

__all__ = [
    "PoseBackend",
    "PoseResult",
    "KeypointData",
    "MediaPipeBackend",
]
