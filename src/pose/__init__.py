"""Pose estimation module with pluggable backends."""

from src.pose.base import PoseBackend, PoseResult, KeypointData
from src.pose.mediapipe_backend import MediaPipeBackend
from src.pose.rtmpose_backend import RTMPoseBackend

__all__ = [
    "PoseBackend",
    "PoseResult",
    "KeypointData",
    "MediaPipeBackend",
    "RTMPoseBackend",
]
