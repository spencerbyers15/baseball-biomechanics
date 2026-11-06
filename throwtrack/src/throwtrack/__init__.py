"""ThrowTrack: count and characterize pitchers' non-mound throws from ordinary video."""

from .pipeline import SessionConfig, analyze_session
from .schema import PoseTrack, ThrowEvent

__all__ = ["PoseTrack", "SessionConfig", "ThrowEvent", "analyze_session"]
__version__ = "0.1.0"
