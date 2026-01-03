"""SQLAlchemy models for baseball biomechanics database."""

import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class PlayerRole(enum.Enum):
    """Enum for player roles in a play."""
    PITCHER = "pitcher"
    BATTER = "batter"
    CATCHER = "catcher"


class Game(Base):
    """
    Represents a baseball game.

    Attributes:
        game_pk: Primary key from MLB (game identifier)
        game_date: Date of the game
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        venue: Stadium name
    """
    __tablename__ = "games"

    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(50), nullable=False)
    away_team: Mapped[str] = mapped_column(String(50), nullable=False)
    venue: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationships
    plays: Mapped[List["Play"]] = relationship("Play", back_populates="game", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Game(game_pk={self.game_pk}, {self.away_team}@{self.home_team}, {self.game_date})>"


class Player(Base):
    """
    Represents a baseball player.

    Attributes:
        player_id: Primary key from MLB (player identifier)
        player_name: Full name of the player
        team: Current team abbreviation
        position: Primary position
        throws: Throwing hand (L/R)
        bats: Batting side (L/R/S)
    """
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    team: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    position: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    throws: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)
    bats: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)

    # Relationships
    pitches_thrown: Mapped[List["Play"]] = relationship(
        "Play", foreign_keys="Play.pitcher_id", back_populates="pitcher"
    )
    at_bats: Mapped[List["Play"]] = relationship(
        "Play", foreign_keys="Play.batter_id", back_populates="batter"
    )
    catches: Mapped[List["Play"]] = relationship(
        "Play", foreign_keys="Play.catcher_id", back_populates="catcher"
    )
    pose_sequences: Mapped[List["PoseSequence"]] = relationship(
        "PoseSequence", back_populates="player"
    )

    def __repr__(self) -> str:
        return f"<Player(player_id={self.player_id}, name={self.player_name}, team={self.team})>"


class Play(Base):
    """
    Represents a single pitch/play with Statcast data.
    """
    __tablename__ = "plays"

    play_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, ForeignKey("games.game_pk"), nullable=False, index=True)
    inning: Mapped[int] = mapped_column(Integer, nullable=False)
    at_bat_number: Mapped[int] = mapped_column(Integer, nullable=False)
    pitch_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Player references
    pitcher_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), nullable=False, index=True)
    batter_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), nullable=False, index=True)
    catcher_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("players.player_id"), nullable=True)

    # Pitch characteristics
    pitch_type: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    release_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spin_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    release_pos_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    release_pos_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Pitch movement
    pfx_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pfx_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Pitch location
    plate_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plate_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    zone: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Hit data
    launch_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    launch_angle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hit_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Outcome
    events: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Video information
    video_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    video_local_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="plays")
    pitcher: Mapped["Player"] = relationship("Player", foreign_keys=[pitcher_id], back_populates="pitches_thrown")
    batter: Mapped["Player"] = relationship("Player", foreign_keys=[batter_id], back_populates="at_bats")
    catcher: Mapped[Optional["Player"]] = relationship("Player", foreign_keys=[catcher_id], back_populates="catches")
    pose_sequences: Mapped[List["PoseSequence"]] = relationship("PoseSequence", back_populates="play", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_play_game_inning", "game_pk", "inning"),
        Index("idx_play_pitcher_date", "pitcher_id", "game_pk"),
    )

    def __repr__(self) -> str:
        return f"<Play(play_id={self.play_id}, pitcher={self.pitcher_id}, batter={self.batter_id}, pitch_type={self.pitch_type})>"


class PoseSequence(Base):
    """Represents a sequence of pose estimations for a player in a play."""
    __tablename__ = "pose_sequences"

    sequence_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    play_id: Mapped[int] = mapped_column(Integer, ForeignKey("plays.play_id"), nullable=False, index=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), nullable=False, index=True)
    player_role: Mapped[PlayerRole] = mapped_column(Enum(PlayerRole), nullable=False)
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    play: Mapped["Play"] = relationship("Play", back_populates="pose_sequences")
    player: Mapped["Player"] = relationship("Player", back_populates="pose_sequences")
    frames: Mapped[List["PoseFrame"]] = relationship("PoseFrame", back_populates="sequence", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<PoseSequence(sequence_id={self.sequence_id}, player_role={self.player_role.value}, frames={self.start_frame}-{self.end_frame})>"


class PoseFrame(Base):
    """Represents a single frame within a pose sequence."""
    __tablename__ = "pose_frames"

    frame_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sequence_id: Mapped[int] = mapped_column(Integer, ForeignKey("pose_sequences.sequence_id"), nullable=False, index=True)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    sequence: Mapped["PoseSequence"] = relationship("PoseSequence", back_populates="frames")
    keypoints: Mapped[List["Keypoint"]] = relationship("Keypoint", back_populates="frame", cascade="all, delete-orphan")
    segmentation_masks: Mapped[List["SegmentationMask"]] = relationship("SegmentationMask", back_populates="frame", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_frame_sequence_number", "sequence_id", "frame_number"),
    )

    def __repr__(self) -> str:
        return f"<PoseFrame(frame_id={self.frame_id}, frame_number={self.frame_number}, timestamp_ms={self.timestamp_ms})>"


class Keypoint(Base):
    """Represents a single keypoint (joint/landmark) in a pose frame."""
    __tablename__ = "keypoints"

    keypoint_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    frame_id: Mapped[int] = mapped_column(Integer, ForeignKey("pose_frames.frame_id"), nullable=False, index=True)
    keypoint_name: Mapped[str] = mapped_column(String(50), nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_occluded: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    frame: Mapped["PoseFrame"] = relationship("PoseFrame", back_populates="keypoints")

    __table_args__ = (
        Index("idx_keypoint_frame_name", "frame_id", "keypoint_name"),
    )

    def __repr__(self) -> str:
        z_val = f"{self.z:.2f}" if self.z else "N/A"
        return f"<Keypoint(keypoint_id={self.keypoint_id}, name={self.keypoint_name}, pos=({self.x:.2f}, {self.y:.2f}, {z_val}))>"


class SegmentationMask(Base):
    """Represents a SAM 3 segmentation mask for a player in a frame."""
    __tablename__ = "segmentation_masks"

    mask_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    frame_id: Mapped[int] = mapped_column(Integer, ForeignKey("pose_frames.frame_id"), nullable=False, index=True)
    player_role: Mapped[PlayerRole] = mapped_column(Enum(PlayerRole), nullable=False)
    mask_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Bounding box
    player_bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    player_bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    player_bbox_width: Mapped[float] = mapped_column(Float, nullable=False)
    player_bbox_height: Mapped[float] = mapped_column(Float, nullable=False)

    # Metadata
    mask_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    text_prompt_used: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    frame: Mapped["PoseFrame"] = relationship("PoseFrame", back_populates="segmentation_masks")

    __table_args__ = (
        Index("idx_mask_frame_role", "frame_id", "player_role"),
    )

    def __repr__(self) -> str:
        return f"<SegmentationMask(mask_id={self.mask_id}, role={self.player_role.value}, prompt='{self.text_prompt_used}')>"


# Standard keypoint names for reference
MEDIAPIPE_KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
