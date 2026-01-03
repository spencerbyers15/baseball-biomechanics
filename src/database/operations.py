"""Database CRUD operations for baseball biomechanics."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import (
    Game,
    Keypoint,
    Play,
    Player,
    PlayerRole,
    PoseFrame,
    PoseSequence,
    SegmentationMask,
)

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """
    Database operations handler for baseball biomechanics data.

    Provides CRUD operations for all database models with proper
    error handling and logging.
    """

    def __init__(self, session: Session):
        """
        Initialize database operations.

        Args:
            session: SQLAlchemy session instance.
        """
        self.session = session

    # ==================== Game Operations ====================

    def create_game(
        self,
        game_pk: int,
        game_date: datetime,
        home_team: str,
        away_team: str,
        venue: Optional[str] = None,
    ) -> Game:
        """
        Create a new game record.

        Args:
            game_pk: MLB game primary key.
            game_date: Date of the game.
            home_team: Home team abbreviation.
            away_team: Away team abbreviation.
            venue: Stadium name (optional).

        Returns:
            Created Game instance.
        """
        game = Game(
            game_pk=game_pk,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            venue=venue,
        )
        self.session.add(game)
        self.session.commit()
        logger.debug(f"Created game: {game}")
        return game

    def get_game(self, game_pk: int) -> Optional[Game]:
        """Get a game by its primary key."""
        return self.session.get(Game, game_pk)

    def get_or_create_game(
        self,
        game_pk: int,
        game_date: datetime,
        home_team: str,
        away_team: str,
        venue: Optional[str] = None,
    ) -> Game:
        """Get existing game or create new one."""
        game = self.get_game(game_pk)
        if game is None:
            game = self.create_game(game_pk, game_date, home_team, away_team, venue)
        return game

    def get_games_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Game]:
        """Get all games within a date range."""
        return (
            self.session.query(Game)
            .filter(Game.game_date >= start_date, Game.game_date <= end_date)
            .order_by(Game.game_date)
            .all()
        )

    # ==================== Player Operations ====================

    def create_player(
        self,
        player_id: int,
        player_name: str,
        team: Optional[str] = None,
        position: Optional[str] = None,
        throws: Optional[str] = None,
        bats: Optional[str] = None,
    ) -> Player:
        """
        Create a new player record.

        Args:
            player_id: MLB player ID.
            player_name: Full name of the player.
            team: Team abbreviation.
            position: Primary position.
            throws: Throwing hand (L/R).
            bats: Batting side (L/R/S).

        Returns:
            Created Player instance.
        """
        player = Player(
            player_id=player_id,
            player_name=player_name,
            team=team,
            position=position,
            throws=throws,
            bats=bats,
        )
        self.session.add(player)
        self.session.commit()
        logger.debug(f"Created player: {player}")
        return player

    def get_player(self, player_id: int) -> Optional[Player]:
        """Get a player by their ID."""
        return self.session.get(Player, player_id)

    def get_or_create_player(
        self,
        player_id: int,
        player_name: str,
        team: Optional[str] = None,
        position: Optional[str] = None,
        throws: Optional[str] = None,
        bats: Optional[str] = None,
    ) -> Player:
        """Get existing player or create new one."""
        player = self.get_player(player_id)
        if player is None:
            player = self.create_player(
                player_id, player_name, team, position, throws, bats
            )
        return player

    def search_players_by_name(self, name: str) -> List[Player]:
        """Search for players by name (case-insensitive partial match)."""
        return (
            self.session.query(Player)
            .filter(Player.player_name.ilike(f"%{name}%"))
            .all()
        )

    # ==================== Play Operations ====================

    def create_play(self, **kwargs) -> Play:
        """
        Create a new play record with Statcast data.

        Args:
            **kwargs: Play attributes (see Play model for fields).

        Returns:
            Created Play instance.
        """
        play = Play(**kwargs)
        self.session.add(play)
        self.session.commit()
        logger.debug(f"Created play: {play}")
        return play

    def get_play(self, play_id: int) -> Optional[Play]:
        """Get a play by its ID."""
        return self.session.get(Play, play_id)

    def get_plays_by_pitcher(
        self,
        pitcher_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Play]:
        """Get all plays for a specific pitcher, optionally filtered by date."""
        query = self.session.query(Play).filter(Play.pitcher_id == pitcher_id)

        if start_date or end_date:
            query = query.join(Game)
            if start_date:
                query = query.filter(Game.game_date >= start_date)
            if end_date:
                query = query.filter(Game.game_date <= end_date)

        return query.order_by(Play.play_id).all()

    def get_plays_by_batter(
        self,
        batter_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Play]:
        """Get all plays for a specific batter, optionally filtered by date."""
        query = self.session.query(Play).filter(Play.batter_id == batter_id)

        if start_date or end_date:
            query = query.join(Game)
            if start_date:
                query = query.filter(Game.game_date >= start_date)
            if end_date:
                query = query.filter(Game.game_date <= end_date)

        return query.order_by(Play.play_id).all()

    def get_plays_without_video(self) -> List[Play]:
        """Get plays that don't have a local video file."""
        return (
            self.session.query(Play)
            .filter(
                Play.video_url.isnot(None),
                (Play.video_local_path.is_(None) | (Play.video_local_path == "")),
            )
            .all()
        )

    def get_plays_without_pose_data(self) -> List[Play]:
        """Get plays that have video but no pose sequences."""
        return (
            self.session.query(Play)
            .filter(
                Play.video_local_path.isnot(None),
                ~Play.pose_sequences.any(),
            )
            .all()
        )

    def update_play_video_path(self, play_id: int, video_path: str) -> Optional[Play]:
        """Update the local video path for a play."""
        play = self.get_play(play_id)
        if play:
            play.video_local_path = video_path
            self.session.commit()
            logger.debug(f"Updated video path for play {play_id}: {video_path}")
        return play

    # ==================== Pose Sequence Operations ====================

    def create_pose_sequence(
        self,
        play_id: int,
        player_id: int,
        player_role: PlayerRole,
        start_frame: int,
        end_frame: int,
        fps: float,
        model_used: str,
    ) -> PoseSequence:
        """
        Create a new pose sequence.

        Args:
            play_id: Associated play ID.
            player_id: Player being tracked.
            player_role: Role of the player (pitcher/batter/catcher).
            start_frame: Starting frame number.
            end_frame: Ending frame number.
            fps: Frames per second of the video.
            model_used: Name of the pose estimation model.

        Returns:
            Created PoseSequence instance.
        """
        sequence = PoseSequence(
            play_id=play_id,
            player_id=player_id,
            player_role=player_role,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            model_used=model_used,
        )
        self.session.add(sequence)
        self.session.commit()
        logger.debug(f"Created pose sequence: {sequence}")
        return sequence

    def get_pose_sequence(self, sequence_id: int) -> Optional[PoseSequence]:
        """Get a pose sequence by its ID."""
        return self.session.get(PoseSequence, sequence_id)

    def get_pose_sequences_for_play(
        self, play_id: int, player_role: Optional[PlayerRole] = None
    ) -> List[PoseSequence]:
        """Get all pose sequences for a play, optionally filtered by role."""
        query = self.session.query(PoseSequence).filter(
            PoseSequence.play_id == play_id
        )
        if player_role:
            query = query.filter(PoseSequence.player_role == player_role)
        return query.all()

    # ==================== Pose Frame Operations ====================

    def create_pose_frame(
        self,
        sequence_id: int,
        frame_number: int,
        timestamp_ms: float,
    ) -> PoseFrame:
        """Create a new pose frame."""
        frame = PoseFrame(
            sequence_id=sequence_id,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
        )
        self.session.add(frame)
        self.session.commit()
        return frame

    def bulk_create_pose_frames(
        self,
        sequence_id: int,
        frame_data: List[Dict[str, Any]],
    ) -> List[PoseFrame]:
        """
        Create multiple pose frames in bulk.

        Args:
            sequence_id: Parent sequence ID.
            frame_data: List of dicts with frame_number and timestamp_ms.

        Returns:
            List of created PoseFrame instances.
        """
        frames = [
            PoseFrame(
                sequence_id=sequence_id,
                frame_number=fd["frame_number"],
                timestamp_ms=fd["timestamp_ms"],
            )
            for fd in frame_data
        ]
        self.session.add_all(frames)
        self.session.commit()
        logger.debug(f"Created {len(frames)} pose frames for sequence {sequence_id}")
        return frames

    # ==================== Keypoint Operations ====================

    def create_keypoint(
        self,
        frame_id: int,
        keypoint_name: str,
        x: float,
        y: float,
        z: Optional[float],
        confidence: float,
        is_occluded: bool = False,
    ) -> Keypoint:
        """Create a new keypoint."""
        keypoint = Keypoint(
            frame_id=frame_id,
            keypoint_name=keypoint_name,
            x=x,
            y=y,
            z=z,
            confidence=confidence,
            is_occluded=is_occluded,
        )
        self.session.add(keypoint)
        self.session.commit()
        return keypoint

    def bulk_create_keypoints(
        self,
        frame_id: int,
        keypoints_data: List[Dict[str, Any]],
    ) -> List[Keypoint]:
        """
        Create multiple keypoints in bulk.

        Args:
            frame_id: Parent frame ID.
            keypoints_data: List of keypoint dicts with name, x, y, z, confidence.

        Returns:
            List of created Keypoint instances.
        """
        keypoints = [
            Keypoint(
                frame_id=frame_id,
                keypoint_name=kp["name"],
                x=kp["x"],
                y=kp["y"],
                z=kp.get("z"),
                confidence=kp["confidence"],
                is_occluded=kp.get("is_occluded", False),
            )
            for kp in keypoints_data
        ]
        self.session.add_all(keypoints)
        self.session.commit()
        return keypoints

    # ==================== Segmentation Mask Operations ====================

    def create_segmentation_mask(
        self,
        frame_id: int,
        player_role: PlayerRole,
        mask_path: str,
        bbox: Dict[str, float],
        mask_confidence: float,
        text_prompt_used: str,
    ) -> SegmentationMask:
        """
        Create a new segmentation mask record.

        Args:
            frame_id: Associated frame ID.
            player_role: Role of the segmented player.
            mask_path: Path to the mask file.
            bbox: Bounding box dict with x, y, width, height.
            mask_confidence: Confidence score of the segmentation.
            text_prompt_used: Text prompt used for segmentation.

        Returns:
            Created SegmentationMask instance.
        """
        mask = SegmentationMask(
            frame_id=frame_id,
            player_role=player_role,
            mask_path=mask_path,
            player_bbox_x=bbox["x"],
            player_bbox_y=bbox["y"],
            player_bbox_width=bbox["width"],
            player_bbox_height=bbox["height"],
            mask_confidence=mask_confidence,
            text_prompt_used=text_prompt_used,
        )
        self.session.add(mask)
        self.session.commit()
        logger.debug(f"Created segmentation mask: {mask}")
        return mask

    def get_segmentation_masks_for_frame(
        self, frame_id: int, player_role: Optional[PlayerRole] = None
    ) -> List[SegmentationMask]:
        """Get segmentation masks for a frame, optionally filtered by role."""
        query = self.session.query(SegmentationMask).filter(
            SegmentationMask.frame_id == frame_id
        )
        if player_role:
            query = query.filter(SegmentationMask.player_role == player_role)
        return query.all()

    # ==================== Statistics and Utility ====================

    def get_database_stats(self) -> Dict[str, int]:
        """
        Get statistics about the database contents.

        Returns:
            Dictionary with counts of each table.
        """
        stats = {
            "games": self.session.query(func.count(Game.game_pk)).scalar(),
            "players": self.session.query(func.count(Player.player_id)).scalar(),
            "plays": self.session.query(func.count(Play.play_id)).scalar(),
            "plays_with_video": self.session.query(func.count(Play.play_id))
            .filter(Play.video_local_path.isnot(None))
            .scalar(),
            "pose_sequences": self.session.query(
                func.count(PoseSequence.sequence_id)
            ).scalar(),
            "pose_frames": self.session.query(func.count(PoseFrame.frame_id)).scalar(),
            "keypoints": self.session.query(func.count(Keypoint.keypoint_id)).scalar(),
            "segmentation_masks": self.session.query(
                func.count(SegmentationMask.mask_id)
            ).scalar(),
        }
        return stats

    def get_player_stats(self, player_id: int) -> Dict[str, Any]:
        """Get statistics for a specific player."""
        player = self.get_player(player_id)
        if not player:
            return {}

        pitches_thrown = self.session.query(func.count(Play.play_id)).filter(
            Play.pitcher_id == player_id
        ).scalar()

        at_bats = self.session.query(func.count(Play.play_id)).filter(
            Play.batter_id == player_id
        ).scalar()

        pose_sequences = self.session.query(func.count(PoseSequence.sequence_id)).filter(
            PoseSequence.player_id == player_id
        ).scalar()

        return {
            "player_id": player_id,
            "player_name": player.player_name,
            "team": player.team,
            "pitches_thrown": pitches_thrown,
            "at_bats": at_bats,
            "pose_sequences": pose_sequences,
        }
