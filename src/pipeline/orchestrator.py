"""Pipeline orchestrator for end-to-end biomechanics analysis."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from src.database.models import PlayerRole
from src.database.operations import DatabaseOperations
from src.database.schema import get_engine, get_session_factory
from src.pose.base import PoseBackend, PoseResult
from src.pose.mediapipe_backend import MediaPipeBackend
from src.scraper.savant import BaseballSavantScraper
from src.scraper.video_downloader import VideoDownloader
from src.segmentation.sam3_tracker import SAM3Tracker, PlayerSegmentationResult
from src.utils.logging_config import get_logger
from src.utils.video_utils import VideoProcessor

logger = get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    SCRAPE = "scrape"
    DOWNLOAD = "download"
    SEGMENT = "segment"
    POSE = "pose"
    ALL = "all"


@dataclass
class PipelineConfig:
    """
    Configuration for the biomechanics pipeline.

    Attributes:
        database_url: Database connection URL.
        video_dir: Directory for downloaded videos.
        mask_dir: Directory for segmentation masks.
        processed_dir: Directory for intermediate outputs.
        roles: Player roles to process.
        pose_backend: Pose estimation backend name.
        custom_prompts: Custom segmentation prompts by role.
        save_intermediate: Whether to save intermediate outputs.
        skip_existing: Whether to skip already processed items.
    """
    database_url: str = "sqlite:///data/baseball_biomechanics.db"
    video_dir: str = "data/videos"
    mask_dir: str = "data/masks"
    processed_dir: str = "data/processed"
    roles: List[str] = field(default_factory=lambda: ["pitcher", "batter", "catcher"])
    pose_backend: str = "mediapipe"
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    save_intermediate: bool = True
    skip_existing: bool = True
    request_delay: float = 2.0


@dataclass
class PipelineProgress:
    """
    Track progress of a pipeline run.

    Attributes:
        total_plays: Total number of plays to process.
        scraped: Number of plays scraped.
        downloaded: Number of videos downloaded.
        segmented: Number of videos segmented.
        pose_estimated: Number of videos with pose estimation.
        failed: Number of failed items.
        current_stage: Current processing stage.
    """
    total_plays: int = 0
    scraped: int = 0
    downloaded: int = 0
    segmented: int = 0
    pose_estimated: int = 0
    failed: int = 0
    current_stage: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_plays": self.total_plays,
            "scraped": self.scraped,
            "downloaded": self.downloaded,
            "segmented": self.segmented,
            "pose_estimated": self.pose_estimated,
            "failed": self.failed,
            "current_stage": self.current_stage,
            "error_count": len(self.errors),
        }


class BiomechanicsPipeline:
    """
    End-to-end pipeline for baseball biomechanics analysis.

    Orchestrates scraping, video download, segmentation, and pose estimation
    with progress tracking and resume capability.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self.progress = PipelineProgress()

        # Initialize components (lazy loading)
        self._scraper: Optional[BaseballSavantScraper] = None
        self._downloader: Optional[VideoDownloader] = None
        self._segmenter: Optional[SAM3Tracker] = None
        self._pose_backend: Optional[PoseBackend] = None
        self._db_ops: Optional[DatabaseOperations] = None
        self._session = None

        # Create directories
        Path(self.config.video_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.mask_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.processed_dir).mkdir(parents=True, exist_ok=True)

    @property
    def scraper(self) -> BaseballSavantScraper:
        """Get or create the scraper instance."""
        if self._scraper is None:
            self._scraper = BaseballSavantScraper(
                request_delay=self.config.request_delay
            )
        return self._scraper

    @property
    def downloader(self) -> VideoDownloader:
        """Get or create the downloader instance."""
        if self._downloader is None:
            self._downloader = VideoDownloader(
                download_dir=self.config.video_dir,
                request_delay=self.config.request_delay,
            )
        return self._downloader

    @property
    def segmenter(self) -> SAM3Tracker:
        """Get or create the segmenter instance."""
        if self._segmenter is None:
            self._segmenter = SAM3Tracker(
                output_dir=self.config.mask_dir,
            )
            if self.config.custom_prompts:
                self._segmenter.set_custom_prompts(self.config.custom_prompts)
        return self._segmenter

    @property
    def pose_backend(self) -> PoseBackend:
        """Get or create the pose estimation backend."""
        if self._pose_backend is None:
            if self.config.pose_backend == "mediapipe":
                self._pose_backend = MediaPipeBackend()
            else:
                raise ValueError(f"Unknown pose backend: {self.config.pose_backend}")
            self._pose_backend.initialize()
        return self._pose_backend

    @property
    def db_ops(self) -> DatabaseOperations:
        """Get or create the database operations instance."""
        if self._db_ops is None:
            engine = get_engine(self.config.database_url)
            SessionLocal = get_session_factory(engine)
            self._session = SessionLocal()
            self._db_ops = DatabaseOperations(self._session)
        return self._db_ops

    def run_full_pipeline(
        self,
        player_id: int,
        start_date: str,
        end_date: str,
        player_type: str = "pitcher",
        roles: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> PipelineProgress:
        """
        Run the full pipeline for a player.

        Args:
            player_id: MLB player ID.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            player_type: "pitcher" or "batter".
            roles: Player roles to segment/analyze.
            show_progress: Whether to show progress bars.

        Returns:
            PipelineProgress with results.
        """
        roles = roles or self.config.roles
        self.progress = PipelineProgress()
        self.progress.current_stage = "scraping"

        logger.info(
            f"Starting pipeline for player {player_id} "
            f"from {start_date} to {end_date}"
        )

        try:
            # Stage 1: Scrape
            plays = self.scrape_player(
                player_id=player_id,
                start_date=start_date,
                end_date=end_date,
                player_type=player_type,
                show_progress=show_progress,
            )
            self.progress.scraped = len(plays)
            self.progress.total_plays = len(plays)
            logger.info(f"Scraped {len(plays)} plays")

            # Stage 2: Download videos
            self.progress.current_stage = "downloading"
            downloaded = self.download_videos(
                plays=plays,
                show_progress=show_progress,
            )
            self.progress.downloaded = len(downloaded)
            logger.info(f"Downloaded {len(downloaded)} videos")

            # Stage 3: Segment and estimate pose for each video
            self.progress.current_stage = "processing"
            for play_id, video_path in tqdm(
                downloaded.items(),
                desc="Processing videos",
                disable=not show_progress,
            ):
                try:
                    # Segment
                    seg_results = self.segment_video(
                        play_id=int(play_id.split("_")[0]) if "_" in play_id else int(play_id),
                        video_path=video_path,
                        roles=roles,
                        show_progress=False,
                    )
                    if seg_results:
                        self.progress.segmented += 1

                    # Pose estimation
                    pose_results = self.estimate_pose(
                        play_id=int(play_id.split("_")[0]) if "_" in play_id else int(play_id),
                        video_path=video_path,
                        segmentation_results=seg_results,
                        roles=roles,
                        show_progress=False,
                    )
                    if pose_results:
                        self.progress.pose_estimated += 1

                except Exception as e:
                    self.progress.failed += 1
                    self.progress.errors.append(f"Play {play_id}: {str(e)}")
                    logger.error(f"Error processing play {play_id}: {e}")

            self.progress.current_stage = "complete"
            logger.info(
                f"Pipeline complete: {self.progress.pose_estimated}/{self.progress.total_plays} "
                f"fully processed, {self.progress.failed} failed"
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.progress.errors.append(str(e))
            raise

        return self.progress

    def scrape_player(
        self,
        player_id: int,
        start_date: str,
        end_date: str,
        player_type: str = "pitcher",
        fetch_video_urls: bool = True,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Scrape Statcast data for a player.

        Args:
            player_id: MLB player ID.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            player_type: "pitcher" or "batter".
            fetch_video_urls: Whether to fetch video URLs.
            show_progress: Whether to show progress.

        Returns:
            List of play dictionaries.
        """
        logger.info(f"Scraping data for player {player_id}")

        # Get Statcast data
        df = self.scraper.search_statcast(
            player_id=player_id,
            start_date=start_date,
            end_date=end_date,
            player_type=player_type,
        )

        if df.empty:
            logger.warning(f"No data found for player {player_id}")
            return []

        # Process and store plays
        plays = []
        for play_data in self.scraper.process_statcast_data(df, fetch_video_urls):
            try:
                # Ensure required players exist
                pitcher_id = play_data.get("pitcher_id")
                batter_id = play_data.get("batter_id")

                if pitcher_id:
                    self.db_ops.get_or_create_player(
                        player_id=pitcher_id,
                        player_name=play_data.get("pitcher_name", f"Player {pitcher_id}"),
                    )

                if batter_id:
                    self.db_ops.get_or_create_player(
                        player_id=batter_id,
                        player_name=f"Player {batter_id}",
                    )

                # Ensure game exists
                game_pk = play_data.get("game_pk")
                game_date = play_data.get("game_date")
                if game_pk and game_date:
                    self.db_ops.get_or_create_game(
                        game_pk=game_pk,
                        game_date=game_date,
                        home_team=play_data.get("home_team", ""),
                        away_team=play_data.get("away_team", ""),
                    )

                # Create play record
                play = self.db_ops.create_play(
                    game_pk=game_pk,
                    inning=play_data.get("inning", 0),
                    at_bat_number=play_data.get("at_bat_number", 0),
                    pitch_number=play_data.get("pitch_number", 0),
                    pitcher_id=pitcher_id,
                    batter_id=batter_id,
                    catcher_id=play_data.get("catcher_id"),
                    pitch_type=play_data.get("pitch_type"),
                    release_speed=play_data.get("release_speed"),
                    spin_rate=play_data.get("spin_rate"),
                    release_pos_x=play_data.get("release_pos_x"),
                    release_pos_z=play_data.get("release_pos_z"),
                    pfx_x=play_data.get("pfx_x"),
                    pfx_z=play_data.get("pfx_z"),
                    plate_x=play_data.get("plate_x"),
                    plate_z=play_data.get("plate_z"),
                    zone=play_data.get("zone"),
                    launch_speed=play_data.get("launch_speed"),
                    launch_angle=play_data.get("launch_angle"),
                    hit_distance=play_data.get("hit_distance"),
                    events=play_data.get("events"),
                    description=play_data.get("description"),
                    video_url=play_data.get("video_url"),
                )

                play_data["play_id"] = play.play_id
                plays.append(play_data)

            except Exception as e:
                logger.error(f"Error storing play: {e}")
                continue

        return plays

    def download_videos(
        self,
        plays: Optional[List[Dict]] = None,
        play_ids: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> Dict[str, str]:
        """
        Download videos for plays.

        Args:
            plays: List of play dictionaries (from scrape).
            play_ids: List of play IDs (from database).
            show_progress: Whether to show progress.

        Returns:
            Dictionary mapping play identifiers to local paths.
        """
        if plays is None and play_ids is None:
            # Get plays without videos from database
            db_plays = self.db_ops.get_plays_without_video()
            videos_to_download = [
                {
                    "url": p.video_url,
                    "game_pk": p.game_pk,
                    "at_bat_number": p.at_bat_number,
                    "pitch_number": p.pitch_number,
                    "game_date": p.game.game_date.strftime("%Y-%m-%d") if p.game else None,
                    "play_id": p.play_id,
                }
                for p in db_plays
                if p.video_url
            ]
        elif plays:
            videos_to_download = [
                {
                    "url": p.get("video_url"),
                    "game_pk": p.get("game_pk"),
                    "at_bat_number": p.get("at_bat_number"),
                    "pitch_number": p.get("pitch_number"),
                    "game_date": p.get("game_date").strftime("%Y-%m-%d")
                    if isinstance(p.get("game_date"), datetime) else p.get("game_date"),
                    "play_id": p.get("play_id"),
                }
                for p in plays
                if p.get("video_url")
            ]
        else:
            videos_to_download = []
            for pid in play_ids:
                play = self.db_ops.get_play(pid)
                if play and play.video_url:
                    videos_to_download.append({
                        "url": play.video_url,
                        "game_pk": play.game_pk,
                        "at_bat_number": play.at_bat_number,
                        "pitch_number": play.pitch_number,
                        "game_date": play.game.game_date.strftime("%Y-%m-%d") if play.game else None,
                        "play_id": play.play_id,
                    })

        if not videos_to_download:
            logger.info("No videos to download")
            return {}

        logger.info(f"Downloading {len(videos_to_download)} videos")

        # Download
        results = self.downloader.download_batch(
            videos_to_download,
            show_progress=show_progress,
        )

        # Update database with local paths
        for play_key, local_path in results.items():
            # Find play_id from videos_to_download
            for v in videos_to_download:
                key = f"{v['game_pk']}_{v['at_bat_number']}_{v['pitch_number']}"
                if key == play_key and v.get("play_id"):
                    self.db_ops.update_play_video_path(v["play_id"], local_path)
                    break

        return results

    def segment_video(
        self,
        play_id: int,
        video_path: str,
        roles: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None,
        show_progress: bool = True,
        multi_player: bool = True,
    ) -> Dict[str, List[PlayerSegmentationResult]]:
        """
        Segment players in a video.

        Args:
            play_id: Play ID for database linking.
            video_path: Path to the video file.
            roles: Player roles to segment.
            custom_prompt: Optional custom prompt for all roles.
            show_progress: Whether to show progress.
            multi_player: If True, use position-based multi-player segmentation.

        Returns:
            Dictionary mapping roles to segmentation results.
        """
        roles = roles or self.config.roles

        logger.info(f"Segmenting video: {video_path} for roles: {roles}")

        if multi_player:
            # Use new multi-player segmentation (position-based)
            results = self.segmenter.segment_video_all_players(
                video_path=video_path,
                show_progress=show_progress,
            )
            # Filter to requested roles
            return {role: results.get(role, []) for role in roles}
        else:
            # Legacy single-mask approach
            if custom_prompt:
                custom_prompts = {role: custom_prompt for role in roles}
                self.segmenter.set_custom_prompts(custom_prompts)

            result = self.segmenter.segment_video(
                video_path=video_path,
                roles=roles,
                show_progress=show_progress,
                save_masks=self.config.save_intermediate,
            )
            return result.player_results

    def estimate_pose(
        self,
        play_id: int,
        video_path: str,
        segmentation_results: Optional[Dict[str, List[PlayerSegmentationResult]]] = None,
        roles: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Dict[str, List[PoseResult]]:
        """
        Estimate pose for players in a video.

        Args:
            play_id: Play ID for database linking.
            video_path: Path to the video file.
            segmentation_results: Optional pre-computed segmentation results.
            roles: Player roles to process.
            show_progress: Whether to show progress.

        Returns:
            Dictionary mapping roles to pose results.
        """
        roles = roles or self.config.roles
        results: Dict[str, List[PoseResult]] = {}

        # Get video info for timestamp calculation
        with VideoProcessor(video_path) as vp:
            fps = vp.info.fps if vp.info else 30.0

        # If we have segmentation results, use cropped frames
        if segmentation_results:
            for role in roles:
                role_seg = segmentation_results.get(role, [])
                if not role_seg:
                    continue

                # Get cropped frames from segmentation
                cropped_frames = self.segmenter.get_cropped_frames(
                    video_path,
                    role_seg,
                    padding=20,
                )

                if cropped_frames:
                    pose_results = self.pose_backend.process_frames(
                        cropped_frames,
                        fps=fps,
                        show_progress=show_progress,
                    )
                    results[role] = pose_results

                    # Store in database
                    self._store_pose_results(
                        play_id=play_id,
                        role=role,
                        pose_results=pose_results,
                        fps=fps,
                    )
        else:
            # Process full video
            pose_results = self.pose_backend.process_video(
                video_path,
                show_progress=show_progress,
            )

            # For full video, we assign to a generic role
            results["full"] = pose_results

            # Store results for each role based on the play
            play = self.db_ops.get_play(play_id)
            if play:
                # Store for pitcher if available
                if "pitcher" in roles and play.pitcher_id:
                    self._store_pose_results(
                        play_id=play_id,
                        role="pitcher",
                        pose_results=pose_results,
                        fps=fps,
                    )
                # Store for batter if available
                elif "batter" in roles and play.batter_id:
                    self._store_pose_results(
                        play_id=play_id,
                        role="batter",
                        pose_results=pose_results,
                        fps=fps,
                    )

        return results

    def _store_pose_results(
        self,
        play_id: int,
        role: str,
        pose_results: List[PoseResult],
        fps: float,
    ) -> None:
        """Store pose results in the database."""
        if not pose_results:
            return

        # Get play to find player IDs
        play = self.db_ops.get_play(play_id)
        if not play:
            logger.warning(f"Play {play_id} not found")
            return

        # Determine player ID based on role
        if role == "pitcher":
            player_id = play.pitcher_id
            player_role = PlayerRole.PITCHER
        elif role == "batter":
            player_id = play.batter_id
            player_role = PlayerRole.BATTER
        elif role == "catcher":
            player_id = play.catcher_id
            player_role = PlayerRole.CATCHER
        else:
            logger.warning(f"Unknown role: {role}")
            return

        if not player_id:
            logger.warning(f"No player ID for role {role}")
            return

        # Get frame range
        valid_results = [r for r in pose_results if r.is_valid]
        if not valid_results:
            return

        start_frame = min(r.frame_number for r in valid_results)
        end_frame = max(r.frame_number for r in valid_results)

        # Create pose sequence
        sequence = self.db_ops.create_pose_sequence(
            play_id=play_id,
            player_id=player_id,
            player_role=player_role,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            model_used=self.config.pose_backend,
        )

        # Create frames and keypoints
        for pose_result in valid_results:
            frame = self.db_ops.create_pose_frame(
                sequence_id=sequence.sequence_id,
                frame_number=pose_result.frame_number,
                timestamp_ms=pose_result.timestamp_ms,
            )

            # Bulk create keypoints
            keypoints_data = [kp.to_dict() for kp in pose_result.keypoints]
            self.db_ops.bulk_create_keypoints(frame.frame_id, keypoints_data)

        logger.debug(
            f"Stored {len(valid_results)} frames for {role} in play {play_id}"
        )

    def run_stage(
        self,
        stage: PipelineStage,
        player_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        play_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Any:
        """
        Run a specific pipeline stage.

        Args:
            stage: Pipeline stage to run.
            player_id: MLB player ID (for scrape/download).
            start_date: Start date (for scrape).
            end_date: End date (for scrape).
            play_ids: Specific play IDs (for segment/pose).
            **kwargs: Additional arguments for the stage.

        Returns:
            Stage-specific results.
        """
        if stage == PipelineStage.SCRAPE:
            if not all([player_id, start_date, end_date]):
                raise ValueError("Scrape requires player_id, start_date, end_date")
            return self.scrape_player(
                player_id=player_id,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )

        elif stage == PipelineStage.DOWNLOAD:
            return self.download_videos(play_ids=play_ids, **kwargs)

        elif stage == PipelineStage.SEGMENT:
            if not play_ids:
                raise ValueError("Segment requires play_ids")
            results = {}
            for pid in play_ids:
                play = self.db_ops.get_play(pid)
                if play and play.video_local_path:
                    results[pid] = self.segment_video(
                        play_id=pid,
                        video_path=play.video_local_path,
                        **kwargs,
                    )
            return results

        elif stage == PipelineStage.POSE:
            if not play_ids:
                raise ValueError("Pose requires play_ids")
            results = {}
            for pid in play_ids:
                play = self.db_ops.get_play(pid)
                if play and play.video_local_path:
                    results[pid] = self.estimate_pose(
                        play_id=pid,
                        video_path=play.video_local_path,
                        **kwargs,
                    )
            return results

        elif stage == PipelineStage.ALL:
            if not all([player_id, start_date, end_date]):
                raise ValueError("Full pipeline requires player_id, start_date, end_date")
            return self.run_full_pipeline(
                player_id=player_id,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._scraper:
            self._scraper.close()
        if self._downloader:
            self._downloader.close()
        if self._pose_backend:
            self._pose_backend.cleanup()
        if self._session:
            self._session.close()

        logger.debug("Pipeline resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
