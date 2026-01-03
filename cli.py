#!/usr/bin/env python3
"""Command-line interface for baseball biomechanics analysis."""

import sys
from pathlib import Path

import click
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return {}


@click.group()
@click.option(
    "--config",
    "-c",
    default="config/config.yaml",
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """Baseball Biomechanics Analysis System.

    Analyze baseball player biomechanics using video segmentation
    and pose estimation.
    """
    ctx.ensure_object(dict)

    # Load config
    cfg = load_config(config)
    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option(
    "--player-id",
    "-p",
    required=True,
    type=int,
    help="MLB player ID",
)
@click.option(
    "--start-date",
    "-s",
    required=True,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    "-e",
    required=True,
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--player-type",
    "-t",
    type=click.Choice(["pitcher", "batter"]),
    default="pitcher",
    help="Type of player",
)
@click.option(
    "--download/--no-download",
    default=True,
    help="Download videos after scraping",
)
@click.pass_context
def scrape(
    ctx: click.Context,
    player_id: int,
    start_date: str,
    end_date: str,
    player_type: str,
    download: bool,
) -> None:
    """Scrape videos and Statcast data from Baseball Savant."""
    # Lazy import to avoid loading broken modules
    from src.utils.logging_config import setup_logging
    from src.pipeline.orchestrator import BiomechanicsPipeline, PipelineConfig

    cfg = ctx.obj["config"]
    log_level = "DEBUG" if ctx.obj["verbose"] else cfg.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)

    pipeline_config = PipelineConfig(
        database_url=cfg.get("database", {}).get("url", "sqlite:///data/baseball_biomechanics.db"),
        video_dir=cfg.get("scraper", {}).get("video_download_dir", "data/videos"),
        request_delay=cfg.get("scraper", {}).get("request_delay_seconds", 2.0),
    )

    with BiomechanicsPipeline(pipeline_config) as pipeline:
        click.echo(f"Scraping data for player {player_id}...")
        plays = pipeline.scrape_player(
            player_id=player_id,
            start_date=start_date,
            end_date=end_date,
            player_type=player_type,
        )
        click.echo(f"Scraped {len(plays)} plays")

        if download and plays:
            click.echo("Downloading videos...")
            downloaded = pipeline.download_videos(plays=plays)
            click.echo(f"Downloaded {len(downloaded)} videos")


@cli.command()
@click.option(
    "--play-id",
    "-p",
    required=True,
    type=int,
    help="Play ID to segment",
)
@click.option(
    "--roles",
    "-r",
    default="pitcher,batter,catcher",
    help="Comma-separated player roles to segment",
)
@click.option(
    "--custom-prompt",
    help="Custom text prompt for segmentation",
)
@click.pass_context
def segment(
    ctx: click.Context,
    play_id: int,
    roles: str,
    custom_prompt: str,
) -> None:
    """Run SAM 3 segmentation on a video."""
    from src.utils.logging_config import setup_logging
    from src.pipeline.orchestrator import BiomechanicsPipeline, PipelineConfig

    cfg = ctx.obj["config"]
    log_level = "DEBUG" if ctx.obj["verbose"] else cfg.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)

    role_list = [r.strip() for r in roles.split(",")]
    custom_prompts = cfg.get("segmentation", {}).get("default_prompts", {})

    pipeline_config = PipelineConfig(
        database_url=cfg.get("database", {}).get("url", "sqlite:///data/baseball_biomechanics.db"),
        mask_dir=cfg.get("segmentation", {}).get("output_dir", "data/masks"),
        roles=role_list,
        custom_prompts=custom_prompts,
    )

    with BiomechanicsPipeline(pipeline_config) as pipeline:
        play = pipeline.db_ops.get_play(play_id)
        if not play:
            click.echo(f"Error: Play {play_id} not found", err=True)
            return

        if not play.video_local_path:
            click.echo(f"Error: No video file for play {play_id}", err=True)
            return

        click.echo(f"Segmenting video: {play.video_local_path}")
        click.echo(f"Roles: {role_list}")

        results = pipeline.segment_video(
            play_id=play_id,
            video_path=play.video_local_path,
            roles=role_list,
            custom_prompt=custom_prompt,
        )

        for role, seg_results in results.items():
            click.echo(f"  {role}: {len(seg_results)} frames segmented")


@cli.command()
@click.option(
    "--play-id",
    "-p",
    required=True,
    type=int,
    help="Play ID for pose estimation",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["mediapipe", "motionbert"]),
    default="mediapipe",
    help="Pose estimation backend",
)
@click.option(
    "--roles",
    "-r",
    default="pitcher,batter,catcher",
    help="Comma-separated player roles to process",
)
@click.pass_context
def pose(
    ctx: click.Context,
    play_id: int,
    backend: str,
    roles: str,
) -> None:
    """Run pose estimation on a video."""
    from src.utils.logging_config import setup_logging
    from src.pipeline.orchestrator import BiomechanicsPipeline, PipelineConfig

    cfg = ctx.obj["config"]
    log_level = "DEBUG" if ctx.obj["verbose"] else cfg.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)

    role_list = [r.strip() for r in roles.split(",")]

    pipeline_config = PipelineConfig(
        database_url=cfg.get("database", {}).get("url", "sqlite:///data/baseball_biomechanics.db"),
        pose_backend=backend,
        roles=role_list,
    )

    with BiomechanicsPipeline(pipeline_config) as pipeline:
        play = pipeline.db_ops.get_play(play_id)
        if not play:
            click.echo(f"Error: Play {play_id} not found", err=True)
            return

        if not play.video_local_path:
            click.echo(f"Error: No video file for play {play_id}", err=True)
            return

        click.echo(f"Running pose estimation on: {play.video_local_path}")
        click.echo(f"Backend: {backend}")

        results = pipeline.estimate_pose(
            play_id=play_id,
            video_path=play.video_local_path,
            roles=role_list,
        )

        for role, pose_results in results.items():
            valid = sum(1 for r in pose_results if r.is_valid)
            click.echo(f"  {role}: {valid}/{len(pose_results)} frames with valid pose")


@cli.command()
@click.option(
    "--player-id",
    "-p",
    required=True,
    type=int,
    help="MLB player ID",
)
@click.option(
    "--start-date",
    "-s",
    required=True,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    "-e",
    required=True,
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--roles",
    "-r",
    default="pitcher,batter",
    help="Comma-separated player roles to process",
)
@click.option(
    "--player-type",
    "-t",
    type=click.Choice(["pitcher", "batter"]),
    default="pitcher",
    help="Type of player for scraping",
)
@click.pass_context
def pipeline(
    ctx: click.Context,
    player_id: int,
    start_date: str,
    end_date: str,
    roles: str,
    player_type: str,
) -> None:
    """Run the full analysis pipeline."""
    from src.utils.logging_config import setup_logging
    from src.pipeline.orchestrator import BiomechanicsPipeline, PipelineConfig

    cfg = ctx.obj["config"]
    log_level = "DEBUG" if ctx.obj["verbose"] else cfg.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)

    role_list = [r.strip() for r in roles.split(",")]

    pipeline_config = PipelineConfig(
        database_url=cfg.get("database", {}).get("url", "sqlite:///data/baseball_biomechanics.db"),
        video_dir=cfg.get("scraper", {}).get("video_download_dir", "data/videos"),
        mask_dir=cfg.get("segmentation", {}).get("output_dir", "data/masks"),
        roles=role_list,
        pose_backend=cfg.get("pose", {}).get("default_backend", "mediapipe"),
        request_delay=cfg.get("scraper", {}).get("request_delay_seconds", 2.0),
    )

    click.echo(f"Running full pipeline for player {player_id}")
    click.echo(f"Date range: {start_date} to {end_date}")
    click.echo(f"Roles: {role_list}")
    click.echo()

    with BiomechanicsPipeline(pipeline_config) as pipe:
        progress = pipe.run_full_pipeline(
            player_id=player_id,
            start_date=start_date,
            end_date=end_date,
            player_type=player_type,
            roles=role_list,
        )

        click.echo()
        click.echo("Pipeline Results:")
        click.echo(f"  Plays scraped: {progress.scraped}")
        click.echo(f"  Videos downloaded: {progress.downloaded}")
        click.echo(f"  Videos segmented: {progress.segmented}")
        click.echo(f"  Pose estimated: {progress.pose_estimated}")
        click.echo(f"  Failed: {progress.failed}")


@cli.group()
def db() -> None:
    """Database management commands."""
    pass


@db.command("init")
@click.pass_context
def db_init(ctx: click.Context) -> None:
    """Initialize the database (create tables)."""
    from src.database.schema import init_db

    cfg = ctx.obj["config"]
    db_url = cfg.get("database", {}).get("url", "sqlite:///data/baseball_biomechanics.db")

    click.echo(f"Initializing database: {db_url}")
    init_db(db_url)
    click.echo("Database initialized successfully!")


@db.command("stats")
@click.pass_context
def db_stats(ctx: click.Context) -> None:
    """Show database statistics."""
    from src.database.schema import get_engine, get_session_factory
    from src.database.operations import DatabaseOperations

    cfg = ctx.obj["config"]
    db_url = cfg.get("database", {}).get("url", "sqlite:///data/baseball_biomechanics.db")

    engine = get_engine(db_url)
    SessionLocal = get_session_factory(engine)
    session = SessionLocal()

    try:
        ops = DatabaseOperations(session)
        stats = ops.get_database_stats()

        click.echo("Database Statistics:")
        click.echo(f"  Games: {stats['games']}")
        click.echo(f"  Players: {stats['players']}")
        click.echo(f"  Plays: {stats['plays']}")
        click.echo(f"  Plays with video: {stats['plays_with_video']}")
        click.echo(f"  Pose sequences: {stats['pose_sequences']}")
        click.echo(f"  Pose frames: {stats['pose_frames']}")
        click.echo(f"  Keypoints: {stats['keypoints']}")
        click.echo(f"  Segmentation masks: {stats['segmentation_masks']}")
    finally:
        session.close()


@db.command("player")
@click.option(
    "--player-id",
    "-p",
    required=True,
    type=int,
    help="Player ID to get stats for",
)
@click.pass_context
def db_player(ctx: click.Context, player_id: int) -> None:
    """Show statistics for a specific player."""
    from src.database.schema import get_engine, get_session_factory
    from src.database.operations import DatabaseOperations

    cfg = ctx.obj["config"]
    db_url = cfg.get("database", {}).get("url", "sqlite:///data/baseball_biomechanics.db")

    engine = get_engine(db_url)
    SessionLocal = get_session_factory(engine)
    session = SessionLocal()

    try:
        ops = DatabaseOperations(session)
        stats = ops.get_player_stats(player_id)

        if not stats:
            click.echo(f"Player {player_id} not found", err=True)
            return

        click.echo(f"Player: {stats['player_name']} (ID: {stats['player_id']})")
        click.echo(f"  Team: {stats['team'] or 'N/A'}")
        click.echo(f"  Pitches thrown: {stats['pitches_thrown']}")
        click.echo(f"  At bats: {stats['at_bats']}")
        click.echo(f"  Pose sequences: {stats['pose_sequences']}")
    finally:
        session.close()


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    click.echo("Baseball Biomechanics Analysis System")
    click.echo("Version: 0.1.0")


if __name__ == "__main__":
    cli()
