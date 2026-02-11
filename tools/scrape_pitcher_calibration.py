#!/usr/bin/env python
"""
Scrape pitcher calibration videos: 10 RHP + 10 LHP per stadium per season.

Downloads videos from 2023, 2024, 2025 for all 30 MLB stadiums to build
per-stadium pitcher zone calibration data.

Usage:
    # Full run (all 30 stadiums, 3 seasons)
    python tools/scrape_pitcher_calibration.py

    # Specific stadiums
    python tools/scrape_pitcher_calibration.py --stadiums "Dodger Stadium,Fenway Park"

    # Specific seasons
    python tools/scrape_pitcher_calibration.py --seasons "2024,2025"

    # Dry run
    python tools/scrape_pitcher_calibration.py --dry-run

    # Resume (skips already-downloaded)
    python tools/scrape_pitcher_calibration.py --skip-existing
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.savant import BaseballSavantScraper
from src.scraper.video_downloader import VideoDownloader

# All 30 MLB stadiums with their codes
MLB_STADIUMS = {
    "Angel Stadium": {"code": "1", "home_team": "LAA"},
    "Chase Field": {"code": "15", "home_team": "ARI"},
    "Citi Field": {"code": "3289", "home_team": "NYM"},
    "Citizens Bank Park": {"code": "2681", "home_team": "PHI"},
    "Comerica Park": {"code": "2394", "home_team": "DET"},
    "Coors Field": {"code": "19", "home_team": "COL"},
    "Dodger Stadium": {"code": "22", "home_team": "LAD"},
    "Fenway Park": {"code": "3", "home_team": "BOS"},
    "Globe Life Field": {"code": "5325", "home_team": "TEX"},
    "Great American Ball Park": {"code": "2602", "home_team": "CIN"},
    "Guaranteed Rate Field": {"code": "4", "home_team": "CWS"},
    "Kauffman Stadium": {"code": "7", "home_team": "KC"},
    "LoanDepot Park": {"code": "4169", "home_team": "MIA"},
    "Minute Maid Park": {"code": "2392", "home_team": "HOU"},
    "Nationals Park": {"code": "3309", "home_team": "WSH"},
    "Oakland Coliseum": {"code": "10", "home_team": "OAK"},
    "Oracle Park": {"code": "2395", "home_team": "SF"},
    "Oriole Park": {"code": "2", "home_team": "BAL"},
    "Petco Park": {"code": "2680", "home_team": "SD"},
    "PNC Park": {"code": "31", "home_team": "PIT"},
    "Progressive Field": {"code": "5", "home_team": "CLE"},
    "Rogers Centre": {"code": "14", "home_team": "TOR"},
    "T-Mobile Park": {"code": "680", "home_team": "SEA"},
    "Target Field": {"code": "3312", "home_team": "MIN"},
    "Tropicana Field": {"code": "12", "home_team": "TB"},
    "Truist Park": {"code": "4705", "home_team": "ATL"},
    "Wrigley Field": {"code": "17", "home_team": "CHC"},
    "Yankee Stadium": {"code": "3313", "home_team": "NYY"},
    "American Family Field": {"code": "32", "home_team": "MIL"},
    "Busch Stadium": {"code": "2889", "home_team": "STL"},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "data" / "videos" / "pitcher_calibration"
METADATA_PATH = PROJECT_ROOT / "data" / "pitcher_calibration_metadata.json"

SEASON_DATES = {
    "2023": ("2023-04-01", "2023-09-30"),
    "2024": ("2024-03-28", "2024-09-29"),
    "2025": ("2025-03-27", "2025-09-28"),
}

PITCHERS_PER_HAND = 10  # 10 RHP + 10 LHP = 20 per stadium-season

LOG_PATH = PROJECT_ROOT / "data" / "pitcher_calibration_scraper.log"

# Module-level logger
logger = logging.getLogger("pitcher_calibration")


def setup_logging():
    """Configure logging to both console and file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)

    # File handler — append mode so we keep history across resumes
    fh = logging.FileHandler(str(LOG_PATH), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)


def log(msg: str, level: str = "info"):
    """Log a message to both console and file."""
    getattr(logger, level)(msg)


def log_progress_summary(metadata: dict, stadiums: dict, seasons: list):
    """Write a progress summary to the log.

    Counts how many stadium-seasons are complete and how many videos
    have been downloaded vs the target.
    """
    total_station_seasons = len(stadiums) * len(seasons)
    target_videos = total_station_seasons * PITCHERS_PER_HAND * 2

    completed_ss = 0
    total_videos = 0
    total_rhp = 0
    total_lhp = 0

    for stadium_name in stadiums:
        stadium_key = stadium_name.replace(" ", "_")
        for season in seasons:
            season_key = f"{stadium_key}_{season}"
            if season_key in metadata:
                videos = metadata[season_key]
                n = len(videos)
                total_videos += n
                total_rhp += sum(1 for v in videos if v.get("p_throws") == "R")
                total_lhp += sum(1 for v in videos if v.get("p_throws") == "L")
                if n >= PITCHERS_PER_HAND:  # At least half of target
                    completed_ss += 1

    pct = total_videos / target_videos * 100 if target_videos > 0 else 0

    log("─" * 50)
    log(f"PROGRESS: {total_videos}/{target_videos} videos ({pct:.1f}%)")
    log(f"  Stadium-seasons done: {completed_ss}/{total_station_seasons}")
    log(f"  RHP: {total_rhp}  LHP: {total_lhp}")
    log("─" * 50)


def load_metadata() -> dict:
    """Load existing metadata for resume support."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict):
    """Save metadata to disk."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def search_stadium_season(
    scraper: BaseballSavantScraper,
    stadium_code: str,
    season: str,
) -> pd.DataFrame:
    """Search for pitches at a stadium in a given season."""
    start_date, end_date = SEASON_DATES[season]

    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",
        "hfPR": "",
        "hfZ": "",
        "hfStadium": f"{stadium_code}|",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": f"{season}|",
        "hfSit": "",
        "hfOuts": "",
        "hfOpponent": "",
        "hfInn": "",
        "hfBBT": "",
        "hfFlag": "",
        "hfSA": "",
        "player_type": "pitcher",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "pitch_number_thisgame",
        "sort_order": "desc",
        "min_pas": "0",
        "type": "details",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
    }

    try:
        response = scraper._make_request(scraper.CSV_ENDPOINT, params)
        df = pd.read_csv(pd.io.common.StringIO(response.text), low_memory=False)
        log(f"    Found {len(df)} pitches")
        return df
    except Exception as e:
        log(f"    Error searching: {e}", "error")
        return pd.DataFrame()


def select_diverse_pitchers(
    df: pd.DataFrame,
    hand: str,
    num_pitchers: int = 10,
) -> pd.DataFrame:
    """Select 1 pitch from each of N unique pitchers of given handedness.

    Args:
        df: Statcast DataFrame with p_throws column.
        hand: "R" or "L".
        num_pitchers: Target number of unique pitchers.

    Returns:
        DataFrame with one row per selected pitcher.
    """
    if df.empty or "p_throws" not in df.columns:
        return pd.DataFrame()

    handed = df[df["p_throws"] == hand]
    if handed.empty:
        return pd.DataFrame()

    # Get unique pitchers
    unique_pitchers = handed["pitcher"].unique()
    random.shuffle(unique_pitchers)

    # Take up to num_pitchers unique pitchers
    selected_pitchers = unique_pitchers[:num_pitchers]

    # Pick 1 random pitch per pitcher
    selected_rows = []
    for pid in selected_pitchers:
        pitcher_pitches = handed[handed["pitcher"] == pid]
        row = pitcher_pitches.sample(1).iloc[0]
        selected_rows.append(row)

    return pd.DataFrame(selected_rows)


def download_video(
    scraper: BaseballSavantScraper,
    downloader: VideoDownloader,
    row: pd.Series,
    stadium_dir: Path,
    season: str,
    play_id_cache: dict,
) -> Optional[dict]:
    """Download a single video.

    Returns metadata dict or None on failure.
    """
    game_pk = int(row["game_pk"])
    at_bat = int(row["at_bat_number"])
    pitch_num = int(row["pitch_number"])
    pitcher_id = int(row["pitcher"])
    pitcher_name = str(row.get("player_name", "unknown"))
    p_throws = str(row.get("p_throws", "?"))
    home_team = str(row.get("home_team", "UNK"))
    away_team = str(row.get("away_team", "UNK"))

    video_id = f"{home_team}_{away_team}_{game_pk}_{at_bat}_{pitch_num}"
    video_path = stadium_dir / f"{video_id}.mp4"

    # Skip if already exists
    if video_path.exists() and video_path.stat().st_size > 0:
        log(f"      Already exists: {video_id}.mp4")
        return {
            "video_id": video_id,
            "pitcher_name": pitcher_name,
            "pitcher_id": pitcher_id,
            "p_throws": p_throws,
            "game_pk": game_pk,
            "video_path": str(video_path),
            "season": season,
        }

    # Get video URL
    video_url = scraper.get_video_url_from_statcast_row(row, play_id_cache)
    if not video_url:
        log(f"      No video URL for {video_id}", "warning")
        return None

    # Download video
    log(f"      Downloading: {video_id}")
    try:
        response = downloader.session.get(video_url, timeout=30)
        response.raise_for_status()
        with open(video_path, "wb") as f:
            f.write(response.content)
        time.sleep(0.5)
    except Exception as e:
        log(f"      Download failed: {e}", "error")
        return None

    return {
        "video_id": video_id,
        "pitcher_name": pitcher_name,
        "pitcher_id": pitcher_id,
        "p_throws": p_throws,
        "game_pk": game_pk,
        "video_path": str(video_path),
        "season": season,
    }


def process_stadium_season(
    scraper: BaseballSavantScraper,
    downloader: VideoDownloader,
    stadium_name: str,
    stadium_info: dict,
    season: str,
    metadata: dict,
    skip_existing: bool = True,
) -> List[dict]:
    """Process a single stadium+season combination.

    Returns list of video metadata dicts.
    """
    stadium_key = stadium_name.replace(" ", "_")
    season_key = f"{stadium_key}_{season}"

    # Check if already fully processed
    if skip_existing and season_key in metadata:
        existing = metadata[season_key]
        if len(existing) >= PITCHERS_PER_HAND * 2 * 0.5:  # At least 50% target
            log(f"  Skipping {stadium_name} {season} — already have {len(existing)} videos")
            return existing

    stadium_dir = CALIBRATION_DIR / stadium_key / season
    stadium_dir.mkdir(parents=True, exist_ok=True)

    # Search for pitches
    log(f"  Searching {stadium_name} {season}...")
    df = search_stadium_season(scraper, stadium_info["code"], season)
    if df.empty:
        return []

    time.sleep(1.0)  # Rate limit between searches

    play_id_cache = {}
    all_videos = []

    # Select RHP pitchers
    log(f"    Selecting {PITCHERS_PER_HAND} RHP pitchers...")
    rhp_selected = select_diverse_pitchers(df, "R", PITCHERS_PER_HAND)
    log(f"    Found {len(rhp_selected)} unique RHP")

    for _, row in rhp_selected.iterrows():
        result = download_video(
            scraper, downloader, row, stadium_dir, season, play_id_cache
        )
        if result:
            all_videos.append(result)

    # Select LHP pitchers
    log(f"    Selecting {PITCHERS_PER_HAND} LHP pitchers...")
    lhp_selected = select_diverse_pitchers(df, "L", PITCHERS_PER_HAND)
    log(f"    Found {len(lhp_selected)} unique LHP")

    for _, row in lhp_selected.iterrows():
        result = download_video(
            scraper, downloader, row, stadium_dir, season, play_id_cache
        )
        if result:
            all_videos.append(result)

    # Save progress after each stadium-season
    metadata[season_key] = all_videos
    save_metadata(metadata)

    return all_videos


def main():
    parser = argparse.ArgumentParser(
        description="Scrape pitcher calibration videos (10 RHP + 10 LHP × 30 stadiums × 3 seasons)"
    )
    parser.add_argument(
        "--stadiums", type=str,
        help="Comma-separated list of stadium names to process",
    )
    parser.add_argument(
        "--seasons", type=str, default="2023,2024,2025",
        help="Comma-separated list of seasons (default: 2023,2024,2025)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip stadiums/seasons already downloaded",
    )
    args = parser.parse_args()

    setup_logging()

    # Parse seasons
    seasons = [s.strip() for s in args.seasons.split(",")]
    for s in seasons:
        if s not in SEASON_DATES:
            log(f"ERROR: Unknown season '{s}'. Valid: {list(SEASON_DATES.keys())}", "error")
            return

    # Filter stadiums if specified
    if args.stadiums:
        stadium_filter = [s.strip() for s in args.stadiums.split(",")]
        stadiums = {k: v for k, v in MLB_STADIUMS.items() if k in stadium_filter}
        if not stadiums:
            log(f"ERROR: No matching stadiums found. Available: {list(MLB_STADIUMS.keys())}", "error")
            return
    else:
        stadiums = MLB_STADIUMS

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    total_expected = len(stadiums) * len(seasons) * PITCHERS_PER_HAND * 2

    log("=" * 70)
    log("PITCHER CALIBRATION DATA SCRAPER")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log(f"Stadiums:       {len(stadiums)}")
    log(f"Seasons:        {seasons}")
    log(f"Per hand/season: {PITCHERS_PER_HAND}")
    log(f"Total expected: ~{total_expected} videos")
    log(f"Output dir:     {CALIBRATION_DIR}")
    log(f"Log file:       {LOG_PATH}")
    log("=" * 70)

    if args.dry_run:
        log("\n[DRY RUN MODE]\n")
        for stadium_name in stadiums:
            for season in seasons:
                log(f"  Would process: {stadium_name} {season} "
                    f"(10 RHP + 10 LHP)")
        log(f"\nTotal: {total_expected} videos to download")
        return

    # Load existing metadata for resume
    metadata = load_metadata()

    # Log initial progress if resuming
    if metadata:
        log("\nResuming from previous run...")
        log_progress_summary(metadata, stadiums, seasons)

    scraper = BaseballSavantScraper(request_delay=2.0, timeout=60)
    downloader = VideoDownloader(download_dir=str(CALIBRATION_DIR))

    stats = {
        "stadiums_processed": 0,
        "seasons_processed": 0,
        "videos_total": 0,
        "rhp_total": 0,
        "lhp_total": 0,
    }

    try:
        for stadium_name, stadium_info in stadiums.items():
            for season in seasons:
                log(f"\n{'=' * 70}")
                log(f"  {stadium_name} — {season}")
                log(f"{'=' * 70}")

                videos = process_stadium_season(
                    scraper, downloader,
                    stadium_name, stadium_info, season,
                    metadata, skip_existing=args.skip_existing,
                )

                rhp_count = sum(1 for v in videos if v.get("p_throws") == "R")
                lhp_count = sum(1 for v in videos if v.get("p_throws") == "L")

                stats["videos_total"] += len(videos)
                stats["rhp_total"] += rhp_count
                stats["lhp_total"] += lhp_count
                stats["seasons_processed"] += 1

                log(f"  Result: {rhp_count} RHP + {lhp_count} LHP = {len(videos)} videos")

                # Log cumulative progress after each stadium-season
                log_progress_summary(metadata, stadiums, seasons)

            stats["stadiums_processed"] += 1

    except KeyboardInterrupt:
        log("\n\nInterrupted! Progress saved to metadata.", "warning")
        save_metadata(metadata)

    finally:
        scraper.close()
        downloader.close()

    log(f"\n{'=' * 70}")
    log("COMPLETE!")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'=' * 70}")
    log(f"Stadiums processed: {stats['stadiums_processed']}")
    log(f"Season-stadiums:    {stats['seasons_processed']}")
    log(f"Total videos:       {stats['videos_total']}")
    log(f"  RHP:              {stats['rhp_total']}")
    log(f"  LHP:              {stats['lhp_total']}")
    log(f"Metadata:           {METADATA_PATH}")
    log(f"\nNext step: python tools/calibrate_pitcher_zones.py")


if __name__ == "__main__":
    main()
