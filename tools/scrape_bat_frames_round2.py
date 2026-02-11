#!/usr/bin/env python
"""
Scrape bat detection training frames ROUND 2 - 2023 Season.

Downloads videos from 2023 (different from existing 2024 data) for more diverse
bat detection training data. Focuses on frames where batter is in stance.
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
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

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
VIDEOS_DIR = PROJECT_ROOT / "data/videos/2023_round2"
FRAMES_DIR = PROJECT_ROOT / "data/bat_frames_round2"

# 2023 Season dates
START_DATE = "2023-04-01"
END_DATE = "2023-09-30"

VIDEOS_PER_STADIUM = 3  # 3 videos per stadium = 90 total videos
FRAMES_PER_VIDEO = 3    # 3 frames per video focusing on stance/load phase


def search_stadium_pitches(
    scraper: BaseballSavantScraper,
    stadium_code: str,
    home_team: str,
) -> pd.DataFrame:
    """Search for pitches at a specific stadium in 2023."""

    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",  # Regular season
        "hfPR": "",
        "hfZ": "",
        "hfStadium": f"{stadium_code}|",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "2023|",  # 2023 SEASON
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
        "game_date_gt": START_DATE,
        "game_date_lt": END_DATE,
    }

    try:
        response = scraper._make_request(scraper.CSV_ENDPOINT, params)
        df = pd.read_csv(pd.io.common.StringIO(response.text), low_memory=False)
        print(f"  Found {len(df)} pitches")
        return df.head(200)  # Limit to avoid huge datasets
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


def select_diverse_pitches(df: pd.DataFrame, num_videos: int = 3) -> pd.DataFrame:
    """Select pitches from different games for diversity."""
    if df.empty:
        return df

    # Shuffle first to avoid bias from sorting
    df = df.sample(frac=1).reset_index(drop=True)

    # Try to get pitches from different games
    if "game_pk" in df.columns:
        games = df["game_pk"].unique()
        selected = []
        for game in games[:num_videos]:
            game_pitches = df[df["game_pk"] == game]
            if len(game_pitches) > 0:
                selected.append(game_pitches.iloc[0])
        if len(selected) >= num_videos:
            return pd.DataFrame(selected[:num_videos])
        # If not enough games, fill with random samples
        while len(selected) < num_videos and len(df) > len(selected):
            remaining = df[~df["game_pk"].isin([s["game_pk"] for s in selected])]
            if len(remaining) == 0:
                remaining = df
            selected.append(remaining.iloc[0])
        return pd.DataFrame(selected)

    return df.head(num_videos)


def extract_stance_frames(
    video_path: str,
    output_dir: Path,
    num_frames: int = 3,
    video_id: str = "",
) -> List[str]:
    """Extract frames from early portion of video where batter is in stance."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 20:
        cap.release()
        return []

    # Focus on first 40% of video (stance/load phase before swing)
    # This is where the bat is most visible and static
    start_frame = int(total_frames * 0.10)  # Skip first 10% (may be loading)
    end_frame = int(total_frames * 0.40)    # Before swing starts

    if end_frame - start_frame < num_frames:
        end_frame = int(total_frames * 0.50)

    frame_indices = []
    step = (end_frame - start_frame) // (num_frames + 1)
    for i in range(1, num_frames + 1):
        frame_indices.append(start_frame + i * step)

    saved_frames = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"{video_id}_f{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(str(frame_path))

    cap.release()
    return saved_frames


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape bat frames round 2 (2023 season)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without actually downloading")
    parser.add_argument("--stadiums", type=str, help="Comma-separated list of stadium names to process")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip stadiums that already have videos")
    args = parser.parse_args()

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Filter stadiums if specified
    if args.stadiums:
        stadium_filter = [s.strip() for s in args.stadiums.split(",")]
        stadiums = {k: v for k, v in MLB_STADIUMS.items() if k in stadium_filter}
    else:
        stadiums = MLB_STADIUMS

    print("=" * 60)
    print("BAT DETECTION FRAMES - ROUND 2 (2023 Season)")
    print("=" * 60)
    print(f"Stadiums: {len(stadiums)}")
    print(f"Videos per stadium: {VIDEOS_PER_STADIUM}")
    print(f"Frames per video: {FRAMES_PER_VIDEO}")
    print(f"Total expected: ~{len(stadiums) * VIDEOS_PER_STADIUM * FRAMES_PER_VIDEO} frames")
    print(f"Season: 2023 ({START_DATE} to {END_DATE})")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No downloads will occur]\n")
        return

    scraper = BaseballSavantScraper(request_delay=3.0, timeout=60)  # Longer timeout for 2023 data
    downloader = VideoDownloader(download_dir=str(VIDEOS_DIR))

    all_frames = []
    frames_info = []
    stats = {"stadiums_processed": 0, "videos_downloaded": 0, "frames_extracted": 0}

    for stadium_name, stadium_info in stadiums.items():
        print(f"\n{'='*60}")
        print(f"Stadium: {stadium_name} ({stadium_info['home_team']})")
        print(f"{'='*60}")

        stadium_code = stadium_info["code"]
        home_team = stadium_info["home_team"]

        stadium_dir = VIDEOS_DIR / stadium_name.replace(" ", "_")
        stadium_dir.mkdir(exist_ok=True)

        # Check if we already have enough videos for this stadium
        existing_videos = list(stadium_dir.glob("*.mp4"))
        if args.skip_existing and len(existing_videos) >= VIDEOS_PER_STADIUM:
            print(f"  Already have {len(existing_videos)} videos, skipping download...")

            # Just extract frames from existing
            for video_path in existing_videos[:VIDEOS_PER_STADIUM]:
                video_id = video_path.stem
                extracted = extract_stance_frames(
                    str(video_path), FRAMES_DIR, FRAMES_PER_VIDEO, video_id
                )
                for frame_path in extracted:
                    frames_info.append({
                        "path": frame_path,
                        "stadium": stadium_name,
                        "home_team": home_team,
                        "video_id": video_id,
                        "season": "2023",
                    })
                all_frames.extend(extracted)
                stats["frames_extracted"] += len(extracted)
            stats["stadiums_processed"] += 1
            continue

        # Search for pitches at this stadium
        df = search_stadium_pitches(scraper, stadium_code, home_team)

        if df.empty:
            print(f"  No pitches found, skipping...")
            continue

        # Select diverse pitches
        selected = select_diverse_pitches(df, VIDEOS_PER_STADIUM)

        if selected.empty:
            print(f"  Could not select pitches, skipping...")
            continue

        # Get video URLs and download
        print(f"  Downloading {len(selected)} videos...")
        play_id_cache = {}

        for idx, row in selected.iterrows():
            game_pk = int(row["game_pk"])
            at_bat = int(row["at_bat_number"])
            pitch_num = int(row["pitch_number"])
            away_team = row.get("away_team", "UNK")

            # Get video URL
            video_url = scraper.get_video_url_from_statcast_row(row, play_id_cache)

            if not video_url:
                print(f"    No video URL for {game_pk}_{at_bat}_{pitch_num}")
                continue

            # Download video
            video_id = f"{home_team}_{away_team}_{game_pk}_{at_bat}_{pitch_num}"
            video_path = stadium_dir / f"{video_id}.mp4"

            if video_path.exists():
                print(f"    Already exists: {video_id}")
            else:
                print(f"    Downloading: {video_id}")
                try:
                    response = downloader.session.get(video_url, timeout=30)
                    response.raise_for_status()
                    with open(video_path, "wb") as f:
                        f.write(response.content)
                    stats["videos_downloaded"] += 1
                    time.sleep(1.5)  # Rate limit
                except Exception as e:
                    print(f"    Download failed: {e}")
                    continue

            # Extract stance frames
            extracted = extract_stance_frames(
                str(video_path), FRAMES_DIR, FRAMES_PER_VIDEO, video_id
            )

            for frame_path in extracted:
                frames_info.append({
                    "path": frame_path,
                    "stadium": stadium_name,
                    "home_team": home_team,
                    "away_team": away_team,
                    "video_id": video_id,
                    "season": "2023",
                })

            all_frames.extend(extracted)
            stats["frames_extracted"] += len(extracted)
            print(f"    Extracted {len(extracted)} stance frames")

        stats["stadiums_processed"] += 1

    # Save frames info
    frames_info_path = FRAMES_DIR / "frames_info_round2.json"
    with open(frames_info_path, "w") as f:
        json.dump(frames_info, f, indent=2)

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Stadiums processed: {stats['stadiums_processed']}")
    print(f"Videos downloaded: {stats['videos_downloaded']}")
    print(f"Frames extracted: {stats['frames_extracted']}")
    print(f"Frames saved to: {FRAMES_DIR}")
    print(f"Frames info: {frames_info_path}")
    print(f"\nNext step: Label bat keypoints using bat_keypoint_labeler.py")

    scraper.close()
    downloader.close()


if __name__ == "__main__":
    main()
