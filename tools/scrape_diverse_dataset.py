"""
Scrape diverse dataset from all MLB stadiums for YOLO training.
Downloads 10 videos per stadium with team diversity, extracts random frames.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# All MLB team abbreviations for diversity
ALL_TEAMS = [
    "LAA", "ARI", "NYM", "PHI", "DET", "COL", "LAD", "BOS", "TEX", "CIN",
    "CWS", "KC", "MIA", "HOU", "WSH", "OAK", "SF", "BAL", "SD", "PIT",
    "CLE", "TOR", "SEA", "MIN", "TB", "ATL", "CHC", "NYY", "MIL", "STL"
]


def search_stadium_pitches(
    scraper: BaseballSavantScraper,
    stadium_code: str,
    home_team: str,
    start_date: str = "2024-04-01",
    end_date: str = "2024-09-30",
    max_results: int = 500,
) -> pd.DataFrame:
    """Search for pitches at a specific stadium."""

    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",  # Regular season
        "hfPR": "",
        "hfZ": "",
        "hfStadium": f"{stadium_code}|",  # Filter by stadium
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "2024|",
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
        print(f"  Found {len(df)} pitches at stadium {stadium_code}")
        return df.head(max_results)
    except Exception as e:
        print(f"  Error searching stadium {stadium_code}: {e}")
        return pd.DataFrame()


def select_diverse_pitches(
    df: pd.DataFrame,
    home_team: str,
    num_videos: int = 10,
) -> pd.DataFrame:
    """Select pitches with diverse visiting teams."""

    if df.empty:
        return df

    # Get unique away teams (visiting teams at this stadium)
    if "away_team" not in df.columns:
        # Just return random sample if no away_team info
        return df.sample(n=min(num_videos, len(df)))

    away_teams = df[df["away_team"] != home_team]["away_team"].unique()

    selected = []
    teams_used = set()

    # First, try to get one pitch per visiting team
    for team in away_teams:
        if len(selected) >= num_videos:
            break
        team_pitches = df[df["away_team"] == team]
        if len(team_pitches) > 0 and team not in teams_used:
            selected.append(team_pitches.sample(n=1).iloc[0])
            teams_used.add(team)

    # If we need more, sample randomly from remaining
    remaining_needed = num_videos - len(selected)
    if remaining_needed > 0:
        remaining = df[~df.index.isin([s.name for s in selected])]
        if len(remaining) > 0:
            extra = remaining.sample(n=min(remaining_needed, len(remaining)))
            selected.extend([row for _, row in extra.iterrows()])

    result = pd.DataFrame(selected)
    print(f"  Selected {len(result)} pitches from {len(teams_used)} different visiting teams")
    return result


def extract_random_frames(
    video_path: str,
    output_dir: Path,
    num_frames: int = 2,
    video_id: str = "",
) -> List[str]:
    """Extract random frames from a video."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        cap.release()
        return []

    # Select random frames from middle portion (skip first/last 10%)
    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)

    frame_indices = random.sample(range(start_frame, end_frame), min(num_frames, end_frame - start_frame))
    frame_indices.sort()

    saved_frames = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"{video_id}_frame{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(str(frame_path))

    cap.release()
    return saved_frames


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape diverse MLB stadium dataset")
    parser.add_argument("--project-dir", default="F:/Claude_Projects/baseball-biomechanics")
    parser.add_argument("--videos-per-stadium", type=int, default=10)
    parser.add_argument("--frames-per-video", type=int, default=2)
    parser.add_argument("--start-date", default="2024-04-01")
    parser.add_argument("--end-date", default="2024-09-30")
    parser.add_argument("--skip-download", action="store_true", help="Skip video download, only extract frames")
    parser.add_argument("--stadiums", type=str, help="Comma-separated list of stadium names to process (default: all)")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    videos_dir = project_dir / "data" / "videos" / "diverse_stadiums"
    frames_dir = project_dir / "data" / "labels" / "diverse_frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Filter stadiums if specified
    if args.stadiums:
        stadium_filter = [s.strip() for s in args.stadiums.split(",")]
        stadiums = {k: v for k, v in MLB_STADIUMS.items() if k in stadium_filter}
    else:
        stadiums = MLB_STADIUMS

    print(f"=" * 60)
    print(f"Scraping diverse dataset from {len(stadiums)} MLB stadiums")
    print(f"Videos per stadium: {args.videos_per_stadium}")
    print(f"Frames per video: {args.frames_per_video}")
    print(f"Total expected frames: {len(stadiums) * args.videos_per_stadium * args.frames_per_video}")
    print(f"=" * 60)

    scraper = BaseballSavantScraper(request_delay=2.0)
    downloader = VideoDownloader(download_dir=str(videos_dir))

    all_frames = []
    frames_info = []

    for stadium_name, stadium_info in stadiums.items():
        print(f"\n{'='*60}")
        print(f"Processing: {stadium_name} ({stadium_info['home_team']})")
        print(f"{'='*60}")

        stadium_code = stadium_info["code"]
        home_team = stadium_info["home_team"]
        stadium_dir = videos_dir / stadium_name.replace(" ", "_")
        stadium_dir.mkdir(exist_ok=True)

        if not args.skip_download:
            # Search for pitches at this stadium
            df = search_stadium_pitches(
                scraper, stadium_code, home_team,
                args.start_date, args.end_date
            )

            if df.empty:
                print(f"  No pitches found, skipping...")
                continue

            # Select diverse pitches
            selected = select_diverse_pitches(df, home_team, args.videos_per_stadium)

            if selected.empty:
                print(f"  Could not select pitches, skipping...")
                continue

            # Get video URLs and download
            print(f"  Fetching video URLs...")
            play_id_cache = {}

            for idx, row in selected.iterrows():
                game_pk = int(row["game_pk"])
                at_bat = int(row["at_bat_number"])
                pitch_num = int(row["pitch_number"])
                game_date = str(row.get("game_date", ""))[:10]
                away_team = row.get("away_team", "UNK")

                # Get video URL
                video_url = scraper.get_video_url_from_statcast_row(row, play_id_cache)

                if not video_url:
                    print(f"    No video URL for {game_pk}_{at_bat}_{pitch_num}")
                    continue

                # Download video
                video_id = f"{stadium_info['home_team']}_{away_team}_{game_pk}_{at_bat}_{pitch_num}"
                video_path = stadium_dir / f"{video_id}.mp4"

                if video_path.exists():
                    print(f"    Video already exists: {video_id}")
                else:
                    print(f"    Downloading: {video_id}")
                    try:
                        response = downloader.session.get(video_url, timeout=30)
                        response.raise_for_status()
                        with open(video_path, "wb") as f:
                            f.write(response.content)
                        time.sleep(1)  # Rate limit
                    except Exception as e:
                        print(f"    Download failed: {e}")
                        continue

                # Extract frames
                extracted = extract_random_frames(
                    str(video_path), frames_dir,
                    args.frames_per_video, video_id
                )

                for frame_path in extracted:
                    frames_info.append({
                        "path": frame_path,
                        "stadium": stadium_name,
                        "home_team": home_team,
                        "away_team": away_team,
                        "video_id": video_id,
                    })

                all_frames.extend(extracted)
                print(f"    Extracted {len(extracted)} frames")

        else:
            # Just extract frames from existing videos
            existing_videos = list(stadium_dir.glob("*.mp4"))
            print(f"  Found {len(existing_videos)} existing videos")

            for video_path in existing_videos[:args.videos_per_stadium]:
                video_id = video_path.stem
                extracted = extract_random_frames(
                    str(video_path), frames_dir,
                    args.frames_per_video, video_id
                )

                for frame_path in extracted:
                    frames_info.append({
                        "path": frame_path,
                        "stadium": stadium_name,
                        "home_team": home_team,
                        "video_id": video_id,
                    })

                all_frames.extend(extracted)

    # Save frames info for labeling
    frames_info_path = frames_dir / "frames_info.json"
    with open(frames_info_path, "w") as f:
        json.dump(frames_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Total frames extracted: {len(all_frames)}")
    print(f"Frames info saved to: {frames_info_path}")
    print(f"\nNext step: Run the mitt labeler on these frames:")
    print(f"  python tools/mitt_labeler.py --frames-dir {frames_dir}")

    scraper.close()
    downloader.close()


if __name__ == "__main__":
    main()
