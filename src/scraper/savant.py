"""Baseball Savant scraper for Statcast data and video URLs."""

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# MLB Stats API endpoints
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1.1"
SPORTY_VIDEOS_URL = "https://baseballsavant.mlb.com/sporty-videos"


class BaseballSavantScraper:
    """
    Scraper for Baseball Savant Statcast data and video URLs.

    This class handles fetching pitch-level data from Baseball Savant,
    including video URLs for individual plays.
    """

    BASE_URL = "https://baseballsavant.mlb.com"
    STATCAST_SEARCH_URL = f"{BASE_URL}/statcast_search"
    CSV_ENDPOINT = f"{STATCAST_SEARCH_URL}/csv"

    # Column mapping for CSV data to our database schema
    COLUMN_MAPPING = {
        "game_pk": "game_pk",
        "game_date": "game_date",
        "pitcher": "pitcher_id",
        "batter": "batter_id",
        "catcher": "catcher_id",
        "inning": "inning",
        "at_bat_number": "at_bat_number",
        "pitch_number": "pitch_number",
        "pitch_type": "pitch_type",
        "release_speed": "release_speed",
        "release_spin_rate": "spin_rate",
        "release_pos_x": "release_pos_x",
        "release_pos_z": "release_pos_z",
        "pfx_x": "pfx_x",
        "pfx_z": "pfx_z",
        "plate_x": "plate_x",
        "plate_z": "plate_z",
        "zone": "zone",
        "launch_speed": "launch_speed",
        "launch_angle": "launch_angle",
        "hit_distance_sc": "hit_distance",
        "events": "events",
        "des": "description",
        "home_team": "home_team",
        "away_team": "away_team",
        "player_name": "pitcher_name",
    }

    def __init__(
        self,
        request_delay: float = 2.0,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize the Baseball Savant scraper.

        Args:
            request_delay: Delay between requests in seconds.
            max_retries: Maximum number of retry attempts for failed requests.
            timeout: Request timeout in seconds.
        """
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        self._last_request_time: Optional[float] = None

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> requests.Response:
        """
        Make an HTTP request with retry logic and rate limiting.

        Args:
            url: URL to request.
            params: Query parameters.
            method: HTTP method.

        Returns:
            Response object.

        Raises:
            requests.RequestException: If request fails after all retries.
        """
        self._rate_limit()

        for attempt in range(self.max_retries):
            try:
                if method == "GET":
                    response = self.session.get(
                        url, params=params, timeout=self.timeout
                    )
                else:
                    response = self.session.post(
                        url, data=params, timeout=self.timeout
                    )
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_delay * (attempt + 1))
                else:
                    raise

    def search_statcast(
        self,
        player_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        player_type: str = "pitcher",
        pitch_type: Optional[str] = None,
        min_pitches: int = 0,
        team: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Search Baseball Savant for Statcast data.

        Args:
            player_id: MLB player ID to filter by.
            start_date: Start date (YYYY-MM-DD format).
            end_date: End date (YYYY-MM-DD format).
            player_type: "pitcher" or "batter".
            pitch_type: Filter by pitch type (FF, SL, CH, etc.).
            min_pitches: Minimum number of pitches.
            team: Team abbreviation to filter by.

        Returns:
            DataFrame containing Statcast data.
        """
        params = {
            "all": "true",
            "hfPT": pitch_type or "",
            "hfAB": "",
            "hfGT": "R|",  # Regular season
            "hfPR": "",
            "hfZ": "",
            "hfStadium": "",
            "hfBBL": "",
            "hfNewZones": "",
            "hfPull": "",
            "hfC": "",
            "hfSea": "",
            "hfSit": "",
            "hfOuts": "",
            "hfOpponent": "",
            "hfInn": "",
            "hfBBT": "",
            "hfFlag": "",
            "hfSA": "",
            "player_type": player_type,
            "min_pitches": str(min_pitches),
            "min_results": "0",
            "group_by": "name",
            "sort_col": "pitches",
            "player_event_sort": "pitch_number_thisgame",
            "sort_order": "desc",
            "min_pas": "0",
            "type": "details",
        }

        if player_id:
            if player_type == "pitcher":
                params["pitchers_lookup[]"] = str(player_id)
            else:
                params["batters_lookup[]"] = str(player_id)

        if start_date:
            params["game_date_gt"] = start_date
        if end_date:
            params["game_date_lt"] = end_date
        if team:
            params["hfTeam"] = f"{team}|"

        logger.info(
            f"Searching Statcast: player_id={player_id}, "
            f"dates={start_date} to {end_date}, type={player_type}"
        )

        try:
            response = self._make_request(self.CSV_ENDPOINT, params)
            df = pd.read_csv(
                pd.io.common.StringIO(response.text),
                low_memory=False,
            )
            logger.info(f"Retrieved {len(df)} pitches from Statcast")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch Statcast data: {e}")
            return pd.DataFrame()

    def get_game_play_ids(self, game_pk: int) -> Dict[Tuple[int, int], str]:
        """
        Get playId UUIDs for all pitches in a game from MLB Stats API.

        Args:
            game_pk: MLB game primary key.

        Returns:
            Dictionary mapping (at_bat_number, pitch_number) to playId UUID.
        """
        url = f"{MLB_STATS_API_BASE}/game/{game_pk}/feed/live"
        play_id_map: Dict[Tuple[int, int], str] = {}

        try:
            response = self._make_request(url)
            data = response.json()

            plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

            for play in plays:
                # at_bat_number is 1-indexed in Statcast, atBatIndex is 0-indexed in API
                at_bat_number = play.get("atBatIndex", 0) + 1

                events = play.get("playEvents", [])
                pitch_count = 0

                for event in events:
                    if event.get("type") == "pitch":
                        pitch_count += 1
                        play_id = event.get("playId")
                        if play_id:
                            play_id_map[(at_bat_number, pitch_count)] = play_id

            logger.debug(f"Found {len(play_id_map)} playIds for game {game_pk}")
            return play_id_map

        except Exception as e:
            logger.error(f"Failed to get play IDs for game {game_pk}: {e}")
            return {}

    def get_video_url_from_play_id(self, play_id: str) -> Optional[str]:
        """
        Get the direct video URL from a playId UUID.

        Args:
            play_id: The playId UUID from MLB Stats API.

        Returns:
            Direct MP4 video URL if found, None otherwise.
        """
        import html

        params = {"playId": play_id}

        try:
            response = self._make_request(SPORTY_VIDEOS_URL, params)

            # Look for sporty-clips.mlb.com MP4 URL in the page
            # The URL may contain HTML entities that need to be decoded
            mp4_pattern = r'(https://sporty-clips\.mlb\.com/[^"\s<>]+\.mp4)'
            match = re.search(mp4_pattern, response.text)

            if match:
                video_url = match.group(1)
                # Decode HTML entities (e.g., &#x3D; -> =)
                video_url = html.unescape(video_url)
                logger.debug(f"Found video URL for playId {play_id[:8]}...")
                return video_url

            logger.debug(f"No video URL found for playId {play_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to get video URL for playId {play_id}: {e}")
            return None

    def get_video_url(self, game_pk: int, play_id: str) -> Optional[str]:
        """
        Get the video URL for a specific play (legacy method).

        Args:
            game_pk: MLB game primary key.
            play_id: Play identifier (format: game_pk_at_bat_pitch).

        Returns:
            Video URL if found, None otherwise.
        """
        # Parse the legacy play_id format
        parts = play_id.split("_")
        if len(parts) != 3:
            logger.warning(f"Invalid play_id format: {play_id}")
            return None

        try:
            at_bat_number = int(parts[1])
            pitch_number = int(parts[2])
        except ValueError:
            logger.warning(f"Invalid play_id format: {play_id}")
            return None

        # Get the playId UUID from the Stats API
        play_id_map = self.get_game_play_ids(game_pk)
        uuid = play_id_map.get((at_bat_number, pitch_number))

        if not uuid:
            logger.warning(f"No playId UUID found for {play_id}")
            return None

        return self.get_video_url_from_play_id(uuid)

    def get_video_url_from_statcast_row(
        self,
        row: pd.Series,
        play_id_cache: Optional[Dict[int, Dict[Tuple[int, int], str]]] = None,
    ) -> Optional[str]:
        """
        Get video URL from a Statcast data row using MLB Stats API playId.

        Args:
            row: A row from Statcast DataFrame.
            play_id_cache: Optional cache of game_pk -> {(ab, pitch): playId}.
                          If provided, uses cached playIds instead of fetching.

        Returns:
            Video URL if found, None otherwise.
        """
        try:
            game_pk = int(row.get("game_pk", 0))
            at_bat_number = int(row.get("at_bat_number", 0))
            pitch_number = int(row.get("pitch_number", 0))

            if not all([game_pk, at_bat_number, pitch_number]):
                return None

            # Get playId UUID from cache or fetch
            if play_id_cache and game_pk in play_id_cache:
                play_id_map = play_id_cache[game_pk]
            else:
                play_id_map = self.get_game_play_ids(game_pk)
                if play_id_cache is not None:
                    play_id_cache[game_pk] = play_id_map

            play_id_uuid = play_id_map.get((at_bat_number, pitch_number))
            if not play_id_uuid:
                logger.debug(
                    f"No playId for game={game_pk}, ab={at_bat_number}, pitch={pitch_number}"
                )
                return None

            # Get the actual video URL
            return self.get_video_url_from_play_id(play_id_uuid)

        except Exception as e:
            logger.error(f"Failed to get video URL: {e}")
            return None

    def get_video_urls_for_dataframe(
        self,
        df: pd.DataFrame,
        max_per_game: Optional[int] = None,
    ) -> Dict[int, str]:
        """
        Efficiently get video URLs for all plays in a Statcast DataFrame.

        Fetches playIds per game (single API call per game) then retrieves
        video URLs for each pitch.

        Args:
            df: Statcast DataFrame with game_pk, at_bat_number, pitch_number.
            max_per_game: Optional limit on videos per game (for testing).

        Returns:
            Dictionary mapping DataFrame index to video URL.
        """
        video_urls: Dict[int, str] = {}
        play_id_cache: Dict[int, Dict[Tuple[int, int], str]] = {}

        # Group by game for efficient API calls
        games = df["game_pk"].unique()
        logger.info(f"Fetching video URLs for {len(games)} games, {len(df)} pitches")

        for game_pk in games:
            game_df = df[df["game_pk"] == game_pk]

            # Fetch playIds for this game (single API call)
            play_id_map = self.get_game_play_ids(int(game_pk))
            play_id_cache[int(game_pk)] = play_id_map

            # Get video URLs for each pitch in this game
            count = 0
            for idx, row in game_df.iterrows():
                if max_per_game and count >= max_per_game:
                    break

                at_bat = int(row.get("at_bat_number", 0))
                pitch = int(row.get("pitch_number", 0))

                play_id_uuid = play_id_map.get((at_bat, pitch))
                if play_id_uuid:
                    video_url = self.get_video_url_from_play_id(play_id_uuid)
                    if video_url:
                        video_urls[idx] = video_url
                        count += 1

            logger.info(
                f"Game {game_pk}: found {count} video URLs"
            )

        logger.info(f"Total: found {len(video_urls)} video URLs")
        return video_urls

    def process_statcast_data(
        self,
        df: pd.DataFrame,
        fetch_video_urls: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process Statcast DataFrame and yield play dictionaries.

        Args:
            df: Statcast DataFrame from search_statcast.
            fetch_video_urls: Whether to attempt fetching video URLs.

        Yields:
            Dictionary for each play with standardized field names.
        """
        # Pre-fetch all video URLs efficiently (one API call per game)
        video_url_map: Dict[int, str] = {}
        play_id_cache: Dict[int, Dict[Tuple[int, int], str]] = {}

        if fetch_video_urls and len(df) > 0:
            # Build play_id cache for all games
            games = df["game_pk"].unique()
            for game_pk in games:
                play_id_cache[int(game_pk)] = self.get_game_play_ids(int(game_pk))

        for idx, row in df.iterrows():
            play_data = {}

            # Map columns to our schema
            for csv_col, db_col in self.COLUMN_MAPPING.items():
                if csv_col in row:
                    value = row[csv_col]
                    # Handle NaN values
                    if pd.isna(value):
                        value = None
                    play_data[db_col] = value

            # Convert game_date to datetime
            if play_data.get("game_date"):
                play_data["game_date"] = pd.to_datetime(
                    play_data["game_date"]
                ).to_pydatetime()

            # Ensure integer IDs
            for id_field in ["game_pk", "pitcher_id", "batter_id", "catcher_id"]:
                if play_data.get(id_field):
                    play_data[id_field] = int(play_data[id_field])

            # Get video URL from cache
            if fetch_video_urls:
                play_data["video_url"] = self.get_video_url_from_statcast_row(
                    row, play_id_cache
                )

            yield play_data

    def get_player_info(self, player_id: int) -> Optional[Dict[str, Any]]:
        """
        Get player information from Baseball Savant.

        Args:
            player_id: MLB player ID.

        Returns:
            Dictionary with player info or None if not found.
        """
        url = f"{self.BASE_URL}/player/{player_id}"

        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.text, "lxml")

            # Extract player name from page
            name_elem = soup.find("h1", class_="player-name")
            if not name_elem:
                name_elem = soup.find("div", class_="player-header")

            player_name = name_elem.get_text(strip=True) if name_elem else "Unknown"

            # Extract additional info
            player_info = {
                "player_id": player_id,
                "player_name": player_name,
                "team": None,
                "position": None,
                "throws": None,
                "bats": None,
            }

            # Look for bio info
            bio_section = soup.find("div", class_="bio")
            if bio_section:
                bio_text = bio_section.get_text()

                # Parse throws/bats
                if "B/T:" in bio_text:
                    bt_match = re.search(r"B/T:\s*(\w)/(\w)", bio_text)
                    if bt_match:
                        player_info["bats"] = bt_match.group(1)
                        player_info["throws"] = bt_match.group(2)

                # Parse team
                team_elem = soup.find("span", class_="team-name")
                if team_elem:
                    player_info["team"] = team_elem.get_text(strip=True)

            logger.debug(f"Retrieved player info: {player_info}")
            return player_info

        except Exception as e:
            logger.error(f"Failed to get player info for {player_id}: {e}")
            return None

    def get_game_info(self, game_pk: int) -> Optional[Dict[str, Any]]:
        """
        Get game information from Baseball Savant.

        Args:
            game_pk: MLB game primary key.

        Returns:
            Dictionary with game info or None if not found.
        """
        url = f"{self.BASE_URL}/gamefeed"
        params = {"gamePk": game_pk}

        try:
            response = self._make_request(url, params)
            soup = BeautifulSoup(response.text, "lxml")

            game_info = {
                "game_pk": game_pk,
                "game_date": None,
                "home_team": None,
                "away_team": None,
                "venue": None,
            }

            # Extract game header info
            header = soup.find("div", class_="game-header")
            if header:
                teams = header.find_all("span", class_="team-name")
                if len(teams) >= 2:
                    game_info["away_team"] = teams[0].get_text(strip=True)
                    game_info["home_team"] = teams[1].get_text(strip=True)

                date_elem = header.find("span", class_="game-date")
                if date_elem:
                    date_str = date_elem.get_text(strip=True)
                    try:
                        game_info["game_date"] = datetime.strptime(
                            date_str, "%B %d, %Y"
                        )
                    except ValueError:
                        pass

                venue_elem = header.find("span", class_="venue")
                if venue_elem:
                    game_info["venue"] = venue_elem.get_text(strip=True)

            logger.debug(f"Retrieved game info: {game_info}")
            return game_info

        except Exception as e:
            logger.error(f"Failed to get game info for {game_pk}: {e}")
            return None

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
        logger.debug("Scraper session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
