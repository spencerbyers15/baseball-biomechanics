"""Video downloader for Baseball Savant videos."""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoDownloader:
    """
    Handler for downloading and managing baseball videos.

    Supports resumable downloads, progress tracking, and organized storage.
    """

    def __init__(
        self,
        download_dir: str = "data/videos",
        chunk_size: int = 8192,
        request_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize the video downloader.

        Args:
            download_dir: Directory to store downloaded videos.
            chunk_size: Chunk size for streaming downloads.
            request_delay: Delay between download requests.
            max_retries: Maximum retry attempts for failed downloads.
            timeout: Request timeout in seconds.
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        })
        self._last_download_time: Optional[float] = None

    def _rate_limit(self) -> None:
        """Enforce rate limiting between downloads."""
        if self._last_download_time is not None:
            elapsed = time.time() - self._last_download_time
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        self._last_download_time = time.time()

    def _get_video_path(
        self,
        game_pk: int,
        at_bat_number: int,
        pitch_number: int,
        game_date: Optional[str] = None,
    ) -> Path:
        """
        Generate the local file path for a video.

        Organizes videos by game date for easier browsing.

        Args:
            game_pk: MLB game primary key.
            at_bat_number: At-bat number within the game.
            pitch_number: Pitch number within the at-bat.
            game_date: Game date string (YYYY-MM-DD format).

        Returns:
            Path object for the video file.
        """
        if game_date:
            # Organize by year/month
            date_parts = game_date.split("-")
            if len(date_parts) >= 2:
                year_month = f"{date_parts[0]}/{date_parts[1]}"
                subdir = self.download_dir / year_month
            else:
                subdir = self.download_dir / "unknown_date"
        else:
            subdir = self.download_dir / "unknown_date"

        subdir.mkdir(parents=True, exist_ok=True)

        filename = f"{game_pk}_{at_bat_number}_{pitch_number}.mp4"
        return subdir / filename

    def _get_url_hash(self, url: str) -> str:
        """Generate a hash of the URL for tracking purposes."""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def is_downloaded(
        self,
        game_pk: int,
        at_bat_number: int,
        pitch_number: int,
        game_date: Optional[str] = None,
    ) -> Tuple[bool, Optional[Path]]:
        """
        Check if a video has already been downloaded.

        Args:
            game_pk: MLB game primary key.
            at_bat_number: At-bat number within the game.
            pitch_number: Pitch number within the at-bat.
            game_date: Game date string.

        Returns:
            Tuple of (is_downloaded, file_path).
        """
        video_path = self._get_video_path(
            game_pk, at_bat_number, pitch_number, game_date
        )
        if video_path.exists() and video_path.stat().st_size > 0:
            return True, video_path
        return False, video_path

    def download_video(
        self,
        url: str,
        game_pk: int,
        at_bat_number: int,
        pitch_number: int,
        game_date: Optional[str] = None,
        force: bool = False,
        show_progress: bool = True,
    ) -> Optional[str]:
        """
        Download a video from Baseball Savant.

        Args:
            url: Video URL to download.
            game_pk: MLB game primary key.
            at_bat_number: At-bat number within the game.
            pitch_number: Pitch number within the at-bat.
            game_date: Game date string for organizing files.
            force: If True, re-download even if file exists.
            show_progress: Whether to show download progress bar.

        Returns:
            Local file path if successful, None otherwise.
        """
        is_downloaded, video_path = self.is_downloaded(
            game_pk, at_bat_number, pitch_number, game_date
        )

        if is_downloaded and not force:
            logger.debug(f"Video already exists: {video_path}")
            return str(video_path)

        self._rate_limit()

        for attempt in range(self.max_retries):
            try:
                # Check if we can resume a partial download
                resume_pos = 0
                temp_path = video_path.with_suffix(".mp4.part")
                headers = {}

                if temp_path.exists():
                    resume_pos = temp_path.stat().st_size
                    headers["Range"] = f"bytes={resume_pos}-"
                    logger.debug(f"Resuming download from byte {resume_pos}")

                # Make request
                response = self.session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                )

                # Handle response
                if response.status_code == 416:
                    # Range not satisfiable - file complete
                    if temp_path.exists():
                        temp_path.rename(video_path)
                        logger.info(f"Downloaded: {video_path}")
                        return str(video_path)

                response.raise_for_status()

                # Get file size for progress bar
                content_length = response.headers.get("Content-Length")
                total_size = int(content_length) if content_length else None
                if resume_pos > 0 and response.status_code == 206:
                    # Partial content - add existing size
                    total_size = (total_size or 0) + resume_pos

                # Download with progress
                mode = "ab" if resume_pos > 0 and response.status_code == 206 else "wb"
                bytes_downloaded = resume_pos if mode == "ab" else 0

                pbar = None
                if show_progress and total_size:
                    pbar = tqdm(
                        total=total_size,
                        initial=bytes_downloaded,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {video_path.name}",
                    )

                try:
                    with open(temp_path, mode) as f:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                bytes_downloaded += len(chunk)
                                if pbar:
                                    pbar.update(len(chunk))
                finally:
                    if pbar:
                        pbar.close()

                # Rename completed download
                temp_path.rename(video_path)
                logger.info(f"Downloaded: {video_path}")
                return str(video_path)

            except requests.RequestException as e:
                logger.warning(
                    f"Download failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to download video: {url}")
                    return None

            except IOError as e:
                logger.error(f"File I/O error during download: {e}")
                return None

        return None

    def download_batch(
        self,
        videos: List[Dict],
        show_progress: bool = True,
    ) -> Dict[str, str]:
        """
        Download multiple videos in batch.

        Args:
            videos: List of dicts with url, game_pk, at_bat_number, pitch_number, game_date.
            show_progress: Whether to show overall progress.

        Returns:
            Dictionary mapping play IDs to local file paths.
        """
        results = {}
        total = len(videos)
        skipped = 0
        downloaded = 0
        failed = 0

        iterator = tqdm(videos, desc="Downloading videos") if show_progress else videos

        for video_info in iterator:
            url = video_info.get("url")
            game_pk = video_info.get("game_pk")
            at_bat_number = video_info.get("at_bat_number")
            pitch_number = video_info.get("pitch_number")
            game_date = video_info.get("game_date")

            if not all([url, game_pk, at_bat_number, pitch_number]):
                logger.warning(f"Skipping video with missing info: {video_info}")
                failed += 1
                continue

            play_id = f"{game_pk}_{at_bat_number}_{pitch_number}"

            # Check if already downloaded
            is_exists, _ = self.is_downloaded(
                game_pk, at_bat_number, pitch_number, game_date
            )
            if is_exists:
                results[play_id] = str(self._get_video_path(
                    game_pk, at_bat_number, pitch_number, game_date
                ))
                skipped += 1
                continue

            # Download
            local_path = self.download_video(
                url=url,
                game_pk=game_pk,
                at_bat_number=at_bat_number,
                pitch_number=pitch_number,
                game_date=game_date,
                show_progress=False,  # Use batch progress instead
            )

            if local_path:
                results[play_id] = local_path
                downloaded += 1
            else:
                failed += 1

        logger.info(
            f"Batch download complete: {downloaded} downloaded, "
            f"{skipped} skipped, {failed} failed out of {total}"
        )
        return results

    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get information about a downloaded video.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with video info or None if file doesn't exist.
        """
        path = Path(video_path)
        if not path.exists():
            return None

        return {
            "path": str(path.absolute()),
            "size_bytes": path.stat().st_size,
            "size_mb": path.stat().st_size / (1024 * 1024),
            "filename": path.name,
            "modified_time": path.stat().st_mtime,
        }

    def list_downloaded_videos(self) -> List[Dict]:
        """
        List all downloaded videos.

        Returns:
            List of video info dictionaries.
        """
        videos = []
        for mp4_file in self.download_dir.rglob("*.mp4"):
            info = self.get_video_info(str(mp4_file))
            if info:
                # Parse game info from filename
                parts = mp4_file.stem.split("_")
                if len(parts) >= 3:
                    info["game_pk"] = parts[0]
                    info["at_bat_number"] = parts[1]
                    info["pitch_number"] = parts[2]
                videos.append(info)

        return videos

    def get_total_size(self) -> float:
        """
        Get total size of all downloaded videos in MB.

        Returns:
            Total size in megabytes.
        """
        total_bytes = sum(
            f.stat().st_size for f in self.download_dir.rglob("*.mp4")
        )
        return total_bytes / (1024 * 1024)

    def cleanup_partial_downloads(self) -> int:
        """
        Remove partial download files (.part files).

        Returns:
            Number of files cleaned up.
        """
        count = 0
        for part_file in self.download_dir.rglob("*.part"):
            try:
                part_file.unlink()
                count += 1
                logger.debug(f"Removed partial download: {part_file}")
            except OSError as e:
                logger.warning(f"Failed to remove {part_file}: {e}")
        return count

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
        logger.debug("Video downloader session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
