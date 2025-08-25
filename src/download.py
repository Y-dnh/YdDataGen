from pathlib import Path
from typing import List, Dict, Tuple, Optional
from yt_dlp import YoutubeDL
from tqdm import tqdm
import logging

from src.config import CONFIG

logger = logging.getLogger(__name__)


class TqdmLogger:
    """Dummy logger to suppress yt-dlp output and prevent interference with tqdm progress bars."""

    def debug(self, msg): pass

    def warning(self, msg): pass

    def error(self, msg): pass


def read_urls() -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Read URLs from file with optional time ranges.

    Expected format per line: URL [start_time] [end_time]
    Lines starting with # are treated as comments.

    Returns:
        List of tuples: (url, start_time, end_time)
    """
    try:
        if not CONFIG.paths.urls_file.exists():
            logger.error(f"URLs file not found: {CONFIG.paths.urls_file}")
            return []

        urls = []
        with open(CONFIG.paths.urls_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0].startswith('#'):
                    continue

                url = parts[0]
                start_time = parts[1] if len(parts) > 1 else None
                end_time = parts[2] if len(parts) > 2 else None
                urls.append((url, start_time, end_time))

        return urls

    except Exception as e:
        logger.error(f"Error reading URLs file: {e}")
        return []


def download_single_video(url: str,
                          ydl_opts: Dict,
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict]]:
    """Download a single video with optional time range trimming.

    Args:
        url: Video URL to download
        ydl_opts: Base yt-dlp options
        start_time: Start time in format "HH:MM:SS", "MM:SS", or "SS"
        end_time: End time in same format as start_time

    Returns:
        Tuple of (video_id, metadata_dict) or (None, None) on failure
    """
    # Configure time-based trimming if specified
    if start_time or end_time:
        def to_seconds(t: str) -> int:
            """Convert time string to seconds. Supports HH:MM:SS, MM:SS, or SS formats."""
            parts = t.split(":")
            parts = [int(p) for p in parts]
            if len(parts) == 1:
                return parts[0]  # Just seconds
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]  # MM:SS
            elif len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]  # HH:MM:SS
            return 0

        start_sec = to_seconds(start_time) if start_time else None
        end_sec = to_seconds(end_time) if end_time else None

        # Clone options and add download range configuration
        ydl_opts = ydl_opts.copy()
        ydl_opts["download_ranges"] = lambda info_dict, ydl: [
            {"start_time": start_sec, "end_time": end_sec}
        ]
        # Ensure proper format after trimming
        ydl_opts.setdefault("postprocessors", []).append({
            "key": "FFmpegVideoRemuxer",
            "preferedformat": "mp4"
        })

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id', f"video_{hash(url) % 10000:04d}")
            video_metadata = {
                'ext': info.get('ext'),
                'filesize': info.get('filesize'),
                'path': None
            }

            # Extract actual file path from download info
            if 'requested_downloads' in info and info['requested_downloads']:
                file_path = Path(info['requested_downloads'][0]['filepath'])
                video_metadata['path'] = str(file_path)

            logger.info(f"Downloaded: {video_id}")
            return video_id, video_metadata

    except Exception as e:
        tqdm.write(f"[ERROR] Failed to download {url}: {e}")
        logger.error(f"Error downloading {url}: {e}")
        return None, None


def download_videos() -> Dict[str, Dict]:
    """Download all videos from URLs file.

    Returns:
        Dictionary mapping video_id -> metadata for successfully downloaded videos
    """
    urls = read_urls()
    if not urls:
        logger.error("No URLs found to download")
        return {}

    output_dir = CONFIG.paths.videos_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting download of {len(urls)} videos")
    print("=" * 50 + "\nDownloading videos\n" + "=" * 50)

    video_info_dict = {}

    # Base yt-dlp configuration
    base_opts = {
        'format': CONFIG.download_quality,
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),  # Use video ID as filename
        'noplaylist': True,
        'quiet': True,
        'logger': TqdmLogger()  # Suppress yt-dlp logs to avoid tqdm interference
    }

    for url, start_time, end_time in tqdm(urls, desc="Downloading", unit="video"):
        video_id, video_metadata = download_single_video(url, base_opts, start_time, end_time)
        if video_id and video_metadata:
            video_info_dict[video_id] = video_metadata

    logger.info(f"Downloaded {len(video_info_dict)} videos successfully")
    return video_info_dict