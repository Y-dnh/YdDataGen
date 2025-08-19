from pathlib import Path
from typing import List, Dict
from yt_dlp import YoutubeDL
from tqdm import tqdm
import logging

from src.config import CONFIG

logger = logging.getLogger(__name__)


class TqdmLogger:
    """
    A custom logger class designed to suppress output from yt-dlp.
    By default, yt-dlp prints a lot of information to the console, which can
    interfere with the visual output of our `tqdm` progress bar.
    This class provides the logging methods that yt-dlp expects, but they
    do nothing, effectively "muting" yt-dlp's own logging.
    """

    def debug(self, msg): pass

    def warning(self, msg): pass

    def error(self, msg): pass


def read_urls() -> List[str]:
    """
    Read video URLs from a text file.
    Skips empty lines and lines starting with '#' (comments).

    Returns:
        List of video URLs
    """
    try:
        if not CONFIG.paths.urls_file.exists():
            logger.error(f"URLs file not found: {CONFIG.paths.urls_file}")
            return []

        with open(CONFIG.paths.urls_file, 'r', encoding='utf-8') as f:
            urls = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
            return urls

    except Exception as e:
        logger.error(f"Error reading URLs file: {e}")
        return []


def download_single_video(url: str, ydl_opts: Dict) -> tuple[str, Dict]:
    """
    Download a single video and return its metadata.

    Args:
        url: Video URL to download
        ydl_opts: yt-dlp options dictionary

    Returns:
        Tuple of (video_id, video_metadata) or (None, None) if failed
    """
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id', f"video_{hash(url) % 10000:04d}")
            video_metadata = {
                'duration': info.get('duration'),
                'ext': info.get('ext'),  # File extension (e.g., 'mp4', 'webm')
                'resolution': info.get('resolution'),
                'filesize': info.get('filesize'),
                'fps': info.get('fps'),
                'path': None
            }

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
    """
    Download videos from URLs listed in the configured file.

    Returns:
        A dictionary containing information about successfully downloaded videos.
        The keys are the unique video IDs, and the values are dictionaries
        of their metadata (duration, path, etc.).
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

    ydl_opts = {
        'format': CONFIG.download_quality,  # The video/audio quality. Pulled from our config.
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        # A template for the output filename. We use the video's ID and extension.
        'noplaylist': True,
        'quiet': True,
        'logger': TqdmLogger()
    }

    for url in tqdm(urls, desc="Downloading", unit="video"):
        video_id, video_metadata = download_single_video(url, ydl_opts)
        if video_id and video_metadata:
            video_info_dict[video_id] = video_metadata

    logger.info(f"Downloaded {len(video_info_dict)} videos successfully")
    return video_info_dict


def get_download_options() -> Dict:
    """
    Get the yt-dlp options dictionary for downloads.

    Returns:
        Dictionary of yt-dlp options
    """
    output_dir = CONFIG.paths.videos_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        'format': CONFIG.download_quality,
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'logger': TqdmLogger()
    }