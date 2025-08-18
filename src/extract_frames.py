import cv2
import logging
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from src.config import CONFIG

logger = logging.getLogger(__name__)


def extract_frames(video_info_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Extracts all frames from each video specified in the input dictionary.

    For each video, it creates a dedicated folder and saves every frame as a
    separate JPG image. It then updates the input dictionary with the total
    frame count and the path to the directory containing the frames.

    Args:
        video_info_dict: A dictionary where keys are video IDs and values are
                         dictionaries containing video metadata. Each inner
                         dictionary MUST have a 'path' key pointing to the video file.

    Returns:
        The updated `video_info_dict` with two new keys for each video:
        "frames" (the total number of extracted frames) and "frames_dir"
        (the path to the directory where frames were saved).
    """
    logger.info("Starting frame extraction process")
    print("=" * 50 + "\nExtracting frames\n" + "=" * 50)

    for video_id, info in video_info_dict.items():
        try:
            video_path = Path(info["path"])
            frames_dir = CONFIG.paths.data_dir / video_path.stem
            frames_dir.mkdir(parents=True, exist_ok=True)

            if not video_path.exists():
                tqdm.write(f"[ERROR] Video file not found: {video_path}")
                logger.error(f"Video file not found: {video_path}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                tqdm.write(f"[ERROR] Cannot open video file: {video_path}")
                logger.error(f"Cannot open video file: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0  # Initialize a counter for the frames we extract.

            with tqdm(total=total_frames, desc=f"{video_id}: extracting", unit="frame",
                      bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Construct the filename for the current frame. e.g., "my_video_00001.jpg".
                    frame_file = frames_dir / f"{video_id}_{frame_idx:05d}.jpg"
                    cv2.imwrite(str(frame_file), frame)

                    frame_idx += 1
                    pbar.update(1)

            cap.release()

            info["frames"] = frame_idx
            info["frames_dir"] = str(frames_dir)

            tqdm.write(f"{video_id}: {frame_idx} frames extracted successfully")
            logger.info(f"Extracted {frame_idx} frames from video {video_id}")

        except Exception as e:
            tqdm.write(f"[ERROR] Failed to extract frames from {video_id}: {e}")
            logger.error(f"Failed to extract frames from {video_id}: {e}")

    logger.info("Frame extraction process completed for all videos")
    return video_info_dict