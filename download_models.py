#!/usr/bin/env python3
"""
A script to download the necessary models for the YtDataGen project.

This script automates the process of fetching YOLO for detection and SAM for segmentation.
Creating configuration files for object trackers BoT-SORT and ByteTrack
based on parameters defined in the main configuration file.

If you need to inference with custom model, just put it into specific folder.
In the CLI write name of your model
"""

import requests
from pathlib import Path
from tqdm import tqdm
import yaml

from src.config import CONFIG


def download_file(url: str, destination: Path, description: str = None) -> bool:
    """
    Downloads a file from a given URL to a specified destination with a progress bar.

    Args:
        url (str): The URL of the file to be downloaded.
        destination (Path): A Path object representing the local file path to save the content.
        description (str, optional): A description for the download, displayed in the console.
                                     If None, the destination filename is used.

    Returns:
        bool: True if the file was downloaded successfully or already exists, False otherwise.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists to avoid re-downloading.
    if destination.exists():
        print(f"{destination.name} already exists, skipping...")
        return True

    try:
        print(f"Downloading {description or destination.name}...")

        # `stream=True` is crucial for downloading large files as it downloads the content
        # in chunks rather than loading the entire file into memory at once.
        response = requests.get(url, stream=True)
        # Raise an `HTTPError` if the HTTP request returned an unsuccessful status code (e.g., 404, 503).
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks.
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"Successfully downloaded {destination.name}")
        return True

    except Exception as e:
        print(f"Failed to download {destination.name}: {e}")
        # If an error occurs during download, delete the partially downloaded file
        # to prevent corruption issues.
        if destination.exists():
            destination.unlink()
        return False


def create_tracker_yaml(tracker_name: str, tracker_dir: Path) -> bool:
    """
    Creates a YAML configuration file for a specific tracker based on CONFIG parameters.

    Args:
        tracker_name (str): The filename for the tracker config (e.g., "botsort.yaml").
        tracker_dir (Path): The directory where the YAML file will be saved.

    Returns:
        bool: True if the file was created successfully, False otherwise.
    """
    tracker_path = tracker_dir / tracker_name

    if tracker_path.exists():
        print(f"{tracker_name} already exists, skipping...")
        return True

    try:
        if tracker_name == "botsort.yaml":
            config_data = {
                "tracker_type": "botsort",
                "track_high_thresh": CONFIG.track_high_thresh,
                "track_low_thresh": CONFIG.track_low_thresh,
                "new_track_thresh": CONFIG.new_track_thresh,
                "track_buffer": CONFIG.track_buffer,
                "match_thresh": CONFIG.match_thresh,
                "fuse_score": CONFIG.fuse_score,
                "gmc_method": CONFIG.gmc_method,
                "proximity_thresh": CONFIG.proximity_thresh,
                "appearance_thresh": CONFIG.appearance_thresh,
                "with_reid": CONFIG.with_reid,
                "model": "auto"
            }
        elif tracker_name == "bytetrack.yaml":
            config_data = {
                "tracker_type": "bytetrack",
                "track_high_thresh": CONFIG.track_high_thresh,
                "track_low_thresh": CONFIG.track_low_thresh,
                "new_track_thresh": CONFIG.new_track_thresh,
                "track_buffer": CONFIG.track_buffer,
                "match_thresh": CONFIG.match_thresh,
                "fuse_score": CONFIG.fuse_score
            }
        else:
            print(f"Unknown tracker type: {tracker_name}")
            return False

        with open(tracker_path, 'w', encoding='utf-8') as f:
            f.write("# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license\n")
            f.write(
                f"# Default Ultralytics settings for {config_data['tracker_type'].upper()} tracker when using mode=\"track\"\n")
            f.write("# Auto-generated from YtDataGen CONFIG parameters\n\n")

            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        print(f"Created {tracker_name} from CONFIG parameters")
        return True

    except Exception as e:
        print(f"Failed to create {tracker_name}: {e}")
        if tracker_path.exists():
            tracker_path.unlink()
        return False


def process(dirs: Path, models: dict) -> None:
    """
    A helper function to process and download a group of models.

    Args:
        dirs (Path): The target directory for downloading the models.
        models (dict): A dictionary where keys are model filenames and values are dicts
                       containing the 'url' and 'description' for each model.
    """
    dirs.mkdir(parents=True, exist_ok=True)

    print("Downloading models...")

    successful = 0
    # Iterate through the models dictionary and download each one.
    for model_name, model_info in models.items():
        destination = dirs / model_name
        if download_file(model_info["url"], destination, model_info["description"]):
            successful += 1

    print(f"\nModel download complete: {successful}/{len(models)} successful")


def download_sam_models():
    """Downloads Segment Anything Model (SAM) variants."""
    sam_models = {
        "sam2.1_t.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt", "description": "SAM 2.1 Tiny model"},
        "sam2.1_s.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt", "description": "SAM 2.1 Small model"},
        "sam2.1_b.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt", "description": "SAM 2.1 Base Plus model"},
        "sam2.1_l.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt", "description": "SAM 2.1 Large model"},
        "mobile_sam.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt", "description": "Mobile SAM model"}
    }
    sam_dir = CONFIG.paths.models_dir / "sam"
    process(sam_dir, sam_models)


def download_yolo_det_models():
    """Downloads YOLO object detection models."""
    yolo_models = {
        "yolov8n.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", "description": "YOLOv8 Nano detection model"},
        "yolov8s.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt", "description": "YOLOv8 Small detection model"},
        "yolov8m.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt", "description": "YOLOv8 Medium detection model"},
        "yolov8l.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt", "description": "YOLOv8 Large detection model"},
        "yolov8x.pt": {"url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt", "description": "YOLOv8 Extra Large detection model"}
    }
    yolo_det_dir = CONFIG.paths.models_dir / "yolo_det"
    process(yolo_det_dir, yolo_models)


def create_trackers():
    """Creates YAML configuration files for object trackers."""
    trackers_dir = CONFIG.paths.models_dir / "trackers"
    trackers_dir.mkdir(parents=True, exist_ok=True)

    print("Creating tracker configurations...")

    successful = 0
    total = 2

    if create_tracker_yaml("botsort.yaml", trackers_dir):
        successful += 1
    if create_tracker_yaml("bytetrack.yaml", trackers_dir):
        successful += 1

    print(f"\nTracker configuration creation complete: {successful}/{total} successful")

if __name__ == "__main__":
    download_sam_models()
    download_yolo_det_models()
    create_trackers()