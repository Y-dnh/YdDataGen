#!/usr/bin/env python3
"""
Script to download required models for YtDataGen
If you want to add custom model, just put it in specific folder
"""

import requests
from pathlib import Path
from tqdm import tqdm
import yaml

from src.config import CONFIG


def download_file(url: str, destination: Path, description: str = None):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"{destination.name} already exists, skipping...")
        return True

    try:
        print(f"Downloading {description or destination.name}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✓ Downloaded {destination.name}")
        return True

    except Exception as e:
        print(f"✗ Failed to download {destination.name}: {e}")
        if destination.exists():
            destination.unlink()
        return False


def create_tracker_yaml(tracker_name: str, tracker_dir: Path):
    """Create tracker YAML configuration file using CONFIG parameters"""

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

        # Write YAML file
        with open(tracker_path, 'w', encoding='utf-8') as f:
            f.write("# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license\n")
            f.write(
                f"# Default Ultralytics settings for {config_data['tracker_type'].upper()} tracker when using mode=\"track\"\n")
            f.write("# Auto-generated from YtDataGen CONFIG parameters\n\n")

            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Created {tracker_name} with CONFIG parameters")
        return True

    except Exception as e:
        print(f"✗ Failed to create {tracker_name}: {e}")
        if tracker_path.exists():
            tracker_path.unlink()
        return False


def process(dirs, models):
    dirs.mkdir(parents=True, exist_ok=True)

    print("Downloading models...")

    successful = 0
    for model_name, model_info in models.items():
        destination = dirs / model_name
        if download_file(model_info["url"], destination, model_info["description"]):
            successful += 1

    print(f"\nModels download complete: {successful}/{len(models)} successful")


def download_sam_models():
    """Download SAM models"""

    sam_models = {
        "sam2.1_t.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt",
            "description": "SAM 2.1 Tiny model"
        },
        "sam2.1_s.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt",
            "description": "SAM 2.1 Small model"
        },
        "sam2.1_b.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt",
            "description": "SAM 2.1 Base Plus model"
        },
        "sam2.1_l.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt",
            "description": "SAM 2.1 Large model"
        },
        "mobile_sam.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt",
            "description": "Mobile SAM model"
        }
    }

    sam_dir = CONFIG.paths.models_dir / "sam"
    process(sam_dir, sam_models)


def download_yolo_det_models():
    """Download YOLO-det models"""

    yolo_models = {
        "yolov8n.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "description": "YOLOv8 Nano detection model"
        },
        "yolov8s.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
            "description": "YOLOv8 Small detection model"
        },
        "yolov8m.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
            "description": "YOLOv8 Medium detection model"
        },
        "yolov8l.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
            "description": "YOLOv8 Large detection model"
        },
        "yolov8x.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt",
            "description": "YOLOv8 Large detection model"
        }
    }

    yolo_det_dir = CONFIG.paths.models_dir / "yolo_det"
    process(yolo_det_dir, yolo_models)


def create_trackers():
    """Create tracker YAML configuration files using CONFIG parameters"""

    trackers_dir = CONFIG.paths.models_dir / "trackers"
    trackers_dir.mkdir(parents=True, exist_ok=True)

    print("Creating tracker configurations...")

    successful = 0
    total = 0

    # Create BoT-SORT tracker
    total += 1
    if create_tracker_yaml("botsort.yaml", trackers_dir):
        successful += 1

    # Create ByteTrack tracker
    total += 1
    if create_tracker_yaml("bytetrack.yaml", trackers_dir):
        successful += 1

    print(f"\nTracker configurations created: {successful}/{total} successful")


# Main execution
if __name__ == "__main__":
    download_sam_models()
    download_yolo_det_models()
    create_trackers()