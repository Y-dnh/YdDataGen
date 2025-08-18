from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch


@dataclass
class ProjectPaths:
    """Contains all project file paths and creates base directories on disk."""
    root: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self):
        # Define all paths
        self.videos_dir = self.root / "dataset" / "videos"
        self.data_dir = self.root / "dataset" / "data"
        self.annotations_dir = self.root / "dataset" / "annotations_per_videos"
        self.labels_final_path = self.root / "dataset" / "labels_final.json"
        self.report_path = self.root / "dataset" / "report.pdf"
        self.models_dir = self.root / "models"
        self.urls_file = self.root / "urls.txt"
        self.logs_dir = self.root / "logs"

    def ensure_dirs(self):
        """Create all necessary directories if they do not already exist."""
        for path in (
                self.videos_dir,
                self.data_dir,
                self.annotations_dir,
                self.models_dir / "yolo_det",
                self.models_dir / "sam",
                self.models_dir / "trackers",
                self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class Config:
    def __init__(self):
        # Project paths
        self.paths = ProjectPaths()

        # Global settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference = 640
        self.half_precision = False

        # Logging settings
        self.log_level = "INFO"
        self.log_file = "ytdatagen.log"
        self.console_output = True
        self.file_output = True
        self.max_log_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5

        # Download settings
        self.video_quality = 640
        self.download_quality = (f"bestvideo[width={self.video_quality}]/"
                                 f"bestvideo[height={self.video_quality}]/"
                                 f"bestvideo[height<={self.video_quality}]/"
                                 f"bestvideo[width<={self.video_quality}]/"
                                 f"best")
        self.download_timeout = 300

        # Tracking settings
        # Set before downloading models if you want to create tracker with custom parameters
        # Change parameters in tracker .yaml file if you want to run inference with custom parameters
        # Check https://docs.ultralytics.com/modes/track/#tracker-arguments for more details
        self.tracker_type = "botsort.yaml"  # or "bytetrack.yaml" or custom tracker name
        self.track_high_thresh = 0.7
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.7
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.fuse_score = True
        # BoT-SORT settings
        self.gmc_method = "sparseOptFlow"
        self.proximity_thresh = 0.5  # minimum IoU for valid match with ReID
        self.appearance_thresh = 0.8  # minimum appearance similarity for ReID
        self.with_reid = True


        #TODO Yolo settings

        # Custom classes mapping
        self.custom_classes = {
            0: "person",
            1: "pet",
            2: "car"
        }


        #TODO SAM

        #TODO Polygon

        #TODO Static cars



# Global configuration instance
CONFIG = Config()

