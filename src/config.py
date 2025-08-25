from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class ProjectPaths:
    root: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self):
        self.videos_dir = self.root / "dataset" / "videos"
        self.data_dir = self.root / "dataset" / "data"
        self.annotations_dir = self.root / "dataset" / "annotations_per_videos"
        self.labels_final_path = self.root / "dataset" / "labels_final.json"
        self.report_path = self.root / "dataset" / "report.pdf"
        self.models_dir = self.root / "models"
        self.urls_file = self.root / "urls.txt"
        self.logs_dir = self.root / "logs"
        self.cvat_annotations_dir = self.root / "cvat_annotations"


    def ensure_dirs(self):
        for path in (
                self.videos_dir,
                self.data_dir,
                self.annotations_dir,
                self.models_dir / "yolo_det",
                self.models_dir / "sam",
                self.models_dir / "trackers",
                self.logs_dir,
                self.cvat_annotations_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class Config:
    def __init__(self):
        # Project paths
        self.paths = ProjectPaths()

        # Global settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clear_cache_after_video = True
        self.half_precision = False

        # Custom classes mapping
        self.custom_classes = {
            0: "person",
            2: "pet",
            1: "car"
        }

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
                                 f"bestvideo")

        # Tracking settings
        # Set before downloading models if you want to create tracker with custom parameters
        # Change parameters in tracker .yaml file if you want to run inference with custom parameters
        # Check https://docs.ultralytics.com/modes/track/#tracker-arguments for more details
        self.tracker_type = "botsort.yaml"  # or "bytetrack.yaml" or custom tracker name
        self.track_high_thresh = 0.5
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.fuse_score = True
        # BoT-SORT settings
        self.gmc_method = "sparseOptFlow"
        self.proximity_thresh = 0.5  # minimum IoU for valid match with ReID
        self.appearance_thresh = 0.8  # minimum appearance similarity for ReID
        self.with_reid = True
        self.min_confidence_for_tracking = 0.5

        # YOLO Detection settings
        self.yolo_model_path = "yolov8n.pt"
        self.yolo_confidence = 0.5
        self.yolo_iou = 0.5
        self.yolo_rect = True
        self.yolo_half = False
        self.yolo_max_det = 300
        self.yolo_classes = None  # None for all classes, or list like [0, 2] for specific
        self.yolo_agnostic_nms = False
        self.yolo_augment = False
        self.stream_buffer = False
        self.yolo_imgsz = 640

        # SAM Segmentation settings
        self.sam_model_path = "sam2.1_t.pt"
        self.sam_enabled = True
        self.sam_confidence = 0.5
        self.sam_iou = 0.5
        self.sam_retina_masks = True
        self.sam_half = False
        self.sam_imgsz = 640
        self.sam_add_center_point: bool = True

        # Segmentation polygon settings
        self.max_points = 100
        self.simplify_tolerance = 0.1
        self.min_area = 50.0
        self.smoothing = True
        self.fill_holes = True

        # Static car detection
        self.static_car_enabled = True
        self.movement_threshold = 100.0  # pixels - if any movement > this, car is not static
        self.min_static_duration = 300  # minimum frames to consider truly static
        self.static_check_interval  = 50

        # CVAT conversion settings
        self.cvat_enabled = True
        self.cvat_keyframe_mode = "fps"  # "fps", "interval", or "custom"
        self.cvat_keyframe_interval = 30  # frames (used when mode is "interval")
        self.cvat_keyframe_fps_multiplier = 1.0  # keyframe every N seconds (used when mode is "fps")
        self.cvat_custom_keyframes = []  # custom frame numbers

    def get_tracker_path(self) -> str:
        """Get full path to tracker configuration."""
        if self.tracker_type in ["botsort.yaml", "bytetrack.yaml"]:
            tracker_path = self.paths.models_dir / "trackers" / self.tracker_type
            if tracker_path.exists():
                return str(tracker_path)
            else:
                print(f"Warning: Tracker file {tracker_path} not found, using default")
                return self.tracker_type

        custom_path = self.paths.models_dir / "trackers" / self.tracker_type
        if custom_path.exists():
            return str(custom_path)

        return self.tracker_type

    def get_yolo_params(self) -> dict:
        """Get YOLO parameters for model.track() call including tracker parameters."""
        params = {
            "conf": self.yolo_confidence,
            "iou": self.yolo_iou,
            "device": self.device,
            "verbose": False,
            "save": False,
            "imgsz": self.yolo_imgsz,
            "max_det": self.yolo_max_det,
            "half": self.yolo_half or self.half_precision,
            "agnostic_nms": self.yolo_agnostic_nms,
            "augment": self.yolo_augment,
            "tracker": self.get_tracker_path(),
            "rect": self.yolo_rect,
            "stream_buffer": self.stream_buffer,
        }

        if self.yolo_classes is not None:
            params["classes"] = self.yolo_classes

        return params

    def get_sam_params(self) -> dict:
        """Get SAM parameters for prediction."""
        return {
            "conf": self.sam_confidence,
            "iou": self.sam_iou,
            "retina_masks": self.sam_retina_masks,
            "device": self.device,
            "verbose": False,
            "save": False,
            "half": self.sam_half or self.half_precision,
            "imgsz": self.sam_imgsz,
        }


# Global configuration instance
CONFIG = Config()


