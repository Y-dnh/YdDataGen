import numpy as np
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any
import cv2
import json

from src.config import CONFIG

_logging_setup_done = False


def setup_logging():
    """
    Configure logging system with both console and file output.

    Sets up rotating file handler and console handler with different formatters.
    Only runs once to prevent duplicate handlers.
    """
    global _logging_setup_done

    if _logging_setup_done:
        return

    CONFIG.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    # Different formatters for file (detailed) and console (simple)
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, CONFIG.log_level))

    # Clear any existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console output handler
    if CONFIG.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, CONFIG.log_level))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File output with rotation to prevent huge log files
    if CONFIG.file_output:
        log_file = CONFIG.paths.logs_dir / CONFIG.log_file
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=CONFIG.max_log_size,
                backupCount=CONFIG.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")

    # Suppress verbose logging from third-party libraries
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    _logging_setup_done = True
    logger.info("Logging setup completed")


def setup_project() -> bool:
    """
    Initialize project environment including logging and directory structure.

    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        setup_logging()
        logger = logging.getLogger(__name__)

        CONFIG.paths.ensure_dirs()
        logger.info("Project directories created")

        logger.info(f"Project root: {CONFIG.paths.root}")
        logger.info(f"Device: {CONFIG.device}")
        logger.info(f"YOLO model: {CONFIG.yolo_model_path}")
        logger.info(f"SAM enabled: {CONFIG.sam_enabled}")

        return True

    except Exception as e:
        print(f"Failed to setup project: {e}")
        return False


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Extract comprehensive video metadata using OpenCV.

    Handles missing files and corrupted videos gracefully with fallback values.
    Also determines video quality category based on resolution.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing video metadata with fallback values
    """
    # Initialize with safe default values
    info = {
        "path": str(video_path),
        "duration": 0.0,
        "fps": 30.0,
        "frames": 0,
        "width": CONFIG.yolo_imgsz,
        "height": CONFIG.yolo_imgsz,
        "resolution": f"{CONFIG.yolo_imgsz}x{CONFIG.yolo_imgsz}"
    }

    try:
        if not video_path.exists():
            info["error"] = "Video file not found"
            return info

        cap = cv2.VideoCapture(str(video_path))

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate duration from frame count and fps
            duration = frame_count / fps if fps > 0 else 0

            info.update({
                "fps": fps if fps > 0 else 30.0,
                "frames": frame_count,
                "width": width if width > 0 else CONFIG.yolo_imgsz,
                "height": height if height > 0 else CONFIG.yolo_imgsz,
                "duration": duration,
                "resolution": f"{width}x{height}" if width > 0 and height > 0 else f"{CONFIG.yolo_imgsz}x{CONFIG.yolo_imgsz}"
            })

            # Categorize video quality based on height
            if height >= 1080:
                info["video_quality"] = "1080p+"
            elif height >= 720:
                info["video_quality"] = "720p"
            elif height >= 480:
                info["video_quality"] = "480p"
            else:
                info["video_quality"] = "Low"

        cap.release()

    except Exception as e:
        logging.getLogger(__name__).warning(f"Error reading video info for {video_path}: {e}")
        info["error"] = str(e)

    return info


def format_duration(seconds: float) -> str:
    """Convert seconds to human-readable format (HH:MM:SS or MM:SS or SS)."""
    if seconds <= 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes with error handling."""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def get_model_path(model_name: str, model_type: str = "yolo") -> str:
    """
    Resolve model path with support for both absolute paths and model directory lookup.

    Handles model discovery in organized subdirectories and provides helpful
    error messages when models are not found.

    Args:
        model_name: Name of the model file (with or without .pt extension)
        model_type: Type of model ("yolo" or "sam")

    Returns:
        String path to model file (even if it doesn't exist)

    Raises:
        ValueError: If model_type is not supported
    """
    # Return absolute paths as-is if they exist
    if Path(model_name).is_absolute() and Path(model_name).exists():
        return model_name

    # Ensure .pt extension
    clean_name = model_name
    if not clean_name.endswith('.pt'):
        clean_name += '.pt'

    # Model type configuration with metadata
    model_configs = {
        "yolo": {
            "subdir": "yolo_det",
            "description": "YOLO detection model",
            "examples": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
        },
        "sam": {
            "subdir": "sam",
            "description": "SAM (Segment Anything Model)",
            "examples": ["sam_vit_b_01ec64.pt", "sam_vit_l_0b3195.pt", "sam_vit_h_4b8939.pt"]
        }
    }

    if model_type not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}. "
                         f"Supported types: {list(model_configs.keys())}")

    config = model_configs[model_type]
    model_dir = CONFIG.paths.models_dir / config["subdir"]
    model_path = model_dir / clean_name

    logger = logging.getLogger(__name__)

    if model_path.exists():
        logger.info(f"Found {config['description']}: {model_path}")
        return str(model_path)

    # Provide helpful guidance for missing models
    logger.warning(f"{config['description']} not found: {model_path}")
    logger.info(f"Please download {clean_name} to: {model_dir}")

    if model_type == "sam":
        logger.info("SAM models are large files that should be downloaded manually "
                    "from the official Segment Anything repository")

    return str(model_path)


def get_yolo_model_path(model_name: str) -> str:
    """Convenience wrapper for YOLO model path resolution."""
    return get_model_path(model_name, model_type="yolo")


def get_sam_model_path(model_name: str) -> str:
    """Convenience wrapper for SAM model path resolution."""
    return get_model_path(model_name, model_type="sam")


def load_json(file_path: Path) -> Any:
    """
    Load JSON file with comprehensive error handling.

    Handles various failure modes including missing files, corrupted JSON,
    permission errors, etc. Returns None on any error.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data or None if loading failed
    """
    try:
        if not file_path.exists():
            logging.warning(f"JSON file not found: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in {file_path}: {e}")
        return None
    except PermissionError as e:
        logging.error(f"Permission denied reading {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        return None


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types that aren't natively JSON serializable."""

    def default(self, obj):
        # Convert NumPy integers to Python int
        if isinstance(obj, (np.integer,)):
            return int(obj)
        # Convert NumPy floats to Python float
        if isinstance(obj, (np.floating,)):
            return float(obj)
        # Convert NumPy arrays to Python lists
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)