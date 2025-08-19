"""
Utility functions for video processing and computer vision pipeline.

This module provides essential utility functions for:
- Logging configuration and management
- Project setup and directory structure
- Video metadata extraction
- Model path resolution
- File operations and data loading

The utilities are designed to support a video processing pipeline that uses
YOLO for object detection/tracking and SAM for segmentation.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any
import cv2
import json

from src.config import CONFIG

# Global flag to prevent multiple logging setups during application lifecycle
_logging_setup_done = False


def setup_logging():
    """
    Setup enhanced logging system with file rotation and console output.

    Configures a comprehensive logging system that supports both console
    and file output with rotation. Prevents duplicate handler creation
    and configures appropriate log levels for different components.

    Features:
    - Rotating file handler to prevent log files from growing too large
    - Console handler for real-time monitoring
    - Different formatters for file (detailed) vs console (simple)
    - Suppression of noisy third-party library logs
    - UTF-8 encoding support for international characters

    Log Levels:
    - File logging: Always DEBUG level for comprehensive debugging
    - Console logging: Configurable via CONFIG.log_level

    Note:
        Uses global flag to ensure logging is only configured once,
        preventing handler duplication in multi-threaded environments.
    """
    global _logging_setup_done

    # Prevent multiple logging configuration calls
    if _logging_setup_done:
        return

    # Ensure logs directory exists (create if needed, don't delete existing)
    CONFIG.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    # Create different formatters for different output targets
    # Detailed formatter for file logs includes timestamp and module info
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Simple formatter for console output focuses on readability
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # Get root logger and set base level
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, CONFIG.log_level))

    # Clear any existing handlers to avoid duplicates
    # This is important when restarting or reconfiguring logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Setup console handler for real-time monitoring
    if CONFIG.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, CONFIG.log_level))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Setup rotating file handler for persistent logging
    if CONFIG.file_output:
        log_file = CONFIG.paths.logs_dir / CONFIG.log_file
        try:
            # Rotating file handler prevents log files from becoming too large
            # maxBytes: Maximum size before rotation
            # backupCount: Number of backup files to keep
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=CONFIG.max_log_size,
                backupCount=CONFIG.backup_count,
                encoding='utf-8'  # Support international characters
            )
            file_handler.setLevel(logging.DEBUG)  # Always detailed for file logs
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Graceful fallback: if file logging fails, continue with console only
            logger.warning(f"Could not setup file logging: {e}")

    # Suppress verbose logging from third-party libraries
    # These libraries tend to produce excessive debug output
    logging.getLogger('ultralytics').setLevel(logging.WARNING)  # YOLO library
    logging.getLogger('PIL').setLevel(logging.WARNING)  # Image processing
    logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Plotting library

    _logging_setup_done = True
    logger.info("Logging setup completed")


def setup_project() -> bool:
    """
    Setup project directories and initialize logging system.

    Performs initial project setup including directory creation and
    logging configuration. This should be called once at application startup.

    Returns:
        bool: True if setup successful, False if critical errors occurred

    Setup Process:
    1. Initialize logging system
    2. Create required project directories
    3. Log configuration summary
    4. Validate critical settings

    Directory Structure Created:
    - Root project directory
    - Data directory (for processed frames)
    - Models directory (for YOLO/SAM models)
    - Logs directory (for application logs)
    - Output directory (for results)
    """
    try:
        # Initialize logging system first (required for error reporting)
        setup_logging()
        logger = logging.getLogger(__name__)

        # Create all required project directories
        # This includes data, models, logs, and output directories
        CONFIG.paths.ensure_dirs()
        logger.info("Project directories created")

        # Log important configuration information for debugging
        logger.info(f"Project root: {CONFIG.paths.root}")
        logger.info(f"Device: {CONFIG.device}")  # CPU/GPU device
        logger.info(f"YOLO model: {CONFIG.yolo_model_path}")  # Detection model
        logger.info(f"SAM enabled: {CONFIG.sam_enabled}")  # Segmentation enabled

        return True

    except Exception as e:
        # Use print instead of logger since logging might not be configured
        print(f"Failed to setup project: {e}")
        return False


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Extract comprehensive video metadata using OpenCV.

    Retrieves essential video properties including resolution, frame rate,
    duration, and quality classification. Handles various video formats
    and provides fallback values for corrupted or incomplete metadata.

    Args:
        video_path (Path): Path to the video file to analyze

    Returns:
        Dict[str, Any]: Video metadata dictionary containing:
            - path: Original video file path
            - duration: Video length in seconds
            - fps: Frames per second
            - frames: Total frame count
            - width: Video width in pixels
            - height: Video height in pixels
            - resolution: Formatted resolution string
            - video_quality: Quality classification (1080p+, 720p, 480p, Low)
            - error: Error message if metadata extraction failed

    Quality Classification:
    - 1080p+: Height >= 1080 pixels
    - 720p: Height >= 720 pixels
    - 480p: Height >= 480 pixels
    - Low: Height < 480 pixels

    Fallback Behavior:
        If video properties cannot be read, uses configured default values
        to ensure processing can continue with reasonable assumptions.
    """
    # Initialize info dictionary with defaults and basic metadata
    info = {
        "path": str(video_path),
        "duration": 0.0,  # Duration in seconds
        "fps": 30.0,  # Default frame rate
        "frames": 0,  # Total frame count
        "width": CONFIG.inference_resolution[0],  # Video width
        "height": CONFIG.inference_resolution[1],  # Video height
        "resolution": f"{CONFIG.inference_resolution[0]}x{CONFIG.inference_resolution[1]}"  # Formatted resolution
    }

    try:
        # Validate video file exists before attempting to read
        if not video_path.exists():
            info["error"] = "Video file not found"
            return info

        # Initialize OpenCV video capture for metadata extraction
        cap = cv2.VideoCapture(str(video_path))

        if cap.isOpened():
            # Extract video properties using OpenCV property getters
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

            # Calculate video duration from frame count and fps
            duration = frame_count / fps if fps > 0 else 0

            # Update info dictionary with extracted values
            # Use fallback values if extracted values are invalid (0 or negative)
            info.update({
                "fps": fps if fps > 0 else 30.0,
                "frames": frame_count,
                "width": width if width > 0 else CONFIG.inference_resolution[0],
                "height": height if height > 0 else CONFIG.inference_resolution[1],
                "duration": duration,
                "resolution": f"{width}x{height}" if width > 0 and height > 0 else f"{CONFIG.inference_resolution[0]}x{CONFIG.inference_resolution[1]}"
            })

            # Classify video quality based on vertical resolution
            # This helps in processing optimization and user feedback
            if height >= 1080:
                info["video_quality"] = "1080p+"
            elif height >= 720:
                info["video_quality"] = "720p"
            elif height >= 480:
                info["video_quality"] = "480p"
            else:
                info["video_quality"] = "Low"

        # Always release video capture to free system resources
        cap.release()

    except Exception as e:
        # Log extraction errors but don't fail completely
        logging.getLogger(__name__).warning(f"Error reading video info for {video_path}: {e}")
        info["error"] = str(e)

    return info


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable time string.

    Converts numeric duration to natural language format suitable
    for user interfaces and logging output.

    Args:
        seconds (float): Duration in seconds (can be fractional)

    Returns:
        str: Formatted duration string

    Examples:
        format_duration(65.5) -> "1m 5s"
        format_duration(3725.0) -> "1h 2m 5s"
        format_duration(45.0) -> "45s"
        format_duration(0) -> "0s"

    Format Rules:
    - Hours included only if duration >= 1 hour
    - Minutes included only if duration >= 1 minute
    - Seconds always included (truncated to integer)
    - Negative durations treated as 0
    """
    if seconds <= 0:
        return "0s"

    # Convert to time components
    hours = int(seconds // 3600)  # Full hours
    minutes = int((seconds % 3600) // 60)  # Remaining minutes after hours
    secs = int(seconds % 60)  # Remaining seconds after minutes

    # Format based on magnitude for natural readability
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes with error handling.

    Safely retrieves file size and converts to MB for user-friendly display.
    Handles missing files and permission errors gracefully.

    Args:
        file_path (Path): Path to file to measure

    Returns:
        float: File size in megabytes, or 0.0 if error occurred

    Note:
        Uses binary megabytes (1024^2 bytes) for accurate size representation.
        Returns 0.0 for any errors to avoid breaking calling code.
    """
    try:
        # Get file statistics and convert bytes to megabytes
        # stat().st_size returns size in bytes
        return file_path.stat().st_size / (1024 * 1024)  # Convert to MB
    except Exception:
        # Return 0 for any errors (file not found, permission denied, etc.)
        return 0.0


def get_yolo_model_path(model_name: str) -> str:
    """
    Resolve YOLO model path with intelligent searching and fallback handling.

    Searches for YOLO model files in the configured models directory,
    handling various naming conventions and providing helpful feedback
    when models are not found.

    Args:
        model_name (str): YOLO model name or path (with or without .pt extension)

    Returns:
        str: Resolved absolute path to YOLO model file

    Search Strategy:
    1. If absolute path provided and exists, return as-is
    2. Clean model name (ensure .pt extension)
    3. Search in models/yolo_det directory
    4. Return expected path with helpful logging if not found

    Example Model Names:
    - "yolov8n" -> searches for "yolov8n.pt"
    - "yolov8n.pt" -> searches for "yolov8n.pt"
    - "/absolute/path/model.pt" -> validates and returns if exists

    Directory Structure:
        models/
        └── yolo_det/
            ├── yolov8n.pt
            ├── yolov8s.pt
            └── custom_model.pt
    """
    # Handle absolute paths - return if valid
    if Path(model_name).is_absolute() and Path(model_name).exists():
        return model_name

    # Normalize model name to ensure .pt extension
    clean_name = model_name
    if not clean_name.endswith('.pt'):
        clean_name += '.pt'

    # Search in configured YOLO models directory
    yolo_dir = CONFIG.paths.models_dir / "yolo_det"
    model_path = yolo_dir / clean_name

    logger = logging.getLogger(__name__)

    # Check if model exists in expected location
    if model_path.exists():
        logger.info(f"Found YOLO model: {model_path}")
        return str(model_path)

    # Model not found - provide helpful guidance
    logger.warning(f"YOLO model not found: {model_path}")
    logger.info(f"Please download {clean_name} to: {yolo_dir}")

    # Return expected path anyway - calling code can handle the missing file
    return str(model_path)


def get_sam_model_path(model_name: str) -> str:
    """
    Resolve SAM (Segment Anything Model) path with intelligent searching.

    Similar to YOLO model resolution but searches in the SAM-specific
    directory. SAM models are typically larger and less frequently changed.

    Args:
        model_name (str): SAM model name or path (with or without .pt extension)

    Returns:
        str: Resolved absolute path to SAM model file

    Search Strategy:
    1. If absolute path provided and exists, return as-is
    2. Clean model name (ensure .pt extension)
    3. Search in models/sam directory
    4. Return expected path with helpful logging if not found

    Common SAM Models:
    - "sam_vit_b_01ec64.pt" - Base ViT model
    - "sam_vit_l_0b3195.pt" - Large ViT model
    - "sam_vit_h_4b8939.pt" - Huge ViT model

    Directory Structure:
        models/
        └── sam/
            ├── sam_vit_b_01ec64.pt
            ├── sam_vit_l_0b3195.pt
            └── sam_vit_h_4b8939.pt

    Note:
        SAM models are large (several GB) and should be downloaded manually
        from the official Segment Anything repository.
    """
    # Handle absolute paths - return if valid
    if Path(model_name).is_absolute() and Path(model_name).exists():
        return model_name

    # Normalize model name to ensure .pt extension
    clean_name = model_name
    if not clean_name.endswith('.pt'):
        clean_name += '.pt'

    # Search in configured SAM models directory
    sam_dir = CONFIG.paths.models_dir / "sam"
    model_path = sam_dir / clean_name

    logger = logging.getLogger(__name__)

    # Check if model exists in expected location
    if model_path.exists():
        logger.info(f"Found SAM model: {model_path}")
        return str(model_path)

    # Model not found - provide helpful guidance
    logger.warning(f"SAM model not found: {model_path}")
    logger.info(f"Please download {clean_name} to: {sam_dir}")

    # Return expected path anyway - calling code can handle the missing file
    return str(model_path)


def load_json(file_path: Path) -> Any:
    """
    Load data from JSON file with comprehensive error handling.

    Safely loads JSON data with proper encoding support and graceful
    error handling. Suitable for loading configuration files, annotation
    data, and other structured data.

    Args:
        file_path (Path): Path to JSON file to load

    Returns:
        Any: Loaded JSON data (dict, list, etc.) or None if loading failed

    Error Handling:
    - File not found: Returns None with warning
    - JSON parse errors: Returns None with error log
    - Permission errors: Returns None with error log
    - Encoding errors: Handled by UTF-8 specification

    Example Usage:
        config_data = load_json(Path("config.json"))
        if config_data is not None:
            # Process loaded data
            process_config(config_data)
        else:
            # Handle loading failure
            use_default_config()
    """
    try:
        # Check file existence before attempting to read
        if not file_path.exists():
            logging.warning(f"JSON file not found: {file_path}")
            return None

        # Open file with explicit UTF-8 encoding for international character support
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        # Specific handling for JSON parsing errors
        logging.error(f"Invalid JSON format in {file_path}: {e}")
        return None
    except PermissionError as e:
        # Handle file permission issues
        logging.error(f"Permission denied reading {file_path}: {e}")
        return None
    except Exception as e:
        # Catch-all for other file operation errors
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        return None