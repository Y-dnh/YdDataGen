import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from ultralytics import YOLO, SAM
import logging
from tqdm import tqdm
import time

from src.config import CONFIG
from src.utils import get_yolo_model_path, get_sam_model_path

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Optimized inference engine for video processing with enhanced SAM and segmentation handling.

    This class provides a complete pipeline for object detection, tracking, and segmentation
    in video sequences. It combines YOLO for object detection/tracking with SAM (Segment
    Anything Model) for precise segmentation masks.

    Key Features:
    - YOLO-based object detection and tracking
    - Optional SAM segmentation for improved mask quality
    - Batch processing optimization for SAM
    - Static object detection (particularly for cars)
    - Memory management and GPU cache clearing
    - Configurable parameters for different use cases

    Attributes:
        tracks (Dict): Stores tracking history for each detected object
        annotation_id (int): Counter for unique annotation IDs
        image_id (int): Counter for unique image IDs
        max_points (int): Maximum number of points in polygon approximation
        tolerance (float): Tolerance for polygon simplification
        min_area (float): Minimum area threshold for valid detections
        _sam_params (Dict): Cached SAM model parameters
        _yolo_params (Dict): Cached YOLO model parameters
        _morphology_kernel (np.ndarray): Cached kernel for morphological operations
        yolo_model: Loaded YOLO model instance
        sam_model: Loaded SAM model instance (optional)
    """

    def __init__(self):
        """
        Initialize the inference engine with default parameters and model loading.

        Sets up caching for frequently used parameters, initializes tracking
        structures, and loads the required models (YOLO and optionally SAM).
        """
        # Initialize tracking structures
        self.tracks = {}  # Stores object tracking history: {track_id: {history: [], class_name: str}}
        self.annotation_id = 1  # Unique identifier for annotations
        self.image_id = 1  # Unique identifier for images

        # Cache segmentation parameters to avoid repeated CONFIG lookups
        self.max_points = CONFIG.max_points  # Maximum polygon vertices
        # todo Douglas-Peucker approximation tolerance
        self.tolerance = CONFIG.simplify_tolerance  # Douglas-Peucker approximation tolerance
        self.min_area = CONFIG.min_area  # Minimum detection area in pixels

        # Cache SAM parameters once during initialization to improve performance
        self._sam_params = None
        if CONFIG.sam_enabled:
            self._sam_params = CONFIG.get_sam_params()

        # Cache YOLO parameters for consistent inference settings
        self._yolo_params = CONFIG.get_yolo_params()

        # Pre-create morphological kernel for hole filling operations
        # Using elliptical kernel for more natural shape processing
        if CONFIG.fill_holes:
            # todo ksize
            self._morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Load and initialize models
        self._load_models()


    def _load_models(self):
        """
        Load YOLO and SAM models using simple path resolution.

        Handles model loading with proper error handling and fallback mechanisms.
        Verifies model paths and provides informative logging for debugging.

        Raises:
            Exception: If YOLO model fails to load (critical error)

        Note:
            SAM loading failures are handled gracefully by disabling SAM functionality
        """
        logger.info("Loading models...")

        # Load YOLO model (required component)
        try:
            # Resolve YOLO model path and load
            yolo_path = get_yolo_model_path(CONFIG.yolo_model_path)
            logger.info(f"Loading YOLO model: {yolo_path}")

            # Initialize YOLO model and move to specified device (CPU/GPU)
            self.yolo_model = YOLO(yolo_path)
            self.yolo_model.to(CONFIG.device)
            logger.info(f"YOLO model loaded successfully: {Path(yolo_path).name}")

            # Verify tracker configuration exists
            tracker_path = CONFIG.get_tracker_path()
            logger.info(f"Tracker configuration path: {tracker_path}")

            if Path(tracker_path).exists():
                logger.info(f"Tracker configuration found: {Path(tracker_path).name}")
            else:
                logger.info(f"Using built-in tracker: {tracker_path}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise  # YOLO is critical - fail if it can't load

        # Load SAM model (optional component)
        self.sam_model = None
        if CONFIG.sam_enabled:
            try:
                # Resolve SAM model path
                sam_path = get_sam_model_path(CONFIG.sam_model_path)
                logger.info(f"Loading SAM model: {sam_path}")

                # Only load if model file exists
                if Path(sam_path).exists():
                    self.sam_model = SAM(sam_path)
                    self.sam_model.to(CONFIG.device)
                    logger.info(f"SAM model loaded successfully: {Path(sam_path).name}")
                else:
                    # Graceful fallback - disable SAM if model not found
                    logger.warning(f"SAM model not found: {sam_path}, disabling SAM")
                    CONFIG.sam_enabled = False

            except Exception as e:
                # SAM failures are non-critical - continue without segmentation
                logger.error(f"Failed to load SAM model: {e}")
                logger.warning("Disabling SAM segmentation")
                CONFIG.sam_enabled = False
                self.sam_model = None


    def process_video(self, video_id: str, video_info: Dict) -> Dict[str, Any]:
        """
        Optimized video processing pipeline with batch SAM and cached parameters.

        Processes all frames in a video directory, performing object detection,
        tracking, and optional segmentation. Results are compiled into a
        comprehensive annotation dictionary.

        Args:
            video_id (str): Unique identifier for the video
            video_info (Dict): Video metadata including file path and properties

        Returns:
            Dict[str, Any]: Complete annotation results containing:
                - video_id: Input video identifier
                - annotations: List of per-frame annotations
                - statistics: Aggregated processing statistics

        Processing Pipeline:
            1. Reset tracking state for new video
            2. Load frame files from video directory
            3. For each frame:
               - Run YOLO detection and tracking
               - Process detection results
               - Apply SAM segmentation (if enabled)
               - Update tracking history
            4. Calculate static object statistics
            5. Compile final results and statistics
        """
        logger.info(f"Processing video: {video_id}")

        # Reset YOLO predictor state to ensure clean tracking for new video
        # This prevents track ID conflicts between different videos
        if hasattr(self.yolo_model, 'predictor') and self.yolo_model.predictor:
            self.yolo_model.predictor = None
        self.tracks = {}  # Clear previous tracking history

        # Resolve video frames directory path
        frames_dir = Path(video_info['frames_dir'])

        # Validate frames directory exists
        if not frames_dir.exists():
            logger.error(f"Frames directory not found: {frames_dir}")
            return {}

        # Get sorted list of frame files (ensures chronological order)
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            logger.error(f"No frames found in: {frames_dir}")
            return {}

        # Initialize comprehensive results structure
        results = {
            "video_id": video_id,
            "annotations": [],
            "statistics": {
                "total_frames": len(frame_files),
                "processed_frames": 0,
                "total_detections": 0,
                "unique_tracks": {"person": set(), "car": set(), "pet": set()},
                "people_count": 0,
                "pets_count": 0,
                "cars_count": 0,
                "static_cars_count": 0,
                "avg_confidence": 0.0
            }
        }

        confidence_sum = 0.0  # For calculating average confidence
        logger.info(f"Using tracker: {self._yolo_params.get('tracker', 'default')}")

        # Process each frame with progress tracking
        with tqdm(frame_files, desc=f"Processing {video_id}", unit="frame") as pbar:
            start_time = time.time()
            for frame_idx, frame_file in enumerate(pbar):
                try:
                    # Run YOLO detection and tracking with cached parameters
                    # persist=True maintains track IDs across frames
                    track_results = self.yolo_model.track(
                        source=str(frame_file),
                        persist=True,  # Maintain tracking across frames
                        **self._yolo_params  # Use pre-cached parameters
                    )

                    # Process YOLO results into standardized annotation format
                    frame_annotation = self._process_frame_results(
                        track_results[0], frame_idx, frame_file, results["statistics"]
                    )

                    # Apply optimized batch SAM segmentation if enabled and detections exist
                    if (CONFIG.sam_enabled and self.sam_model and frame_annotation["detections"]):
                        frame_annotation = self._apply_batch_sam_segmentation(frame_file, frame_annotation)

                    # Accumulate confidence scores for average calculation
                    for det in frame_annotation["detections"]:
                        confidence_sum += det["confidence"]

                    # Store frame results
                    results["annotations"].append(frame_annotation)
                    results["statistics"]["processed_frames"] += 1

                    # Periodic GPU memory cleanup to prevent OOM errors
                    # Clean every 50 frames to balance performance and memory usage
                    if frame_idx % 50 == 0 and frame_idx > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    # Log frame processing errors but continue with next frame
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue
            processing_time = time.time() - start_time
            results["statistics"]["processing_time"] = processing_time
        moving_cars_count = 0
        static_cars_count = 0
        # Post-processing: Calculate static cars based on movement analysis
        if CONFIG.static_car_enabled:
            for track_info in self.tracks.values():
                if track_info.get('class_name') != 'car':
                    continue

                # Якщо прапорець стоїть - машина точно рухома.
                if track_info['is_moving']:
                    moving_cars_count += 1
                else:
                    # Якщо прапорець не стоїть, вмикаємо наш "фільтр від листя".
                    duration = track_info['last_seen_frame'] - track_info['start_frame']
                    if duration >= CONFIG.min_static_duration:
                        static_cars_count += 1
                    # Якщо тривалість менша - ігноруємо. Це і є захист від "листка".

            # Записуємо фінальний результат
            results["statistics"]["moving_cars_count"] = moving_cars_count
            results["statistics"]["static_cars_count"] = static_cars_count

        # Finalize statistics calculations
        if results["statistics"]["total_detections"] > 0:
            results["statistics"]["avg_confidence"] = confidence_sum / results["statistics"]["total_detections"]

        logger.info(f"Completed {video_id}: {results['statistics']}")
        return results


    def _process_frame_results(self, yolo_result, frame_idx: int, frame_file: Path, stats: Dict) -> Dict:
        """
        Process YOLO detection results for a single frame into standardized format.

        Converts YOLO output tensors into structured annotations with bounding boxes,
        confidence scores, class information, and basic segmentation if available.

        Args:
            yolo_result: YOLO model output containing detections
            frame_idx (int): Current frame index in video sequence
            frame_file (Path): Path to current frame image file
            stats (Dict): Statistics dictionary to update

        Returns:
            Dict: Structured frame annotation containing:
                - frame_id: Frame sequence number
                - file_name: Original frame filename
                - detections: List of detection objects with bbox, class, confidence, etc.
        """
        # Initialize frame annotation structure
        frame_annotation = {
            "frame_id": frame_idx,
            "file_name": frame_file.name,
            "detections": []
        }

        # Process detection results if any exist
        if yolo_result.boxes is not None:
            # Convert tensor data to numpy for easier manipulation
            boxes = yolo_result.boxes.cpu().numpy()

            for i, box in enumerate(boxes.data):
                # Ensure box has all required fields
                # Expected format: [x1, y1, x2, y2, track_id, confidence, class_id]
                # Apply confidence filtering to reduce false positives
                if len(box) < 7 or box[5] < CONFIG.min_confidence_for_tracking:
                    continue

                # Extract detection parameters
                x1, y1, x2, y2 = box[:4]  # Bounding box coordinates
                track_id = int(box[4])  # Unique tracking ID
                conf = box[5]  # Detection confidence score
                cls_id = int(box[6])  # Class identifier

                # Map class ID to human-readable name
                class_name = CONFIG.custom_classes.get(cls_id, f'class_{cls_id}')

                # Extract basic segmentation mask from YOLO if available
                # YOLO models may include instance segmentation masks
                mask = None
                if hasattr(yolo_result, 'masks') and yolo_result.masks is not None:
                    if i < len(yolo_result.masks.data):
                        mask_data = yolo_result.masks.data[i].cpu().numpy()
                        mask = self._approximate_segmentation(mask_data)

                # Create standardized detection object
                detection = {
                    "track_id": track_id,  # Unique tracking identifier
                    "class_id": cls_id,  # Numeric class identifier
                    "class_name": class_name,  # Human-readable class name
                    "confidence": float(conf),  # Detection confidence [0-1]
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # [x, y, width, height]
                    "segmentation": mask,  # Polygon segmentation (if available)
                    "area": float((x2 - x1) * (y2 - y1)),  # Bounding box area
                    "sam_applied": False  # Flag indicating SAM processing
                }

                frame_annotation["detections"].append(detection)

                # Update aggregate statistics
                stats["total_detections"] += 1
                stats["unique_tracks"][class_name.lower()].add(track_id)

                # Оновлюємо, коли востаннє бачили трек (це дешева операція)
                if track_id in self.tracks:
                    self.tracks[track_id]['last_seen_frame'] = frame_idx

                if class_name.lower() == 'car':
                    stats["cars_count"] += 1

                    is_new_track = track_id not in self.tracks

                    # Визначаємо, чи потрібно перевіряти цей трек саме зараз
                    should_check = False
                    if not is_new_track:
                        # Перевіряємо, тільки якщо він ще не позначений як рухомий
                        if not self.tracks[track_id]['is_moving']:
                            # Твоя проста і зрозуміла перевірка по інтервалу
                            if frame_idx % CONFIG.static_check_interval == 0:
                                should_check = True

                    # Викликаємо "аналітика" тільки в потрібний момент
                    if is_new_track or should_check:
                        self._update_car_mobility(track_id, [x1, y1, x2, y2], frame_idx)

                # Оновлюємо лічильники для інших класів
                elif class_name.lower() == 'person':
                    stats["people_count"] += 1
                elif class_name.lower() == 'pet':
                    stats["pets_count"] += 1

        return frame_annotation


    def _apply_batch_sam_segmentation(self, frame_file: Path, frame_annotation: Dict) -> Dict:
        """
        Apply optimized batch SAM segmentation to improve mask quality.

        This method implements an efficient batch processing approach for SAM:
        - Processes all detections in a single SAM call
        - Uses cached parameters to avoid repeated configuration
        - Reads the image only once per frame
        - Provides fallback to individual processing if batch fails

        Args:
            frame_file (Path): Path to the current frame image
            frame_annotation (Dict): Frame annotation with detections to process

        Returns:
            Dict: Updated frame annotation with improved segmentation masks

        Optimization Benefits:
        - Reduced I/O operations (single image read)
        - Fewer SAM model calls (batch vs individual)
        - Better GPU utilization through batch processing
        - Consistent parameter usage across detections
        """
        try:
            # Read image once for all SAM operations (major I/O optimization)
            image = cv2.imread(str(frame_file))
            if image is None:
                logger.warning(f"Could not read image: {frame_file}")
                return frame_annotation

            # Prepare all bounding boxes for batch SAM processing
            all_boxes = []  # Bounding box prompts for SAM
            detection_indices = []  # Corresponding detection indices

            for det_idx, detection in enumerate(frame_annotation["detections"]):
                bbox = detection["bbox"]
                x, y, w, h = bbox
                # Convert from [x, y, width, height] to [x1, y1, x2, y2] format expected by SAM
                box_prompt = [x, y, x + w, y + h]
                all_boxes.append(box_prompt)
                detection_indices.append(det_idx)

            try:
                # Batch SAM prediction for all boxes simultaneously
                # This is significantly faster than individual predictions
                sam_results = self.sam_model.predict(
                    source=image,
                    bboxes=all_boxes,  # Process all bounding boxes at once
                    **self._sam_params  # Use cached parameters for consistency
                )

                # Process batch prediction results
                if sam_results and hasattr(sam_results[0], 'masks') and sam_results[0].masks is not None:
                    masks_data = sam_results[0].masks.data

                    # Apply each mask to corresponding detection
                    for i, det_idx in enumerate(detection_indices):
                        if i < len(masks_data):
                            try:
                                # Extract and process individual mask
                                mask_data = masks_data[i].cpu().numpy()
                                segmentation = self._approximate_segmentation(mask_data)

                                # Update detection with improved segmentation
                                if segmentation:
                                    detection = frame_annotation["detections"][det_idx]
                                    detection["segmentation"] = segmentation
                                    detection["area"] = float(np.sum(mask_data > 0.5))  # Actual segmented area
                                    detection["sam_applied"] = True

                            except Exception as e:
                                logger.debug(f"Failed to process mask {i}: {e}")
                                continue

            except Exception as e:
                logger.debug(f"Batch SAM failed: {e}")
                # Fallback to individual processing if batch fails
                return self._fallback_individual_sam(image, frame_annotation)

        except Exception as e:
            logger.warning(f"SAM segmentation failed for {frame_file.name}: {e}")

        return frame_annotation


    def _fallback_individual_sam(self, image: np.ndarray, frame_annotation: Dict) -> Dict:
        """
        Fallback method for individual SAM processing when batch processing fails.

        This method processes each detection individually, which is slower but
        more robust when batch processing encounters errors.

        Args:
            image (np.ndarray): Loaded frame image
            frame_annotation (Dict): Frame annotation to update

        Returns:
            Dict: Updated frame annotation with individual SAM results

        Note:
            This is a fallback mechanism - batch processing is preferred for performance
        """
        logger.debug("Using individual SAM processing as fallback")

        # Process each detection individually
        for det_idx, detection in enumerate(frame_annotation["detections"]):
            bbox = detection["bbox"]
            x, y, w, h = bbox
            # Convert to SAM expected format
            box_prompt = [x, y, x + w, y + h]

            try:
                # Individual SAM prediction
                sam_results = self.sam_model.predict(
                    source=image,
                    bboxes=[box_prompt],  # Single bounding box
                    **self._sam_params
                )

                # Process individual result
                if (sam_results and hasattr(sam_results[0], 'masks') and
                        sam_results[0].masks is not None and len(sam_results[0].masks.data) > 0):

                    mask_data = sam_results[0].masks.data[0].cpu().numpy()
                    segmentation = self._approximate_segmentation(mask_data)

                    # Update detection with segmentation if successful
                    if segmentation:
                        frame_annotation["detections"][det_idx]["segmentation"] = segmentation
                        frame_annotation["detections"][det_idx]["area"] = float(np.sum(mask_data > 0.5))
                        frame_annotation["detections"][det_idx]["sam_applied"] = True

            except Exception as e:
                logger.debug(f"Individual SAM failed for detection {det_idx}: {e}")
                continue

        return frame_annotation


    def _approximate_segmentation(self, mask: np.ndarray) -> Optional[List[List[float]]]:
        """
        Optimized segmentation approximation with improved speed and quality.

        Converts binary masks to polygon approximations using advanced techniques:
        - Fast binarization and normalization
        - Morphological closing instead of fillPoly for hole filling
        - Adaptive epsilon based on contour perimeter
        - Vectorized smoothing operations

        Args:
            mask (np.ndarray): Binary or probability mask from segmentation model

        Returns:
            Optional[List[List[float]]]: Polygon coordinates as list of [x,y] pairs,
                                       or None if processing fails or area too small

        Optimization Features:
        - Efficient mask binarization
        - Morphological operations for hole filling
        - Adaptive polygon approximation
        - Intelligent point reduction
        - Optional smoothing for complex polygons
        """
        if mask is None or mask.size == 0:
            return None

        try:
            # Handle multi-dimensional masks (e.g., from batch processing)
            if len(mask.shape) == 3:
                mask = mask[0]  # Take first channel/batch

            # Efficient binarization based on mask type
            if mask.dtype != np.uint8:
                # Probability mask - threshold at 0.5
                mask = (mask > 0.5).astype(np.uint8)
            elif not np.all((mask == 0) | (mask == 1)):
                # Non-binary uint8 mask - convert to binary
                mask = (mask > 0).astype(np.uint8)

            # Optimized hole filling using morphological closing
            # This is faster than fillPoly and more predictable
            if CONFIG.fill_holes:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morphology_kernel)

            # Find external contours (we want object boundaries, not holes)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Efficiently find largest contour
            if len(contours) == 1:
                # Optimization: skip area calculation if only one contour
                largest_contour = contours[0]
                area = cv2.contourArea(largest_contour)
            else:
                # Calculate areas and find maximum
                areas = [cv2.contourArea(c) for c in contours]
                max_idx = np.argmax(areas)
                largest_contour = contours[max_idx]
                area = areas[max_idx]

            # Filter out tiny segmentations that are likely noise
            if area < self.min_area:
                return None

            # Adaptive polygon approximation based on contour complexity
            perimeter = cv2.arcLength(largest_contour, True)
            # Use percentage of perimeter instead of fixed tolerance
            # This scales appropriately for objects of different sizes
            epsilon = max(self.tolerance, perimeter * 0.002)  # Minimum 0.2% of perimeter

            # Douglas-Peucker polygon approximation
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            polygon = approx.flatten().tolist()

            # Intelligent point reduction for overly complex polygons
            if len(polygon) > self.max_points * 2:  # *2 because polygon is [x,y,x,y,...]
                polygon = self._reduce_points_optimized(polygon, self.max_points)

            # Apply smoothing only for moderately complex polygons
            # Avoid smoothing very simple or very complex shapes
            # todo constants
            if CONFIG.smoothing and 8 <= len(polygon) <= 100:
                polygon = self._smooth_polygon_optimized(polygon)

            # Return polygon if it has enough points for a valid shape (minimum triangle)
            return [polygon] if len(polygon) >= 6 else None

        except Exception as e:
            logger.warning(f"Segmentation approximation failed: {e}")
            return None


    def _reduce_points_optimized(self, polygon: List[float], max_points: int) -> List[float]:
        """
        Optimized point reduction with uniform distribution along perimeter.

        Reduces polygon complexity while maintaining shape characteristics by
        selecting points that are evenly distributed along the contour perimeter.

        Args:
            polygon (List[float]): Original polygon as flat list [x1,y1,x2,y2,...]
            max_points (int): Maximum number of points to retain

        Returns:
            List[float]: Reduced polygon with uniform point distribution

        Algorithm:
            Uses linear interpolation to select evenly spaced points along
            the original contour, preserving overall shape while reducing complexity.
        """
        if len(polygon) <= max_points * 2:
            return polygon

        # Convert flat list to point pairs for easier processing
        points = np.array([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)])

        # Use uniform distribution along contour
        n_points = len(points)
        # Select evenly spaced indices to maintain shape characteristics
        indices = np.linspace(0, n_points - 1, max_points, dtype=int)

        # Extract reduced point set
        reduced_points = points[indices]

        # Convert back to flat coordinate list
        return [coord for point in reduced_points for coord in point]


    def _smooth_polygon_optimized(self, polygon: List[float]) -> List[float]:
        """
        Vectorized polygon smoothing with adaptive window sizing.

        Applies moving average smoothing to polygon vertices to reduce noise
        and create more natural-looking contours. Window size adapts to polygon complexity.

        Args:
            polygon (List[float]): Input polygon as flat coordinate list

        Returns:
            List[float]: Smoothed polygon coordinates

        Algorithm:
            - Converts to point array for vectorized operations
            - Applies adaptive windowing based on polygon complexity
            - Uses numpy operations for efficient computation
            - Preserves overall polygon shape while reducing vertex noise
        """
        if len(polygon) < 8:  # Skip smoothing for very simple polygons
            return polygon

        # Convert to point array for vectorized operations
        points = np.array([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)])
        smoothed = np.zeros_like(points)

        # Adaptive window size based on polygon complexity
        # More complex polygons get larger smoothing windows
        window = min(3, max(1, len(points) // 6))

        # Apply moving average smoothing with boundary handling
        for i in range(len(points)):
            # Calculate window boundaries with proper edge handling
            start_idx = max(0, i - window // 2)
            end_idx = min(len(points), i + window // 2 + 1)

            # Vectorized mean calculation for smoothing
            smoothed[i] = np.mean(points[start_idx:end_idx], axis=0)

        # Convert back to flat coordinate list
        return [coord for point in smoothed for coord in point]


    def _update_car_mobility(self, track_id: int, bbox: List[float], frame_idx: int):
        """Створює або перевіряє статус мобільності автомобіля."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        current_center = (center_x, center_y)

        # Якщо це новий трек - створюємо для нього "паспорт"
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'class_name': 'car',  # Ми вже знаємо, що це машина
                'start_center': current_center,  # Якір. Встановлюється один раз.
                'start_frame': frame_idx,
                'is_moving': False,
                'last_seen_frame': frame_idx
            }
            return

        # Якщо ми тут, значить, настав час перевірки для існуючого треку.
        track_info = self.tracks[track_id]

        # Розраховуємо зміщення від початкової точки.
        displacement = np.sqrt(
            (current_center[0] - track_info['start_center'][0]) ** 2 +
            (current_center[1] - track_info['start_center'][1]) ** 2
        )

        # Якщо зміщення перевищило поріг - ставимо прапорець. Назавжди.
        if displacement > CONFIG.movement_threshold:
            track_info['is_moving'] = True


    def clear_memory(self):
        """
        Clear memory and cache after video processing.

        Performs cleanup operations to prevent memory accumulation during
        batch video processing. This is especially important for GPU memory
        management and long processing sessions.

        Cleanup Operations:
        - Clear GPU cache if available
        - Reset tracking history
        - Free temporary data structures

        Note:
            Should be called after each video is fully processed,
            especially when processing multiple videos in sequence.
        """
        if CONFIG.clear_cache_after_video:
            # Clear GPU memory cache to prevent OOM errors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear tracking history to free memory
            self.tracks.clear()