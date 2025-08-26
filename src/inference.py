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
    """Handles object detection, tracking, and segmentation for video frames.

    Combines YOLO detection/tracking with optional SAM segmentation to create
    detailed annotations with polygon approximation and car mobility analysis.
    """

    def __init__(self):
        self.tracks = {}  # Track ID -> metadata for mobility analysis

        self.max_points = CONFIG.max_points
        # Douglas-Peucker approximation tolerance for polygon simplification
        self.min_area = CONFIG.min_area

        self._sam_params = None
        if CONFIG.sam_enabled:
            self._sam_params = CONFIG.get_sam_params()

        self._yolo_params = CONFIG.get_yolo_params()

        self._load_models()

    def _load_models(self):
        """Load YOLO and optionally SAM models with proper error handling."""
        logger.info("Loading models...")

        try:
            yolo_path = get_yolo_model_path(CONFIG.yolo_model_path)
            logger.info(f"Loading YOLO model: {yolo_path}")

            self.yolo_model = YOLO(yolo_path)
            self.yolo_model.to(CONFIG.device)
            logger.info(f"YOLO model loaded successfully: {Path(yolo_path).name}")

            tracker_path = CONFIG.get_tracker_path()
            logger.info(f"Tracker configuration path: {tracker_path}")

            if Path(tracker_path).exists():
                logger.info(f"Tracker configuration found: {Path(tracker_path).name}")
            else:
                logger.info(f"Using built-in tracker: {tracker_path}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

        self.sam_model = None
        if CONFIG.sam_enabled:
            try:
                sam_path = get_sam_model_path(CONFIG.sam_model_path)
                logger.info(f"Loading SAM model: {sam_path}")

                if Path(sam_path).exists():
                    self.sam_model = SAM(sam_path)
                    self.sam_model.to(CONFIG.device)
                    logger.info(f"SAM model loaded successfully: {Path(sam_path).name}")
                else:
                    logger.warning(f"SAM model not found: {sam_path}, disabling SAM")
                    CONFIG.sam_enabled = False

            except Exception as e:
                logger.error(f"Failed to load SAM model: {e}")
                logger.warning("Disabling SAM segmentation")
                CONFIG.sam_enabled = False
                self.sam_model = None

    def process_video(self, video_id: str, video_info: Dict) -> Dict[str, Any]:
        """Process all frames of a video for detection, tracking, and segmentation.

        Args:
            video_id: Unique identifier for the video
            video_info: Metadata including frames directory path

        Returns:
            Complete annotation results with statistics
        """
        logger.info(f"Processing video: {video_id}")

        # Reset predictor state to avoid memory issues between videos
        if hasattr(self.yolo_model, 'predictor') and self.yolo_model.predictor:
            self.yolo_model.predictor = None
        self.tracks = {}

        frames_dir = Path(video_info['frames_dir'])

        if not frames_dir.exists():
            logger.error(f"Frames directory not found: {frames_dir}")
            return {}

        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            logger.error(f"No frames found in: {frames_dir}")
            return {}

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

        confidence_sum = 0.0
        logger.info(f"Using tracker: {self._yolo_params.get('tracker', 'default')}")

        with tqdm(frame_files, desc=f"Processing {video_id}", unit="frame") as pbar:
            start_time = time.time()
            for frame_idx, frame_file in enumerate(pbar):
                try:
                    track_results = self.yolo_model.track(
                        source=str(frame_file),
                        persist=True,  # Maintain track IDs across frames
                        **self._yolo_params
                    )

                    frame_annotation = self._process_frame_results(
                        track_results[0], frame_idx, frame_file, results["statistics"]
                    )

                    # Apply SAM segmentation if enabled and detections exist
                    if (CONFIG.sam_enabled and self.sam_model and frame_annotation["detections"]):
                        frame_annotation = self._apply_batch_sam_segmentation(frame_file, frame_annotation)

                    for det in frame_annotation["detections"]:
                        confidence_sum += det["confidence"]

                    results["annotations"].append(frame_annotation)
                    results["statistics"]["processed_frames"] += 1

                    # Periodic memory cleanup to prevent CUDA OOM
                    if frame_idx % 50 == 0 and frame_idx > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue

            processing_time = time.time() - start_time
            results["statistics"]["processing_time"] = processing_time

        # Analyze car mobility after processing all frames
        moving_cars_count = 0
        static_cars_count = 0
        if CONFIG.static_car_enabled:
            for track_info in self.tracks.values():
                if track_info.get('class_name') != 'car':
                    continue

                if track_info['is_moving']:
                    moving_cars_count += 1
                else:
                    # Only count as static if observed for minimum duration
                    duration = track_info['last_seen_frame'] - track_info['start_frame']
                    if duration >= CONFIG.min_static_duration:
                        static_cars_count += 1

            results["statistics"]["moving_cars_count"] = moving_cars_count
            results["statistics"]["static_cars_count"] = static_cars_count

        if results["statistics"]["total_detections"] > 0:
            results["statistics"]["avg_confidence"] = confidence_sum / results["statistics"]["total_detections"]

        logger.info(f"Completed {video_id}: {results['statistics']}")
        return results

    def _process_frame_results(self, yolo_result, frame_idx: int, frame_file: Path, stats: Dict) -> Dict:
        """Convert YOLO tracking results to standardized detection format."""
        frame_annotation = {
            "frame_id": frame_idx,
            "file_name": frame_file.name,
            "detections": []
        }

        if yolo_result.boxes is not None:
            boxes = yolo_result.boxes.cpu().numpy()

            for i, box in enumerate(boxes.data):
                # Skip detections below confidence threshold or malformed boxes
                if len(box) < 7 or box[5] < CONFIG.min_confidence_for_tracking:
                    continue

                x1, y1, x2, y2 = box[:4]
                track_id = int(box[4])
                conf = box[5]
                cls_id = int(box[6])

                class_name = CONFIG.custom_classes.get(cls_id, f'class_{cls_id}')

                detection = {
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    # COCO format: [x, y, width, height]
                    "segmentation": None,
                    "area": float((x2 - x1) * (y2 - y1)),
                    "sam_applied": False
                }

                frame_annotation["detections"].append(detection)

                stats["total_detections"] += 1
                stats["unique_tracks"][class_name.lower()].add(track_id)

                # Update track last seen frame
                if track_id in self.tracks:
                    self.tracks[track_id]['last_seen_frame'] = frame_idx

                # Handle class-specific processing
                if class_name.lower() == 'car':
                    stats["cars_count"] += 1

                    is_new_track = track_id not in self.tracks

                    # Check car mobility: new tracks always, existing static tracks periodically
                    should_check = False
                    if not is_new_track:
                        if not self.tracks[track_id]['is_moving']:
                            if frame_idx % CONFIG.static_check_interval == 0:
                                should_check = True

                    if is_new_track or should_check:
                        self._update_car_mobility(track_id, [x1, y1, x2, y2], frame_idx)

                elif class_name.lower() == 'person':
                    stats["people_count"] += 1
                elif class_name.lower() == 'pet':
                    stats["pets_count"] += 1

        return frame_annotation

    def _apply_batch_sam_segmentation(self, frame_file: Path, frame_annotation: Dict) -> Dict:
        """Apply SAM segmentation to all detections in a frame using batch processing for efficiency."""
        try:
            image = cv2.imread(str(frame_file))
            if image is None:
                logger.warning(f"Could not read image: {frame_file}")
                return frame_annotation

            # Prepare batch inputs for SAM
            all_boxes = []
            detection_indices = []

            for det_idx, detection in enumerate(frame_annotation["detections"]):
                bbox = detection["bbox"]
                x, y, w, h = bbox

                box_prompt = [x, y, x + w, y + h]
                all_boxes.append(box_prompt)
                detection_indices.append(det_idx)

            try:
                sam_results = self.sam_model.predict(
                    source=image,
                    bboxes=all_boxes,
                    **self._sam_params
                )

                if sam_results and hasattr(sam_results[0], 'masks') and sam_results[0].masks is not None:
                    masks_data = sam_results[0].masks.data

                    for i, det_idx in enumerate(detection_indices):
                        if i < len(masks_data):
                            try:
                                mask_data = masks_data[i].cpu().numpy()
                                segmentation = self._approximate_segmentation(mask_data)


                                if segmentation:
                                    detection = frame_annotation["detections"][det_idx]
                                    detection["segmentation"] = segmentation
                                    detection["area"] = float(np.sum(mask_data > 0.5))
                                    detection["sam_applied"] = True

                            except Exception as e:
                                logger.debug(f"Failed to process mask {i} for det {det_idx}: {e}")
                                continue

            except Exception as e:
                logger.debug(f"Batch SAM failed: {e}")
                return None

        except Exception as e:
            logger.warning(f"SAM segmentation failed for {frame_file.name}: {e}")

        return frame_annotation


    def _approximate_segmentation(self, mask: np.ndarray) -> Optional[List[List[float]]]:
        """Convert binary mask to polygon with guaranteed point limit.

        Uses a two-stage approach:
        1. Douglas-Peucker approximation with binary search for optimal epsilon
        2. Uniform sampling fallback if needed
        """
        if mask is None or mask.size == 0:
            return None

        try:
            # Handle 3D masks by taking first channel
            if len(mask.shape) == 3:
                mask = mask[0]

            # Ensure binary mask
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
            elif mask.max() > 1:
                mask = (mask > 0).astype(np.uint8)

            # Optional morphological closing for hole filling
            if CONFIG.fill_holes:
                # Adaptive kernel size based on object size
                mask_area = np.sum(mask)
                kernel_size = max(3, min(9, int(np.sqrt(mask_area) / 30)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours with simple approximation for efficiency
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < self.min_area:
                return None

            # Use binary search to find optimal epsilon for Douglas-Peucker
            polygon = self._douglas_peucker_with_limit(largest_contour, self.max_points)

            # Fallback to uniform sampling if Douglas-Peucker still produces too many points
            if len(polygon) > self.max_points * 2:
                polygon = self._uniform_sample_polygon_fixed(polygon, self.max_points)

            return [polygon] if len(polygon) >= 6 else None

        except Exception as e:
            logger.warning(f"Segmentation approximation failed: {e}")
            return None


    def _douglas_peucker_with_limit(self, contour: np.ndarray, max_points: int) -> List[float]:
        """Apply Douglas-Peucker with binary search to find optimal epsilon."""

        # Calculate perimeter-based epsilon bounds
        perimeter = cv2.arcLength(contour, True)

        # Binary search bounds for epsilon
        min_epsilon = 0.1
        max_epsilon = perimeter * 0.1  # Up to 10% of perimeter

        best_polygon = None

        # Binary search for optimal epsilon
        for _ in range(10):  # Max 10 iterations
            epsilon = (min_epsilon + max_epsilon) / 2

            approx = cv2.approxPolyDP(contour, epsilon, True)
            num_points = len(approx)

            if num_points <= max_points:
                # Good approximation, try to reduce epsilon for better quality
                best_polygon = approx.flatten().tolist()
                max_epsilon = epsilon

                if num_points >= max_points * 0.8:  # Close to target, stop
                    break
            else:
                # Too many points, increase epsilon
                min_epsilon = epsilon

        # If binary search failed, use aggressive epsilon
        if best_polygon is None:
            epsilon = perimeter * 0.05
            approx = cv2.approxPolyDP(contour, epsilon, True)
            best_polygon = approx.flatten().tolist()

        return best_polygon


    def _uniform_sample_polygon_fixed(self, polygon: List[float], max_points: int) -> List[float]:
        """Fixed uniform sampling that guarantees exact point count."""
        if len(polygon) <= max_points * 2:
            return polygon

        # Convert to points array
        points = np.array([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)])

        if len(points) <= max_points:
            return polygon

        # Simple uniform sampling by index
        step = len(points) / max_points
        indices = [int(i * step) for i in range(max_points)]

        # Ensure we don't exceed array bounds and get exactly max_points
        indices = list(set(indices))  # Remove duplicates
        indices.sort()

        # If we have fewer indices than max_points, fill with additional points
        while len(indices) < max_points and len(indices) < len(points):
            # Find largest gap and insert point there
            gaps = [(indices[i + 1] - indices[i], i) for i in range(len(indices) - 1)]
            if gaps:
                _, gap_idx = max(gaps)
                new_idx = (indices[gap_idx] + indices[gap_idx + 1]) // 2
                if new_idx not in indices:
                    indices.insert(gap_idx + 1, new_idx)
                    indices.sort()

        # Take exactly max_points
        indices = indices[:max_points]

        sampled_points = points[indices]
        return [coord for point in sampled_points for coord in point]


    def _update_car_mobility(self, track_id: int, bbox: List[float], frame_idx: int):
        """Track car movement by analyzing center point displacement over time."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        current_center = (center_x, center_y)

        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'class_name': 'car',
                'start_center': current_center,
                'start_frame': frame_idx,
                'is_moving': False,
                'last_seen_frame': frame_idx
            }
            return

        track_info = self.tracks[track_id]

        # Calculate displacement from initial position
        displacement = np.sqrt(
            (current_center[0] - track_info['start_center'][0]) ** 2 +
            (current_center[1] - track_info['start_center'][1]) ** 2
        )

        # Mark as moving if displacement exceeds threshold
        if displacement > CONFIG.movement_threshold:
            track_info['is_moving'] = True

    def clear_memory(self):
        """Clear GPU cache and tracking data to free memory between videos."""
        if CONFIG.clear_cache_after_video:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.tracks.clear()