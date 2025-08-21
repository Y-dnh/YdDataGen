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
    def __init__(self):
        self.tracks = {}
        self.annotation_id = 1
        self.image_id = 1

        self.max_points = CONFIG.max_points
        # todo Douglas-Peucker approximation tolerance explain
        self.tolerance = CONFIG.simplify_tolerance
        self.min_area = CONFIG.min_area

        self._sam_params = None
        if CONFIG.sam_enabled:
            self._sam_params = CONFIG.get_sam_params()

        self._yolo_params = CONFIG.get_yolo_params()

        if CONFIG.fill_holes:
            # todo ksize explain
            self._morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self._load_models()


    def _load_models(self):
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

        logger.info(f"Processing video: {video_id}")
        # todo predictor explain
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
                        persist=True,
                        **self._yolo_params
                    )

                    frame_annotation = self._process_frame_results(
                        track_results[0], frame_idx, frame_file, results["statistics"]
                    )

                    if (CONFIG.sam_enabled and self.sam_model and frame_annotation["detections"]):
                        frame_annotation = self._apply_batch_sam_segmentation(frame_file, frame_annotation)

                    for det in frame_annotation["detections"]:
                        confidence_sum += det["confidence"]

                    results["annotations"].append(frame_annotation)
                    results["statistics"]["processed_frames"] += 1

                    #todo memory check
                    if frame_idx % 50 == 0 and frame_idx > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue
            processing_time = time.time() - start_time
            results["statistics"]["processing_time"] = processing_time
        moving_cars_count = 0
        static_cars_count = 0
        if CONFIG.static_car_enabled:
            for track_info in self.tracks.values():
                if track_info.get('class_name') != 'car':
                    continue

                if track_info['is_moving']:
                    moving_cars_count += 1
                else:
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

        frame_annotation = {
            "frame_id": frame_idx,
            "file_name": frame_file.name,
            "detections": []
        }

        if yolo_result.boxes is not None:
            boxes = yolo_result.boxes.cpu().numpy()

            for i, box in enumerate(boxes.data):
                if len(box) < 7 or box[5] < CONFIG.min_confidence_for_tracking:
                    continue

                x1, y1, x2, y2 = box[:4]
                track_id = int(box[4])
                conf = box[5]
                cls_id = int(box[6])

                class_name = CONFIG.custom_classes.get(cls_id, f'class_{cls_id}')

                mask = None
                if hasattr(yolo_result, 'masks') and yolo_result.masks is not None:
                    if i < len(yolo_result.masks.data):
                        mask_data = yolo_result.masks.data[i].cpu().numpy()
                        mask = self._approximate_segmentation(mask_data)

                detection = {
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "segmentation": mask,
                    "area": float((x2 - x1) * (y2 - y1)),
                    "sam_applied": False
                }

                frame_annotation["detections"].append(detection)

                stats["total_detections"] += 1
                stats["unique_tracks"][class_name.lower()].add(track_id)

                if track_id in self.tracks:
                    self.tracks[track_id]['last_seen_frame'] = frame_idx

                if class_name.lower() == 'car':
                    stats["cars_count"] += 1

                    is_new_track = track_id not in self.tracks

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
        try:
            image = cv2.imread(str(frame_file))
            if image is None:
                logger.warning(f"Could not read image: {frame_file}")
                return frame_annotation

            all_boxes = []
            all_points = []
            all_point_labels = []
            detection_indices = []

            for det_idx, detection in enumerate(frame_annotation["detections"]):
                bbox = detection["bbox"]
                x, y, w, h = bbox

                box_prompt = [x, y, x + w, y + h]
                all_boxes.append(box_prompt)
                detection_indices.append(det_idx)

                if CONFIG.sam_add_center_point and detection["class_name"] == "car":
                    center_x = x + w / 2
                    center_y = y + h / 2
                    all_points.append([center_x, center_y])
                    all_point_labels.append(1)

            points_to_predict = np.array(all_points) if all_points else None
            labels_to_predict = np.array(all_point_labels) if all_point_labels else None

            try:
                sam_results = self.sam_model.predict(
                    source=image,
                    bboxes=all_boxes,
                    # points=points_to_predict,
                    # labels=labels_to_predict,
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
                return self._fallback_individual_sam(image, frame_annotation)

        except Exception as e:
            logger.warning(f"SAM segmentation failed for {frame_file.name}: {e}")

        return frame_annotation



    def _fallback_individual_sam(self, image: np.ndarray, frame_annotation: Dict) -> Dict:
        logger.debug("Using individual SAM processing as fallback")

        for det_idx, detection in enumerate(frame_annotation["detections"]):
            bbox = detection["bbox"]
            x, y, w, h = bbox
            box_prompt = [x, y, x + w, y + h]

            try:
                sam_results = self.sam_model.predict(
                    source=image,
                    bboxes=[box_prompt],
                    **self._sam_params
                )

                if (sam_results and hasattr(sam_results[0], 'masks') and
                        sam_results[0].masks is not None and len(sam_results[0].masks.data) > 0):

                    mask_data = sam_results[0].masks.data[0].cpu().numpy()
                    segmentation = self._approximate_segmentation(mask_data)

                    if segmentation:
                        frame_annotation["detections"][det_idx]["segmentation"] = segmentation
                        frame_annotation["detections"][det_idx]["area"] = float(np.sum(mask_data > 0.5))
                        frame_annotation["detections"][det_idx]["sam_applied"] = True

            except Exception as e:
                logger.debug(f"Individual SAM failed for detection {det_idx}: {e}")
                continue

        return frame_annotation


    def _approximate_segmentation(self, mask: np.ndarray) -> Optional[List[List[float]]]:

        if mask is None or mask.size == 0:
            return None

        try:
            if len(mask.shape) == 3:
                mask = mask[0]

            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
            elif not np.all((mask == 0) | (mask == 1)):
                mask = (mask > 0).astype(np.uint8)

            if CONFIG.fill_holes:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morphology_kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            if len(contours) == 1:
                largest_contour = contours[0]
                area = cv2.contourArea(largest_contour)
            else:
                areas = [cv2.contourArea(c) for c in contours]
                max_idx = np.argmax(areas)
                largest_contour = contours[max_idx]
                area = areas[max_idx]

            if area < self.min_area:
                return None

            perimeter = cv2.arcLength(largest_contour, True)
            epsilon = max(self.tolerance, perimeter * 0.002)

            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            polygon = approx.flatten().tolist()

            if len(polygon) > self.max_points * 2:
                polygon = self._reduce_points_optimized(polygon, self.max_points)

            # todo constants
            if CONFIG.smoothing and 8 <= len(polygon) <= 100:
                polygon = self._smooth_polygon_optimized(polygon)

            return [polygon] if len(polygon) >= 6 else None

        except Exception as e:
            logger.warning(f"Segmentation approximation failed: {e}")
            return None


    def _reduce_points_optimized(self, polygon: List[float], max_points: int) -> List[float]:

        if len(polygon) <= max_points * 2:
            return polygon

        points = np.array([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)])

        n_points = len(points)
        indices = np.linspace(0, n_points - 1, max_points, dtype=int)

        reduced_points = points[indices]

        return [coord for point in reduced_points for coord in point]


    def _smooth_polygon_optimized(self, polygon: List[float]) -> List[float]:
        if len(polygon) < 8:
            return polygon

        points = np.array([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)])
        smoothed = np.zeros_like(points)

        window = min(3, max(1, len(points) // 6))

        for i in range(len(points)):

            start_idx = max(0, i - window // 2)
            end_idx = min(len(points), i + window // 2 + 1)

            smoothed[i] = np.mean(points[start_idx:end_idx], axis=0)

        return [coord for point in smoothed for coord in point]


    def _update_car_mobility(self, track_id: int, bbox: List[float], frame_idx: int):
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

        displacement = np.sqrt(
            (current_center[0] - track_info['start_center'][0]) ** 2 +
            (current_center[1] - track_info['start_center'][1]) ** 2
        )

        if displacement > CONFIG.movement_threshold:
            track_info['is_moving'] = True


    def clear_memory(self):
        if CONFIG.clear_cache_after_video:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.tracks.clear()