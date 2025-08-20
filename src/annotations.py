import json
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
from pycocotools.coco import COCO

from src.config import CONFIG

logger = logging.getLogger(__name__)


class COCOAnnotationGenerator:
    """Simplified COCO annotations generator using pycocotools standard."""

    def __init__(self):
        self.annotation_id = 1
        self.image_id = 1
        self.video_id_counter = 0

    def create_coco_dataset(self, video_results: Dict, video_id: str, video_info: Dict) -> Dict[str, Any]:
        """Create COCO format dataset for a single video."""
        logger.info(f"Creating COCO annotations for video: {video_id}")

        # Get video dimensions
        height, width = self._get_video_dimensions(video_id, video_info)

        coco_data = {
            "info": self._create_info_section(),
            "licenses": [],  # Not used
            "images": [],
            "annotations": [],
            "categories": self._create_categories_section(),
            "videos": [self._create_video_section(video_id, video_info, width, height)]
        }

        # Process annotations
        for frame_data in video_results.get("annotations", []):
            frame_id = frame_data["frame_id"]
            # Add image info
            image_info = {
                "id": self.image_id,
                "width": width,
                "height": height,
                "file_name": frame_data["file_name"],
                "frame_id": frame_id
            }
            coco_data["images"].append(image_info)

            # Process detections
            for detection in frame_data.get("detections", []):
                if self._validate_detection(detection, width, height):
                    annotation = self._create_annotation(
                        detection, self.image_id, frame_id
                    )
                    coco_data["annotations"].append(annotation)

            self.image_id += 1

        logger.info(f"Created COCO dataset with {len(coco_data['images'])} images, "
                    f"{len(coco_data['annotations'])} annotations")

        return coco_data

    def _get_video_dimensions(self, video_id: str, video_info: Dict) -> Tuple[int, int]:
        """Get video dimensions with fallback strategies."""

        # Try from video_info
        if "width" in video_info and "height" in video_info:
            try:
                width, height = int(video_info["width"]), int(video_info["height"])
                if width > 0 and height > 0:
                    return height, width
            except (ValueError, TypeError):
                pass

        # Try to read from first frame
        try:
            video_path = Path(video_info["path"])
            frames_dir = CONFIG.paths.data_dir / video_path.stem
            frame_files = list(frames_dir.glob("frame_*.jpg"))

            if frame_files:
                img = cv2.imread(str(frame_files[0]))
                if img is not None:
                    return img.shape[:2]
        except Exception as e:
            logger.debug(f"Could not read frame for dimensions: {e}")

        # Use inference resolution as fallback
        width, height = CONFIG.inference_resolution
        logger.warning(f"Using inference resolution {width}x{height} for video {video_id}")
        return height, width

    def _create_info_section(self) -> Dict[str, Any]:
        """Create COCO info section."""
        return {
            "description": "YtDataGen Video Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "YtDataGen",
            "date_created": datetime.now().isoformat()
        }

    def _create_categories_section(self) -> List[Dict[str, Any]]:
        """Create COCO categories section."""
        categories = []

        for class_id, class_name in CONFIG.custom_classes.items():
            categories.append({
                "id": class_id,
                "name": class_name,
                "supercategory": None
            })

        return categories

    def _create_video_section(self, video_id: str, video_info: Dict, width: int, height: int) -> Dict[str, Any]:
        """Create video information section."""
        return {
            "id": self.video_id_counter,
            "name": video_id,
            "fps": float(video_info.get("fps", 30.0)),
            "frames": int(video_info.get("frames", 0)),
            "width": width,
            "height": height,
            "duration": float(video_info.get("duration", 0.0))
        }

    def _validate_detection(self, detection: Dict, img_width: int, img_height: int) -> bool:
        """Validate detection data."""
        bbox = detection.get("bbox", [])
        if len(bbox) != 4:
            return False

        x, y, w, h = bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return False

        if x + w > img_width or y + h > img_height:
            return False

        return True

    def _create_annotation(self, detection: Dict, image_id: int, frame_id: int) -> Dict[str, Any]:
        """Create COCO annotation."""
        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": detection["class_id"],
            "bbox": detection["bbox"],
            "area": detection["area"],
            "iscrowd": 0,
            "track_id": detection["track_id"],
            "score": detection["confidence"],
        }

        # Add segmentation if available
        if detection.get("segmentation"):
            annotation["segmentation"] = detection["segmentation"]
        else:
            annotation["segmentation"] = []

        self.annotation_id += 1
        return annotation

    def save_video_annotations(self, video_results: Dict, video_id: str, video_info: Dict) -> Path:
        """Save annotations for a single video."""
        coco_data = self.create_coco_dataset(video_results, video_id, video_info)

        output_file = CONFIG.paths.annotations_dir / f"{video_id}_annotations.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Saved annotations to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to save annotations for {video_id}: {e}")
            raise

        # Validate with pycocotools
        self._validate_coco_format(output_file)

        self.video_id_counter += 1
        return output_file


    def create_final_dataset(self, all_video_annotations: List[Path], video_info_dict: Dict) -> Dict[str, Any]:
        """
        Створює фінальний об'єднаний COCO датасет з індивідуальних файлів,
        інтелектуально перепризначаючи ID, щоб уникнути конфліктів.
        """
        logger.info("Creating final combined COCO dataset")

        final_dataset = {
            "info": self._create_info_section(),
            "licenses": [],
            "categories": self._create_categories_section(),
            "videos": [],
            "images": [],
            "annotations": []
        }

        # Глобальні лічильники для нових, унікальних ID
        new_image_id_counter = 1
        new_annotation_id_counter = 1
        new_video_id_counter = 1

        # Сортуємо файли для стабільного порядку
        for annotation_file in sorted(all_video_annotations):
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    video_data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read or parse {annotation_file}: {e}")
                continue

            # Мапа для відстеження старих ID зображень -> нових ID у межах цього файлу
            image_id_map = {}

            # 1. Обробка зображень (images)
            for image in video_data.get("images", []):
                old_image_id = image["id"]

                # Створюємо запис у мапі: старий ID тепер вказує на новий унікальний ID
                image_id_map[old_image_id] = new_image_id_counter

                # Призначаємо новий унікальний ID для зображення
                image["id"] = new_image_id_counter
                final_dataset["images"].append(image)
                new_image_id_counter += 1

            # 2. Обробка анотацій (annotations)
            for annotation in video_data.get("annotations", []):
                old_image_id = annotation["image_id"]

                # Знаходимо новий ID зображення, до якого прив'язана ця анотація
                if old_image_id in image_id_map:
                    annotation["image_id"] = image_id_map[old_image_id]

                    # Призначаємо новий унікальний ID для самої анотації
                    annotation["id"] = new_annotation_id_counter
                    final_dataset["annotations"].append(annotation)
                    new_annotation_id_counter += 1
                else:
                    logger.warning(
                        f"Skipping orphan annotation from {annotation_file} with missing image_id: {old_image_id}")

            # 3. Обробка відео (videos)
            for video in video_data.get("videos", []):
                video["id"] = new_video_id_counter
                final_dataset["videos"].append(video)
                new_video_id_counter += 1

        logger.info(f"Final dataset contains {len(final_dataset['videos'])} videos, "
                    f"{len(final_dataset['images'])} images, "
                    f"{len(final_dataset['annotations'])} annotations")

        return final_dataset


    def save_final_annotations(self, all_video_annotations: List[Path], video_info_dict: Dict) -> Path:
        """Save the final combined annotations file."""
        final_dataset = self.create_final_dataset(all_video_annotations, video_info_dict)

        try:
            # ✅ ВИПРАВЛЕННЯ: забезпечуємо що директорія існує
            CONFIG.paths.labels_final_path.parent.mkdir(parents=True, exist_ok=True)

            with open(CONFIG.paths.labels_final_path, 'w', encoding='utf-8') as f:
                json.dump(final_dataset, f, indent=2, ensure_ascii=False, default=str)

            # Validate with pycocotools
            self._validate_coco_format(CONFIG.paths.labels_final_path)

            logger.info(f"Saved final annotations to: {CONFIG.paths.labels_final_path}")

        except Exception as e:
            logger.error(f"Failed to save final annotations: {e}")
            raise

        return CONFIG.paths.labels_final_path

    def _validate_coco_format(self, annotation_file: Path):
        """Validate COCO format using pycocotools."""
        try:
            coco = COCO(str(annotation_file))
            logger.info(f"COCO validation passed for {annotation_file}")
            return True
        except Exception as e:
            logger.warning(f"COCO validation failed: {e}")
            return False