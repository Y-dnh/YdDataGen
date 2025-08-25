#!/usr/bin/env python3
"""
YtDataGen Visualization Script - Enhanced Version

Створює анотовані відео, використовуючи надійний мапінг
між кадрами відео та ID зображень в анотаціях.
Enhanced with improved visual styling and text rendering.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from collections import defaultdict

from src.config import CONFIG
from src.utils import setup_logging, load_json

setup_logging()
logger = logging.getLogger(__name__)

# --- Enhanced visual configuration ---
DEFAULT_COLORS = [
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 255),  # Light Blue
    (128, 128, 128)  # Gray
]

# Enhanced text configuration
TEXT_CONFIG = {
    "font": cv2.FONT_HERSHEY_DUPLEX,
    "scale": 0.6,
    "thickness": 2,
    "color": (255, 255, 255),
    "bg_color": (0, 0, 0),
    "padding": 8,
    "line_height": 25,
    "margin": 5
}


class VideoAnnotator:
    """Creates annotated videos with enhanced visual styling and text rendering."""

    def __init__(self):
        self.colors = {}

    def _get_color(self, class_id: int) -> tuple:
        """Assigns consistent colors to class IDs using cycling through predefined palette."""
        if class_id not in self.colors:
            self.colors[class_id] = DEFAULT_COLORS[len(self.colors) % len(DEFAULT_COLORS)]
        return self.colors[class_id]

    def _get_text_size(self, text: str) -> Tuple[int, int]:
        """Get text size for consistent formatting."""
        return cv2.getTextSize(
            text,
            TEXT_CONFIG['font'],
            TEXT_CONFIG['scale'],
            TEXT_CONFIG['thickness']
        )[0]

    def _draw_text_with_background(self, frame: np.ndarray, text: str,
                                   position: Tuple[int, int], color: Tuple[int, int, int],
                                   background_alpha: float = 0.8) -> np.ndarray:
        """
        Draws text with semi-transparent background for better readability.
        Uses alpha blending to create overlay effect without completely obscuring the video.
        """
        x, y = position
        text_width, text_height = self._get_text_size(text)

        # Calculate background rectangle bounds with padding
        bg_x1 = x - TEXT_CONFIG['padding'] // 2
        bg_y1 = y - text_height - TEXT_CONFIG['padding']
        bg_x2 = x + text_width + TEXT_CONFIG['padding'] // 2
        bg_y2 = y + TEXT_CONFIG['padding'] // 2

        # Clamp to frame boundaries to prevent drawing outside
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(frame.shape[1], bg_x2)
        bg_y2 = min(frame.shape[0], bg_y2)

        # Create semi-transparent background using overlay technique
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(overlay, background_alpha, frame, 1 - background_alpha, 0, frame)

        # Add border for better definition
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                      tuple(max(0, c - 50) for c in color), 1)

        cv2.putText(frame, text, (x, y - TEXT_CONFIG['padding'] // 4),
                    TEXT_CONFIG['font'], TEXT_CONFIG['scale'],
                    TEXT_CONFIG['color'], TEXT_CONFIG['thickness'])

        return frame

    def _draw_multi_line_text(self, frame: np.ndarray, texts: List[str],
                              box_position: Tuple[int, int], color: Tuple[int, int, int]):
        """
        Draws multiple text lines near bounding box with intelligent positioning.
        Automatically positions text above or below bbox based on available space.
        """
        x, y = box_position
        total_height = 0
        text_infos = []

        # Pre-calculate all text dimensions
        for text in texts:
            text_width, text_height = self._get_text_size(text)
            text_infos.append((text, text_width, text_height))
            total_height += text_height + TEXT_CONFIG['padding']

        start_y = y - TEXT_CONFIG['margin']
        current_y = start_y

        # If text would go above frame bounds, position below the bbox instead
        if start_y - total_height < 0:
            current_y = y + TEXT_CONFIG['line_height']

        for i, (text, text_width, text_height) in enumerate(text_infos):
            # Ensure text stays within frame horizontally
            text_x = max(TEXT_CONFIG['padding'], min(x, frame.shape[1] - text_width - TEXT_CONFIG['padding']))
            text_y = max(text_height + TEXT_CONFIG['padding'],
                         min(current_y, frame.shape[0] - TEXT_CONFIG['padding']))

            self._draw_text_with_background(frame, text, (text_x, text_y), color)

            # Update position for next line based on positioning strategy
            if start_y - total_height < 0:
                current_y += TEXT_CONFIG['line_height']  # Going down
            else:
                current_y -= (text_height + TEXT_CONFIG['padding'])  # Going up

    def _draw_bbox_only(self, frame: np.ndarray, bbox: List[int], color: Tuple[int, int, int]):
        """Draws bounding box with enhanced dual-border style for better visibility."""
        x, y, w, h = [int(c) for c in bbox]
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Main border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        # Inner highlight border for depth effect
        inner_color = tuple(min(255, c + 30) for c in color)
        cv2.rectangle(frame, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), inner_color, 1)

    def _prepare_annotation_texts(self, ann: Dict[str, Any], categories: Dict[int, str], options: Dict[str, bool]) -> \
    List[str]:
        """
        Prepares list of text strings to display based on annotation data and user options.
        Handles optional confidence scores and track IDs.
        """
        texts_to_draw = []
        label = categories.get(ann['category_id'], "Unknown")
        confidence = ann.get("score")
        track_id = ann.get("track_id")

        if options.get("show_labels", True):
            if options.get("show_confidence", True) and confidence is not None:
                texts_to_draw.append(f"{label} ({confidence:.2f})")
            else:
                texts_to_draw.append(label)

        if options.get("show_tracks", True) and track_id is not None:
            texts_to_draw.append(f"ID: {track_id}")

        return texts_to_draw

    def _draw_polygon_mask(self, frame, polygons, color, alpha=0.25):
        """
        Renders segmentation masks with semi-transparent fill and outlined borders.
        Uses overlay blending for professional appearance without obscuring details.
        """
        if not polygons: return

        # Create filled overlay for transparency effect
        overlay = frame.copy()
        for polygon_coords in polygons:
            if not isinstance(polygon_coords, list) or not polygon_coords or len(polygon_coords) < 6:
                continue
            points = np.array(polygon_coords).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(overlay, [points], color)

        # Blend overlay with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Add crisp borders for mask definition
        for polygon_coords in polygons:
            if not isinstance(polygon_coords, list) or not polygon_coords or len(polygon_coords) < 6:
                continue
            points = np.array(polygon_coords).reshape(-1, 2).astype(np.int32)
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
            # Inner highlight for depth
            inner_color = tuple(min(255, c + 40) for c in color)
            cv2.polylines(frame, [points], isClosed=True, color=inner_color, thickness=1)

    def process_single_video(self, annotation_file: Path, **vis_options):
        """
        Processes a single video file, creating annotated version with flexible visualization options.

        Args:
            annotation_file: Path to COCO annotation JSON file
            **vis_options: Dictionary of visualization toggles (show_boxes, show_masks, etc.)
        """
        logger.info(f"Processing annotation file: {annotation_file}")

        try:
            data = load_json(annotation_file)
            if not data or "videos" not in data or not data["videos"]:
                logger.error(f"Invalid or empty annotation file: {annotation_file}")
                return
        except Exception as e:
            logger.error(f"Failed to load or parse JSON {annotation_file}: {e}")
            return

        video_info = data["videos"][0]
        video_id = video_info.get("file_name", annotation_file.stem.replace("_annotations", ""))
        video_path = next(CONFIG.paths.videos_dir.glob(f"{video_id}.*"), None)

        if not video_path.exists():
            logger.error(f"Video file not found for annotations: {video_path}")
            return

        # Build lookup structures for efficient frame-to-annotation mapping
        images_index = {img['id']: img for img in data.get('images', [])}
        annotations_by_img_id = defaultdict(list)
        for ann in data.get('annotations', []):
            annotations_by_img_id[ann['image_id']].append(ann)

        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        self.colors = {cat_id: self._get_color(cat_id) for cat_id in categories}

        # Create frame index to image ID mapping for video synchronization
        frame_to_image_id = {img.get('frame_idx', img['id']): img['id'] for img in images_index.values()}
        logger.info(f"Created mapping for {len(frame_to_image_id)} frames.")

        # Initialize video processing
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_dir = CONFIG.paths.root / "visualized_videos"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{video_id}_visualized.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Visualizing {video_id}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Look up annotations for current frame using mapping
                image_id = frame_to_image_id.get(frame_idx)
                if image_id and image_id in annotations_by_img_id:
                    for ann in annotations_by_img_id[image_id]:
                        cat_id = ann['category_id']
                        if cat_id not in categories: continue

                        color = self.colors.get(cat_id)

                        # --- Modular visualization logic with separate toggles ---

                        # 1. Render masks first (background layer)
                        if vis_options.get("show_masks", True) and "segmentation" in ann:
                            self._draw_polygon_mask(frame, ann["segmentation"], color)

                        # 2. Render bounding boxes (foreground layer)
                        if vis_options.get("show_boxes", True) and "bbox" in ann:
                            self._draw_bbox_only(frame, ann["bbox"], color)

                        # 3. Render text labels (top layer)
                        # Text positioning uses bbox even if boxes are hidden
                        if "bbox" in ann:
                            texts_to_draw = self._prepare_annotation_texts(ann, categories, vis_options)
                            if texts_to_draw:
                                x, y, _, _ = [int(c) for c in ann["bbox"]]
                                self._draw_multi_line_text(frame, texts_to_draw, (x, y), color)

                out.write(frame)
                pbar.update(1)
                frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"Successfully created visualized video: {output_path}")

    def process_all_videos(self, **vis_options):
        """Processes all annotation files in the configured directory."""
        annotation_files = list(CONFIG.paths.annotations_dir.glob("*_annotations.json"))
        if not annotation_files:
            logger.warning(f"No annotation files found in {CONFIG.paths.annotations_dir}")
            return

        logger.info(f"Found {len(annotation_files)} annotation files to process.")
        for ann_file in annotation_files:
            self.process_single_video(ann_file, **vis_options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Візуалізація анотацій COCO на відео з покращеним стилем.")
    parser.add_argument("-f", "--annotation_file", type=Path, help="Шлях до файлу анотацій COCO (.json).")
    parser.add_argument("--all", action="store_true", help="Запустити візуалізацію для всіх файлів анотацій.")
    parser.add_argument("--no-boxes", action="store_true", help="Не показувати рамки.")
    parser.add_argument("--no-masks", action="store_true", help="Не показувати маски.")
    parser.add_argument("--no-tracks", action="store_true", help="Не показувати ID треків.")
    parser.add_argument("--no-labels", action="store_true", help="Не показувати мітки.")
    parser.add_argument("--no-confidence", action="store_true", help="Не показувати впевненість.")
    args = parser.parse_args()

    if not args.all and args.annotation_file is None:
        parser.error("Вкажіть 'annotation_file' або використайте --all.")

    visualizer = VideoAnnotator()
    vis_options = {
        "show_boxes": not args.no_boxes, "show_masks": not args.no_masks,
        "show_tracks": not args.no_tracks, "show_labels": not args.no_labels,
        "show_confidence": not args.no_confidence
    }

    if args.all:
        visualizer.process_all_videos(**vis_options)
    else:
        visualizer.process_single_video(args.annotation_file, **vis_options)