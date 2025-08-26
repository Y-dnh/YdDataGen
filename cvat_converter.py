#!/usr/bin/env python3
"""
CVAT Converter - YtDataGen

Standalone converter from COCO annotations to CVAT format.
Can be run independently of the main pipeline.

Usage:
    python cvat_converter.py --all
    python cvat_converter.py --file path/to/annotations.json
    python cvat_converter.py --all --keyframe-mode fps --fps-multiplier 2.0
"""

import argparse
import json
import logging
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

from src.config import CONFIG
from src.utils import setup_logging, load_json, get_video_info

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Default output directory (can be overridden)
CVAT_OUTPUT_DIR = CONFIG.paths.cvat_annotations_dir


class COCOToCVATConverter:
    """Converts COCO annotations to CVAT format with intelligent keyframing."""

    def __init__(self):
        self.categories = {}
        self.images = {}
        self.annotations = []
        self.image_id_to_frame = {}  # Mapping from image_id to frame number
        self.track_id_counter = 0

    def reset_counters(self):
        """Reset counters for new conversion session."""
        self.track_id_counter = 0
        self.categories = {}
        self.images = {}
        self.annotations = []
        self.image_id_to_frame = {}

    def load_coco_json(self, json_path):
        """Load COCO JSON file"""
        logger.info(f"Loading COCO data from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Store categories
        for category in coco_data.get('categories', []):
            if isinstance(category, dict) and 'id' in category and 'name' in category:
                self.categories[category['id']] = category['name']

        # Store image information
        for image in coco_data.get('images', []):
            if isinstance(image, dict) and 'id' in image:
                self.images[image['id']] = {
                    'file_name': image.get('file_name', f"frame_{image['id']}.jpg"),
                    'width': image.get('width', CONFIG.yolo_imgsz),
                    'height': image.get('height', CONFIG.yolo_imgsz)
                }

        # Create mapping from image_id to sequential frame numbers (0, 1, 2, ...)
        sorted_image_ids = sorted(self.images.keys())
        self.image_id_to_frame = {img_id: idx for idx, img_id in enumerate(sorted_image_ids)}

        # Store annotations
        annotations = coco_data.get('annotations', [])
        self.annotations = [ann for ann in annotations if isinstance(ann, dict)]

        logger.info(
            f"Loaded {len(self.categories)} categories, {len(self.images)} images, {len(self.annotations)} annotations")
        return coco_data

    def auto_detect_fps(self, video_id: str, coco_data: Dict) -> float:
        """Auto-detect FPS from multiple sources."""
        fps = 30.0  # Default fallback

        # 1. Try COCO videos section
        videos = coco_data.get("videos", [])
        if isinstance(videos, list) and len(videos) > 0 and isinstance(videos[0], dict):
            video_info = videos[0]
            if "fps" in video_info:
                try:
                    fps = float(video_info["fps"])
                    logger.debug(f"Got FPS from COCO data: {fps}")
                    return fps
                except (ValueError, TypeError) as e:
                    logger.debug(f"Invalid FPS value in COCO data: {e}")

        # 2. Try actual video file
        try:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            for ext in video_extensions:
                video_path = CONFIG.paths.videos_dir / f"{video_id}{ext}"
                if video_path.exists():
                    video_info = get_video_info(video_path)
                    if video_info and video_info.get("fps", 0) > 0:
                        fps = float(video_info["fps"])
                        logger.debug(f"Got FPS from video file: {fps}")
                        return fps
        except Exception as e:
            logger.debug(f"Could not get FPS from video file: {e}")

        logger.debug(f"Using default FPS: {fps}")
        return fps

    def determine_keyframes(self, total_frames: int, fps: float, video_id: str,
                            keyframe_mode: str, keyframe_interval: int, fps_multiplier: float) -> Set[int]:
        """Determine keyframes based on configuration and video properties."""
        keyframes = set()

        # Validate inputs
        if total_frames <= 0:
            logger.warning(f"Video {video_id}: Invalid total_frames: {total_frames}")
            return {0}  # Return at least one keyframe

        if keyframe_mode == "fps":
            if fps > 0:
                # Keyframe every N seconds
                interval = max(1, int(fps * fps_multiplier))
                keyframes = set(range(0, total_frames, interval))
                logger.info(f"Video {video_id}: FPS-based keyframes every {interval} frames "
                            f"({fps_multiplier}s @ {fps:.1f}fps)")
            else:
                # Fallback to interval mode
                interval = max(1, keyframe_interval)
                keyframes = set(range(0, total_frames, interval))
                logger.warning(f"Video {video_id}: No FPS data, using interval {interval}")

        elif keyframe_mode == "interval":
            interval = max(1, keyframe_interval)
            keyframes = set(range(0, total_frames, interval))
            logger.info(f"Video {video_id}: Interval-based keyframes every {interval} frames")

        elif keyframe_mode == "auto":
            # Intelligent auto mode based on video length and FPS
            if total_frames <= 30:  # Very short video
                keyframes = set(range(total_frames))  # All frames
            elif total_frames <= 300:  # Short video (< 10s @ 30fps)
                interval = max(1, total_frames // 10)
                keyframes = set(range(0, total_frames, interval))
            else:  # Longer video
                # Aim for keyframes every 1-2 seconds
                optimal_interval = max(1, int(fps * 1.5))
                keyframes = set(range(0, total_frames, optimal_interval))
            logger.info(f"Video {video_id}: Auto keyframes, {len(keyframes)} keyframes for {total_frames} frames")

        # Always include first and last frames
        keyframes.add(0)
        if total_frames > 1:
            keyframes.add(total_frames - 1)

        logger.info(f"Video {video_id}: Generated {len(keyframes)} keyframes from {total_frames} total frames")
        return keyframes

    def segmentation_to_polygon(self, segmentation):
        """Convert segmentation to CVAT polygon format"""
        if isinstance(segmentation, list) and len(segmentation) > 0:
            # Take first polygon if there are multiple
            polygon = segmentation[0]
            points = []
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    points.append(f"{polygon[i]},{polygon[i + 1]}")
            return ";".join(points)
        return ""

    def create_cvat_xml(self, task_name="converted_task"):
        """Create CVAT XML structure - using working version from second file"""
        # Create root element
        root = ET.Element("annotations")

        # Add version
        version = ET.SubElement(root, "version")
        version.text = "1.1"

        # Add meta information
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")

        # Task ID
        task_id = ET.SubElement(task, "id")
        task_id.text = "1"

        # Task name
        name = ET.SubElement(task, "name")
        name.text = task_name

        # Task size (number of frames)
        size = ET.SubElement(task, "size")
        size.text = str(len(self.images)) if self.images else "1"

        # Mode
        mode = ET.SubElement(task, "mode")
        mode.text = "annotation"

        # Overlap
        overlap = ET.SubElement(task, "overlap")
        overlap.text = "0"

        # Bugtracker
        bugtracker = ET.SubElement(task, "bugtracker")

        # Flipped
        flipped = ET.SubElement(task, "flipped")
        flipped.text = "False"

        # Creation and update dates
        created = ET.SubElement(task, "created")
        created.text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%z")

        updated = ET.SubElement(task, "updated")
        updated.text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%z")

        # Labels
        labels = ET.SubElement(task, "labels")

        # Create labels for categories
        for cat_id, cat_name in self.categories.items():
            label = ET.SubElement(labels, "label")
            label_name = ET.SubElement(label, "name")
            label_name.text = cat_name
            attributes = ET.SubElement(label, "attributes")

        # Segments
        segments = ET.SubElement(task, "segments")
        segment = ET.SubElement(segments, "segment")
        seg_id = ET.SubElement(segment, "id")
        seg_id.text = "1"
        start = ET.SubElement(segment, "start")
        start.text = "0"
        stop = ET.SubElement(segment, "stop")
        stop.text = str(len(self.images) - 1) if self.images else "0"
        url = ET.SubElement(segment, "url")
        url.text = "http://localhost:8080/?id=1"

        # Owner
        owner = ET.SubElement(task, "owner")
        username = ET.SubElement(owner, "username")
        username.text = "admin"
        email = ET.SubElement(owner, "email")
        email.text = ""

        # Original size (use first image)
        if self.images:
            first_image = next(iter(self.images.values()))
            original_size = ET.SubElement(task, "original_size")
            width = ET.SubElement(original_size, "width")
            width.text = str(first_image['width'])
            height = ET.SubElement(original_size, "height")
            height.text = str(first_image['height'])

        # Dump date
        dumped = ET.SubElement(meta, "dumped")
        dumped.text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%z")

        return root

    def group_annotations_by_track(self):
        """Group annotations into tracks - using working version from second file"""
        tracks = {}

        # Sort annotations by image_id (frame)
        sorted_annotations = sorted(self.annotations, key=lambda x: self.image_id_to_frame.get(x.get('image_id', 0), 0))

        for ann in sorted_annotations:
            if not isinstance(ann, dict) or 'image_id' not in ann or 'category_id' not in ann:
                continue

            # Use track_id if available, otherwise group by category
            if 'track_id' in ann and ann['track_id'] is not None:
                track_key = f"{ann['category_id']}_{ann['track_id']}"
            else:
                track_key = f"{ann['category_id']}_0"

            if track_key not in tracks:
                tracks[track_key] = {
                    'category_id': ann['category_id'],
                    'annotations': []
                }

            tracks[track_key]['annotations'].append(ann)

        return tracks

    def add_interpolated_frames(self, track_annotations, keyframes: Set[int]):
        """Add interpolated frames (outside=1) between keyframes"""
        if len(track_annotations) <= 1:
            return track_annotations

        # Sort by frame number
        sorted_anns = sorted(track_annotations, key=lambda x: self.image_id_to_frame.get(x.get('image_id', 0), 0))
        result = []

        for i, ann in enumerate(sorted_anns):
            current_frame = self.image_id_to_frame.get(ann.get('image_id', 0), 0)

            # Mark as keyframe if in keyframe set
            ann['keyframe'] = current_frame in keyframes
            result.append(ann)

            # Add "outside" frame after each annotation (except last)
            if i < len(sorted_anns) - 1:
                next_frame = self.image_id_to_frame.get(sorted_anns[i + 1].get('image_id', 0), 0)

                # If there's a gap between frames, add outside frame
                if next_frame > current_frame + 1:
                    outside_ann = ann.copy()
                    outside_ann['frame_num'] = current_frame + 1
                    outside_ann['outside'] = True
                    outside_ann['keyframe'] = False
                    result.append(outside_ann)

        return result

    def add_annotations_to_xml(self, root, keyframes: Set[int]):
        """
        Adds annotations to the XML tree with correct track interruption logic.
        - Sets outside="1" on the last frame of a continuous segment.
        - Sets keyframe="1" at the start and end of each segment.
        """
        tracks = self.group_annotations_by_track()
        track_id_counter = 0

        for track_key, track_data in tracks.items():
            # Sort annotations by frame number to process them in order
            sorted_annotations = sorted(
                track_data['annotations'],
                key=lambda x: self.image_id_to_frame.get(x.get('image_id', 0), 0)
            )

            if not sorted_annotations:
                continue

            # --- Create the main track element ---
            track = ET.SubElement(root, "track")
            track.set("id", str(track_id_counter))
            category_name = self.categories.get(track_data['category_id'], "unknown_category")
            track.set("label", category_name)
            track.set("source", "manual")

            # --- Iterate through annotations to build segments ---
            for i, ann in enumerate(sorted_annotations):
                frame_num = self.image_id_to_frame.get(ann.get('image_id', 0), 0)

                # --- Determine keyframe and outside status ---
                is_first_in_track = (i == 0)
                is_last_in_track = (i == len(sorted_annotations) - 1)

                # Check for a gap AFTER this frame.
                # If there's a gap, this is the last frame of a segment.
                is_last_in_segment = is_last_in_track
                if not is_last_in_track:
                    next_ann = sorted_annotations[i + 1]
                    next_frame_num = self.image_id_to_frame.get(next_ann.get('image_id', 0), 0)
                    if next_frame_num > frame_num + 1:
                        is_last_in_segment = True

                # Check if this frame is the START of a new segment (after a gap)
                is_first_in_segment = is_first_in_track
                if not is_first_in_track:
                    prev_ann = sorted_annotations[i - 1]
                    prev_frame_num = self.image_id_to_frame.get(prev_ann.get('image_id', 0), 0)
                    if frame_num > prev_frame_num + 1:
                        is_first_in_segment = True

                # outside = 1 if it's the last frame of a segment, otherwise 0
                outside_val = "1" if is_last_in_segment else "0"

                # keyframe = 1 if it's in the keyframe list, OR at the start/end of a segment
                is_keyframe = (
                        frame_num in keyframes or
                        is_first_in_segment or
                        is_last_in_segment
                )
                keyframe_val = "1" if is_keyframe else "0"

                # --- Create the XML element (box or polygon) ---
                element = None
                if 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
                    element = ET.SubElement(track, "box")
                    x, y, w, h = ann['bbox']
                    element.set("xtl", str(x))
                    element.set("ytl", str(y))
                    element.set("xbr", str(x + w))
                    element.set("ybr", str(y + h))

                elif 'segmentation' in ann and ann['segmentation']:
                    element = ET.SubElement(track, "polygon")
                    points = self.segmentation_to_polygon(ann['segmentation'])
                    if points:
                        element.set("points", points)

                if element is not None:
                    element.set("frame", str(frame_num))
                    element.set("outside", outside_val)
                    element.set("occluded", "0")
                    element.set("keyframe", keyframe_val)

            track_id_counter += 1

    def convert_single_video(self, annotation_file: Path, keyframe_mode: str = "fps",
                             keyframe_interval: int = 30, fps_multiplier: float = 1.0) -> Path:
        """Convert a single COCO annotation file to CVAT format."""
        logger.info(f"Converting COCO to CVAT: {annotation_file.name}")

        # Reset state for new conversion
        self.reset_counters()

        # Load and validate COCO data
        coco_data = self.load_coco_json(annotation_file)
        if not coco_data or not isinstance(coco_data, dict):
            raise ValueError(f"Could not load valid COCO data from {annotation_file}")

        # Extract video information
        video_id = annotation_file.stem.replace("_annotations", "")

        if not self.images:
            raise ValueError(f"No valid images found in {annotation_file}")

        # Get video metadata
        total_frames = len(self.images)
        fps = self.auto_detect_fps(video_id, coco_data)

        # Determine keyframes
        keyframes = self.determine_keyframes(
            total_frames, fps, video_id, keyframe_mode, keyframe_interval, fps_multiplier
        )

        # Create CVAT XML
        root = self.create_cvat_xml(f"YtDataGen_{video_id}")

        # Add annotations with keyframe information
        self.add_annotations_to_xml(root, keyframes)

        # Save CVAT file
        output_file = CVAT_OUTPUT_DIR / f"{video_id}_cvat.xml"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.save_xml(root, output_file)
        logger.info(f"Saved CVAT annotations: {output_file}")
        return output_file

    def convert_all_videos(self, keyframe_mode: str = "fps", keyframe_interval: int = 30,
                           fps_multiplier: float = 1.0) -> List[Path]:
        """Convert all COCO annotation files to CVAT format."""
        annotation_files = list(CONFIG.paths.annotations_dir.glob("*_annotations.json"))

        if not annotation_files:
            logger.warning(f"No COCO annotation files found in {CONFIG.paths.annotations_dir}")
            return []

        logger.info(f"Found {len(annotation_files)} COCO annotation files to convert")

        converted_files = []

        for annotation_file in annotation_files:
            try:
                output_file = self.convert_single_video(
                    annotation_file, keyframe_mode, keyframe_interval, fps_multiplier
                )
                converted_files.append(output_file)
            except Exception as e:
                logger.error(f"Failed to convert {annotation_file.name}: {e}")
                continue

        logger.info(f"Successfully converted {len(converted_files)}/{len(annotation_files)} files to CVAT format")
        return converted_files

    def save_xml(self, root, output_path):
        """Save XML file with formatting"""
        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ", encoding='utf-8')

        with open(output_path, 'wb') as f:
            f.write(pretty_xml)


def main():
    """CLI entry point for CVAT conversion."""
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to CVAT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all
  %(prog)s --file video123_annotations.json
  %(prog)s --all --keyframe-mode fps --fps-multiplier 2.0
  %(prog)s --all --keyframe-mode interval --keyframe-interval 5
        """
    )

    parser.add_argument('-f', '--file', type=Path,
                        help="Single COCO annotation file to convert")
    parser.add_argument('--all', action='store_true',
                        help="Convert all COCO annotation files")

    parser.add_argument('--keyframe-mode', choices=['fps', 'interval', 'auto'],
                        default='fps', help="Keyframe selection mode (default: fps)")
    parser.add_argument('--keyframe-interval', type=int, default=30,
                        help="Keyframe interval in frames (for interval mode, default: 30)")
    parser.add_argument('--fps-multiplier', type=float, default=1.0,
                        help="Keyframe every N seconds (for fps mode, default: 1.0)")

    parser.add_argument('--output-dir', type=Path,
                        help="Output directory for CVAT files")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Verbose logging")
    parser.add_argument('--debug', action='store_true',
                        help="Debug logging")

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        CONFIG.log_level = 'DEBUG'
    elif args.verbose:
        CONFIG.log_level = 'INFO'
    setup_logging()

    # Override output directory if specified
    if args.output_dir:
        global CVAT_OUTPUT_DIR
        CVAT_OUTPUT_DIR = args.output_dir
        CVAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using custom output directory: {CVAT_OUTPUT_DIR}")

    # Validate arguments
    if not args.all and not args.file:
        parser.error("Specify either --file or --all")

    if args.file and not args.file.exists():
        parser.error(f"File not found: {args.file}")

    # Ensure directories exist
    CONFIG.paths.ensure_dirs()
    CVAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize converter
    converter = COCOToCVATConverter()

    logger.info("=" * 60)
    logger.info("CVAT CONVERTER - YtDataGen")
    logger.info(f"Keyframe mode: {args.keyframe_mode}")
    if args.keyframe_mode == 'fps':
        logger.info(f"FPS multiplier: {args.fps_multiplier}s")
    elif args.keyframe_mode == 'interval':
        logger.info(f"Frame interval: {args.keyframe_interval}")
    logger.info("=" * 60)

    try:
        if args.all:
            converted_files = converter.convert_all_videos(
                args.keyframe_mode, args.keyframe_interval, args.fps_multiplier
            )
            logger.info("=" * 60)
            logger.info("CONVERSION COMPLETE")
            logger.info(f"Successfully converted: {len(converted_files)} files")
            logger.info(f"Output directory: {CVAT_OUTPUT_DIR}")
            logger.info("=" * 60)
        else:
            output_file = converter.convert_single_video(
                args.file, args.keyframe_mode, args.keyframe_interval, args.fps_multiplier
            )
            logger.info("=" * 60)
            logger.info("CONVERSION COMPLETE")
            logger.info(f"Output file: {output_file}")
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=args.debug)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())