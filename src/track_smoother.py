import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

from src.config import CONFIG

logger = logging.getLogger(__name__)


class TrackSmoother:
    """Handles track smoothing, classification fixing, and false positive reduction."""

    def __init__(self):
        self.min_track_length = CONFIG.min_track_length
        self.max_gap_frames = CONFIG.max_gap_frames
        self.min_confidence_for_gap_fill = CONFIG.min_confidence_for_gap_fill
        self.class_smoothing_window = CONFIG.class_smoothing_window
        self.class_confidence_threshold = CONFIG.class_confidence_threshold
        self.interpolate_missing = CONFIG.interpolate_missing_detections

    def smooth_video_tracks(self, annotations: List[Dict]) -> List[Dict]:
        """Main entry point for track smoothing."""
        if not CONFIG.track_smoothing_enabled:
            return annotations

        logger.info("Applying track smoothing...")

        # Build track database from all detections
        track_db = self._build_track_database(annotations)

        # Filter out short tracks (likely false positives)
        track_db = self._filter_short_tracks(track_db)

        # Smooth classifications for each track
        track_db = self._smooth_classifications(track_db)

        # Fill gaps in tracks
        if self.interpolate_missing:
            track_db = self._fill_track_gaps(track_db)

        # Rebuild annotations from smoothed tracks
        smoothed_annotations = self._rebuild_annotations(annotations, track_db)

        logger.info(f"Track smoothing complete. Original tracks: {len(track_db)}")
        return smoothed_annotations

    def _build_track_database(self, annotations: List[Dict]) -> Dict[int, Dict]:
        """Build a complete database of all tracks across all frames."""
        track_db = defaultdict(lambda: {
            'detections': {},  # frame_id -> detection
            'class_history': [],  # (frame_id, class_id, confidence)
            'bbox_history': [],  # (frame_id, bbox)
            'frames': set()
        })

        for frame_data in annotations:
            frame_id = frame_data["frame_id"]

            for detection in frame_data.get("detections", []):
                track_id = detection["track_id"]

                # Store detection by frame
                track_db[track_id]['detections'][frame_id] = detection.copy()
                track_db[track_id]['frames'].add(frame_id)

                # Store class and confidence history
                track_db[track_id]['class_history'].append((
                    frame_id,
                    detection["class_id"],
                    detection["confidence"]
                ))

                # Store bbox history for interpolation
                track_db[track_id]['bbox_history'].append((
                    frame_id,
                    detection["bbox"]
                ))

        # Sort histories by frame for easier processing
        for track_id in track_db:
            track_db[track_id]['class_history'].sort(key=lambda x: x[0])
            track_db[track_id]['bbox_history'].sort(key=lambda x: x[0])

        return dict(track_db)

    def _filter_short_tracks(self, track_db: Dict[int, Dict]) -> Dict[int, Dict]:
        """Remove tracks that are too short (likely false positives)."""
        filtered_tracks = {}
        removed_count = 0

        for track_id, track_data in track_db.items():
            track_length = len(track_data['frames'])

            if track_length >= self.min_track_length:
                filtered_tracks[track_id] = track_data
            else:
                removed_count += 1
                logger.debug(f"Removed short track {track_id} with {track_length} frames")

        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} short tracks (< {self.min_track_length} frames)")

        return filtered_tracks

    def _smooth_classifications(self, track_db: Dict[int, Dict]) -> Dict[int, Dict]:
        """Fix classification blinking by finding most stable class per track."""
        for track_id, track_data in track_db.items():
            class_history = track_data['class_history']
            if len(class_history) < 2:
                continue

            # Find dominant class with high confidence
            dominant_class = self._find_dominant_class(class_history)

            if dominant_class is not None:
                # Update all detections to use dominant class
                for frame_id, detection in track_data['detections'].items():
                    old_class = detection["class_id"]
                    detection["class_id"] = dominant_class
                    detection["class_name"] = CONFIG.custom_classes.get(dominant_class, f'class_{dominant_class}')

                    if old_class != dominant_class:
                        logger.debug(f"Track {track_id} frame {frame_id}: class {old_class} -> {dominant_class}")

        return track_db

    # Updated function to replace the one in track_smoother.py
    def _find_dominant_class(self, class_history: List[Tuple[int, int, float]]) -> Optional[int]:
        """Find most reliable class using simple weighted voting + frequency filtering."""
        if not class_history:
            return None

        # Weighted voting: each detection votes with weight = confidence
        class_weights = defaultdict(float)
        class_counts = defaultdict(int)

        for frame_id, class_id, confidence in class_history:
            class_weights[class_id] += confidence
            class_counts[class_id] += 1

        # Filter out rare classes (likely false positives)
        total_detections = len(class_history)
        min_detections = max(2, int(total_detections * self.class_confidence_threshold))  # At least 20% or 2 detections

        valid_classes = {
            class_id: weight
            for class_id, weight in class_weights.items()
            if class_counts[class_id] >= min_detections
        }

        # Fallback to most frequent if all filtered out
        if not valid_classes:
            return max(class_counts.keys(), key=class_counts.get)

        # Return class with the highest weighted vote
        return max(valid_classes.keys(), key=valid_classes.get)

    def _fill_track_gaps(self, track_db: Dict[int, Dict]) -> Dict[int, Dict]:
        """Fill gaps in tracks by interpolating missing frames."""
        for track_id, track_data in track_db.items():
            frames = sorted(track_data['frames'])
            bbox_history = track_data['bbox_history']

            if len(frames) < 2:
                continue

            # Find gaps that can be filled
            filled_frames = []

            for i in range(len(frames) - 1):
                current_frame = frames[i]
                next_frame = frames[i + 1]
                gap_size = next_frame - current_frame - 1

                # Only fill small gaps with confident detections
                if 0 < gap_size <= self.max_gap_frames:
                    current_det = track_data['detections'][current_frame]
                    next_det = track_data['detections'][next_frame]

                    # Check if both detections are confident enough
                    if (current_det['confidence'] >= self.min_confidence_for_gap_fill and
                            next_det['confidence'] >= self.min_confidence_for_gap_fill):

                        # Interpolate missing frames
                        for gap_frame in range(current_frame + 1, next_frame):
                            interpolated_det = self._interpolate_detection(
                                current_det, next_det, current_frame, next_frame, gap_frame
                            )
                            track_data['detections'][gap_frame] = interpolated_det
                            filled_frames.append(gap_frame)

            if filled_frames:
                track_data['frames'].update(filled_frames)
                logger.debug(f"Track {track_id}: filled {len(filled_frames)} gap frames")

        return track_db

    def _interpolate_detection(
            self,
            det1: Dict,
            det2: Dict,
            frame1: int,
            frame2: int,
            target_frame: int
    ) -> Dict:
        """Interpolate detection between two frames."""
        # Calculate interpolation weight
        weight = (target_frame - frame1) / (frame2 - frame1)

        # Interpolate bounding box
        bbox1 = det1['bbox']
        bbox2 = det2['bbox']

        interpolated_bbox = [
            bbox1[i] + weight * (bbox2[i] - bbox1[i]) for i in range(4)
        ]

        # Use average confidence, slightly reduced for interpolated detections
        avg_confidence = (det1['confidence'] + det2['confidence']) / 2 * 0.9

        # Create interpolated detection
        interpolated_det = {
            "track_id": det1["track_id"],
            "class_id": det1["class_id"],  # Use same class (already smoothed)
            "class_name": det1["class_name"],
            "confidence": avg_confidence,
            "bbox": interpolated_bbox,
            "segmentation": None,  # Don't interpolate segmentation
            "area": interpolated_bbox[2] * interpolated_bbox[3],
            "sam_applied": False,
            "interpolated": True  # Mark as interpolated
        }

        return interpolated_det

    def _rebuild_annotations(
            self,
            original_annotations: List[Dict],
            track_db: Dict[int, Dict]
    ) -> List[Dict]:
        """Rebuild frame annotations from smoothed track database."""
        frame_detections = defaultdict(list)

        # Collect all detections by frame from smoothed tracks
        for track_id, track_data in track_db.items():
            for frame_id, detection in track_data['detections'].items():
                frame_detections[frame_id].append(detection)

        # Rebuild annotations maintaining original frame structure
        smoothed_annotations = []

        for frame_data in original_annotations:
            frame_id = frame_data["frame_id"]

            new_frame_data = {
                "frame_id": frame_id,
                "file_name": frame_data["file_name"],
                "detections": frame_detections.get(frame_id, [])
            }

            smoothed_annotations.append(new_frame_data)

        return smoothed_annotations