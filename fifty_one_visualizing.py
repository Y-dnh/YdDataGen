"""
FiftyOne COCO Dataset Loader with Track ID Support

This script loads COCO format annotations and images into FiftyOne for visualization.
Supports both bounding box detections and polygon segmentation masks with track IDs.

Features:
- Loads images from multiple subdirectories
- Converts COCO bounding boxes to FiftyOne format
- Handles polygon segmentations as separate polylines
- Supports track_id for object tracking across frames
- Progress tracking with tqdm
- Safe error handling and validation
"""

import fiftyone as fo
from fiftyone import ViewField as F
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_coco_to_fiftyone(data_path: str, labels_path: str, dataset_name: str = "merged_dataset1"):
    """
    Load COCO dataset into FiftyOne for visualization and analysis.

    Args:
        data_path (str): Path to directory containing image subdirectories
        labels_path (str): Path to COCO JSON annotation file
        dataset_name (str): Name for the FiftyOne dataset

    Returns:
        fo.Dataset: FiftyOne dataset with loaded samples
    """

    # Convert paths to Path objects for easier manipulation
    data_path = Path(data_path)
    labels_path = Path(labels_path)

    # Load COCO annotation file
    print("Loading COCO annotations...")
    with open(labels_path) as f:
        coco_data = json.load(f)

    # Create category mapping from ID to name
    category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    print(f"Found {len(category_map)} categories: {list(category_map.values())}")

    # Delete existing dataset if it exists to avoid conflicts
    if fo.dataset_exists(dataset_name):
        print(f"Deleting existing dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)

    # Create new FiftyOne dataset
    dataset = fo.Dataset(dataset_name)

    # Create annotation lookup for faster processing
    # Maps image_id -> list of annotations for that image
    print("Building annotation index...")
    annotations_by_image = {}
    track_ids = set()  # Keep track of all unique track IDs

    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

        # Collect track IDs if present
        if "track_id" in ann:
            track_ids.add(ann["track_id"])

    print(f"Found {len(track_ids)} unique track IDs" if track_ids else "No track IDs found in annotations")
    print(f"Processing {len(coco_data['images'])} images...")

    # Process each image in the dataset
    for img_info in tqdm(coco_data["images"], desc="Processing images"):

        file_name = img_info["file_name"]
        img_path = None

        # Search for image in subdirectories
        # This allows for images to be organized in multiple folders
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                potential_path = subdir / file_name
                if potential_path.exists():
                    img_path = potential_path
                    break

        # Skip if image not found
        if img_path is None:
            print(f"Image not found: {file_name}")
            continue

        # Get image dimensions for coordinate normalization
        # FiftyOne requires normalized coordinates [0, 1]
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # Create FiftyOne sample
        sample = fo.Sample(filepath=str(img_path))

        # Lists to store detections and segmentations
        detections = []
        polylines = []  # For segmentation polygons
        img_id = img_info["id"]

        # Add frame number or timestamp if available in image info
        if "frame_id" in img_info:
            sample["frame_id"] = img_info["frame_id"]
        if "timestamp" in img_info:
            sample["timestamp"] = img_info["timestamp"]

        # Process annotations for this image
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                label = category_map[ann["category_id"]]

                # Convert COCO bbox format [x, y, width, height] (pixels)
                # to FiftyOne format [x, y, width, height] (normalized 0-1)
                x, y, w, h = ann["bbox"]

                # Normalize coordinates to [0, 1] range
                norm_x = x / img_width
                norm_y = y / img_height
                norm_w = w / img_width
                norm_h = h / img_height

                # Validate bounding box coordinates
                if norm_x < 0 or norm_y < 0 or norm_x + norm_w > 1 or norm_y + norm_h > 1:
                    print(f"Invalid bbox for {file_name}: {[norm_x, norm_y, norm_w, norm_h]}")
                    continue

                # Create detection object with track_id support
                detection_kwargs = {
                    "label": label,
                    "bounding_box": [norm_x, norm_y, norm_w, norm_h],
                    "confidence": ann.get("score", 1.0)  # Use confidence if available
                }

                # Add track_id if present
                if "track_id" in ann:
                    detection_kwargs["track_id"] = ann["track_id"]

                # Add any additional annotation attributes
                if "area" in ann:
                    detection_kwargs["area"] = ann["area"]
                if "iscrowd" in ann:
                    detection_kwargs["iscrowd"] = ann["iscrowd"]

                detection = fo.Detection(**detection_kwargs)

                # Process segmentation polygons
                if "segmentation" in ann and ann["segmentation"]:
                    try:
                        segmentation = ann["segmentation"]

                        # Handle polygon segmentation format
                        if isinstance(segmentation, list) and len(segmentation) > 0:
                            for seg in segmentation:
                                # Each segment is a flat list [x1, y1, x2, y2, ...]
                                if isinstance(seg, list) and len(seg) >= 6:  # Minimum 3 points

                                    # Convert flat list to list of [x, y] points
                                    normalized_points = []
                                    for j in range(0, len(seg), 2):
                                        if j + 1 < len(seg):
                                            # Normalize polygon coordinates
                                            norm_x_poly = seg[j] / img_width
                                            norm_y_poly = seg[j + 1] / img_height
                                            normalized_points.append([norm_x_poly, norm_y_poly])

                                    # Create polyline if we have enough points
                                    if len(normalized_points) >= 3:
                                        polyline_kwargs = {
                                            "label": label,
                                            "points": [normalized_points],  # List of points as one contour
                                            "closed": True,  # Close the polygon
                                            "filled": True  # Fill the polygon
                                        }

                                        # Add track_id to polyline if present
                                        if "track_id" in ann:
                                            polyline_kwargs["track_id"] = ann["track_id"]

                                        polyline = fo.Polyline(**polyline_kwargs)
                                        polylines.append(polyline)

                    except Exception as e:
                        print(f"Error processing segmentation for {file_name}: {e}")

                detections.append(detection)

        # Add detections to sample if any exist
        if detections:
            sample["detections"] = fo.Detections(detections=detections)

        # Add segmentations to sample if any exist
        if polylines:
            sample["segmentations"] = fo.Polylines(polylines=polylines)

        # Add sample to dataset
        dataset.add_sample(sample)

    return dataset


def print_dataset_stats(dataset: fo.Dataset):
    """
    Print comprehensive statistics about the dataset.

    Args:
        dataset (fo.Dataset): FiftyOne dataset to analyze
    """
    print(f"\nDataset created with {len(dataset)} samples")

    # Count samples with different types of annotations
    detections_count = 0
    segmentations_count = 0
    tracked_detections_count = 0
    unique_track_ids = set()

    print("Computing dataset statistics...")
    for sample in tqdm(dataset, desc="Counting annotations"):
        # Check for detection annotations
        if hasattr(sample, 'detections') and sample.detections and len(sample.detections.detections) > 0:
            detections_count += 1

            # Count detections with track IDs
            for detection in sample.detections.detections:
                if hasattr(detection, 'track_id') and detection.track_id is not None:
                    tracked_detections_count += 1
                    unique_track_ids.add(detection.track_id)

        # Check for segmentation annotations
        if hasattr(sample, 'segmentations') and sample.segmentations and len(sample.segmentations.polylines) > 0:
            segmentations_count += 1

    print(f"Samples with detections: {detections_count}")
    print(f"Samples with segmentations: {segmentations_count}")
    print(f"Samples with tracked detections: {tracked_detections_count}")
    print(f"Unique track IDs found: {len(unique_track_ids)}")

    # Compute and display metadata
    print("\nComputing metadata...")
    dataset.compute_metadata()
    print(dataset.stats())


def analyze_tracks(dataset: fo.Dataset):
    """
    Analyze tracking information in the dataset.

    Args:
        dataset (fo.Dataset): FiftyOne dataset to analyze
    """
    print("\n=== TRACK ANALYSIS ===")

    # Dictionary to store track information: track_id -> list of frame info
    tracks = {}

    for sample in dataset:
        if hasattr(sample, 'detections') and sample.detections:
            frame_info = {
                'filepath': sample.filepath,
                'frame_id': getattr(sample, 'frame_id', None),
                'timestamp': getattr(sample, 'timestamp', None)
            }

            for detection in sample.detections.detections:
                if hasattr(detection, 'track_id') and detection.track_id is not None:
                    track_id = detection.track_id
                    if track_id not in tracks:
                        tracks[track_id] = []

                    track_info = {
                        'frame_info': frame_info,
                        'label': detection.label,
                        'confidence': detection.confidence,
                        'bbox': detection.bounding_box
                    }
                    tracks[track_id].append(track_info)

    if tracks:
        print(f"Found {len(tracks)} unique tracks")

        # Analyze track lengths
        track_lengths = [len(track_frames) for track_frames in tracks.values()]
        print(f"Average track length: {np.mean(track_lengths):.2f} frames")
        print(f"Longest track: {max(track_lengths)} frames")
        print(f"Shortest track: {min(track_lengths)} frames")

        # Show some example tracks
        print("\nExample tracks:")
        for track_id, track_frames in list(tracks.items())[:5]:
            labels = set(frame['label'] for frame in track_frames)
            print(f"Track {track_id}: {len(track_frames)} frames, labels: {labels}")
    else:
        print("No tracks found in dataset")


def main():
    """
    Main function to load COCO dataset and launch FiftyOne app.
    """
    # Configuration
    DATA_PATH = "C:/YtDataGen/dataset/data"
    LABELS_PATH = "C:/YtDataGen/dataset/annotations_per_videos/v4vO49ekCmg_annotations.json"
    DATASET_NAME = "merged_dataset_with_tracks"
    PORT = 5151

    try:
        # Load dataset
        dataset = load_coco_to_fiftyone(DATA_PATH, LABELS_PATH, DATASET_NAME)

        # Print statistics
        print_dataset_stats(dataset)

        # Analyze tracks
        analyze_tracks(dataset)

        # Launch FiftyOne app
        print(f"\nLaunching FiftyOne app on port {PORT}...")
        session = fo.launch_app(dataset, port=PORT)
        print(f"FiftyOne running at http://localhost:{PORT}")

        # Keep the session alive
        print("Press Ctrl+C to stop the session")
        session.wait()

    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()