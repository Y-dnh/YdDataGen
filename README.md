# YtDataGen - YouTube Video Dataset Generation Tool

YtDataGen is a comprehensive tool for generating computer vision datasets from YouTube videos. It automates the entire pipeline from video downloading to creating COCO-format annotations with object detection, tracking, and segmentation capabilities.

## Features

- **Automated Video Processing**: Download YouTube videos and extract frames
- **Multi-Model Inference**: YOLO for object detection + SAM for segmentation
- **Object Tracking**: Support for BoT-SORT and ByteTrack trackers
- **Static Object Detection**: Specialized detection for stationary objects
- **COCO Format Output**: Industry-standard annotation format
- **Comprehensive Reporting**: Automated PDF report generation
- **Visualization Tools**: Create annotated videos for validation
- **CVAT Converter Tools**: Convert COCO annotations to CVAT for video 
- **FiftyOne launcher**: Open your dataset in FiftyOne app
- **Flexible Configuration**: Extensive customization options

## Table of Contents

- [Visual Results](#visual-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Arguments](#command-line-arguments)
- [Configuration](#configuration)
- [Module Architecture](#module-architecture)
- [Output Structure](#output-structure)
- [Visualization](#visualization)
- [CVAT Export](#cvat-export)
- [Model Management](#model-management)
- [Performance Comparison](#performance-comparison)

## Visual Results

Below are clickable previews that link to YouTube videos

| Only detection on 8k video | Detection with segmentation with people walking | Football match |
|-----------------------------|-----------------------------------------------|----------------|
| [![8k detection](https://img.youtube.com/vi/tDtCKGIMQ7w/hqdefault.jpg)](https://youtu.be/tDtCKGIMQ7w) | [![People walking](https://img.youtube.com/vi/iY_VSz0VP_E/hqdefault.jpg)](https://www.youtube.com/watch?v=iY_VSz0VP_E) | [![Football match](https://img.youtube.com/vi/m11oYyfsEas/hqdefault.jpg)](https://www.youtube.com/watch?v=m11oYyfsEas) |

| Detection with segmentation of road traffic | Detection with segmentation on road traffic | Light tracking of cars and people |
|---------------------------------------------|---------------------------------------------|----------------------------------|
| [![Road traffic detection](https://img.youtube.com/vi/BdRI90mrDNA/hqdefault.jpg)](https://www.youtube.com/watch?v=BdRI90mrDNA) | [![Road traffic segmentation](https://img.youtube.com/vi/pQPVOvtUTik/hqdefault.jpg)](https://youtu.be/pQPVOvtUTik) | [![Light tracking](https://img.youtube.com/vi/wiBmTBmQ5qw/hqdefault.jpg)](https://www.youtube.com/watch?v=wiBmTBmQ5qw) |

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- FFmpeg (for video processing)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Y-dnh/YdDataGen.git
cd YdDataGen

# Install requirements
pip install -r requirements.txt
```
or
```bash
conda env create -f environment.yml
conda activate YdDataGen
```

Rewrite the config.py file with your own tracker parameters. If you want to control it after loading the models, rewrite the tracker yaml file.

```bash
# Download required models
python download_models.py
```

## Quick Start

1. **Prepare URLs file**: Create `urls.txt` with YouTube URLs (one per line)
   ```
   https://www.youtube.com/watch?v=VIDEO_ID1
   https://www.youtube.com/watch?v=VIDEO_ID2 00:00:10 00:01:30
   https://www.youtube.com/watch?v=VIDEO_ID3 
   ```

2. **Run the pipeline**:
   ```bash
   python main.py --urls urls.txt
   ```

3. **Check outputs**: Find results in `dataset/` directory

## Command Line Arguments

### Basic Usage
```bash
python main.py --urls <path_to_urls> [OPTIONS]
```

## Command Line Arguments

### Core Arguments

| Argument             | Type | Default      | Description                               |
| -------------------- | ---- | ------------ | ----------------------------------------- |
| `--urls`, `-u`       | str  | **Required** | Path to text file containing YouTube URLs |
| `--output-dir`, `-o` | str  | Current dir  | Output directory for all results          |

### Model Configuration

| Argument       | Type | Default                  | Description                          |
| -------------- | ---- | ------------------------ | ------------------------------------ |
| `--yolo-model` | str  | `CONFIG.yolo_model_path` | YOLO model path for object detection |
| `--sam-model`  | str  | `CONFIG.sam_model_path`  | SAM model path for segmentation      |
| `--tracker`    | str  | `CONFIG.tracker_type`    | Tracker configuration file           |

### Feature Toggles

| Argument             | Type | Default | Description                                 |
| -------------------- | ---- | ------- | ------------------------------------------- |
| `--no-sam`           | flag | False   | Disable SAM segmentation                    |
| `--no-static-cars`   | flag | False   | Disable static car detection                |
| `--no-smoothing`     | flag | False   | Disable track smoothing                     |
| `--no-interpolation` | flag | False   | Disable interpolation of missing detections |

### Processing Control

| Argument             | Type | Default | Description                                        |
| -------------------- | ---- | ------- | -------------------------------------------------- |
| `--skip-download`    | flag | False   | Skip video download, use existing files            |
| `--skip-frames`      | flag | False   | Skip frame extraction, use existing frames         |
| `--skip-inference`   | flag | False   | Skip inference (detection, tracking, segmentation) |
| `--skip-annotations` | flag | False   | Skip annotation (COCO JSON) generation             |
| `--skip-report`      | flag | False   | Skip consolidated report generation                |

### Logging Options

| Argument          | Type | Default | Description                  |
| ----------------- | ---- | ------- | ---------------------------- |
| `--verbose`, `-v` | flag | False   | Verbose logging output       |
| `--debug`         | flag | False   | Enable debug-level logging   |
| `--quiet`, `-q`   | flag | False   | Suppress most logging output |

### Example Commands

```bash
# Basic usage with default settings
python main.py --urls urls.txt

# Run with custom YOLO + SAM models
python main.py --urls urls.txt --yolo-model yolov8x.pt --sam-model sam2.1_l.pt

# Fast processing without segmentation and interpolation
python main.py --urls urls.txt --no-sam --no-interpolation

# Save results to custom directory with verbose logs
python main.py --urls urls.txt --output-dir /path/to/output --verbose

# Skip downloading and reuse existing frames for inference
python main.py --urls urls.txt --skip-download --skip-frames
```

## Configuration

The system uses a centralized configuration in `src/config.py`. Key settings include:

### Custom Classes
You can write your own classes
```python
custom_classes = {
    0: "person",
    1: "pet", 
    2: "car"
}
```

1. Model Paths
2. Logging settings
3. Download settings
4. Tracker settings
5. YOLO settings
6. SAM settings
7. Polygon settings
   

Douglas-Peucker Polygon Simplification Examples

| Max Points = 5 | Max Points = 10 | Max Points = 20 |
|-------------|--------------|--------------|
| ![douglas_peucker_5](https://github.com/user-attachments/assets/7d5a65f8-ac24-4d09-9378-7692b146673f) | ![douglas_peucker_10](https://github.com/user-attachments/assets/fa299abc-4d2e-4d2a-b51f-ea7244a923f6) | ![douglas_peucker_20](https://github.com/user-attachments/assets/dd7d7073-293b-4a9c-84c4-2d0f23a1071f) |



8. Static Car Detection
9. CVAT conversion settings

## Module Architecture

### Core Modules

#### 1. `main.py` - Entry Point
**Purpose**: Orchestrates the entire pipeline

#### 2. `config.py` - Configuration Management
**Purpose**: Centralized configuration system

**Key Classes**:
- `ProjectPaths`: Directory structure management
- `Config`: Main configuration container

#### 3. `download.py` - Video Acquisition
**Purpose**: YouTube video downloading with time constraints

**URL Format Support**:
```
https://youtube.com/watch?v=VIDEO_ID
https://youtube.com/watch?v=VIDEO_ID 00:00:10 00:01:30  # 10s to 1m30s
https://youtube.com/watch?v=VIDEO_ID 
```

#### 4. `extract_frames.py` - Frame Extraction
**Purpose**: Convert videos to individual frames

#### 5. `inference.py` - AI Processing Engine
**Purpose**: Core computer vision processing
- YOLO object detection
- SAM segmentation (optional)
- Object tracking integration
- Static object analysis

**Processing Pipeline**:
1. **Detection**: YOLO identifies objects and assigns tracking IDs
2. **Tracking**: Maintains object identity across frames
3. **Segmentation**: SAM generates precise masks (if enabled)
4. **Polygon Processing**: Simplifies and optimizes segmentation masks
5. **Static Analysis**: Identifies non-moving objects
6. **Track and Class smoother**: Smooth tracks, fix classifications

**Advanced Features**:
- **Batch SAM Processing**: Optimized segmentation for multiple objects
- **Polygon Simplification**: Douglas-Peucker algorithm for efficient masks
- **Memory Management**: Automatic cleanup for long video processing
- **Dynamic Tracking**: Configurable tracker types and parameters

#### 5. `track_smoother.py` - Track smoothing, classification fixing, false positive reduction
**Purpose**: Post-processing tracks to improve detection quality and reduce noise
- Track Filtering: removal of tracks that are too short (likely false positives)
- Class Smoothing: elimination of class “flickering,” determination of the dominant class for the entire track
- Gap Filling: interpolation of bbox between adjacent frames to restore missed detections
- Confidence-Based Filtering: use of confidence thresholds for interpolation and class selection decisions
- Annotation Rebuild: generation of updated annotations taking into account smoothed tracks

#### 6. `annotations.py` - COCO Format Generation
**Purpose**: Convert detections to standard COCO format
- Individual video annotations
- Combined dataset creation

**Output Structure**:
```json
{
  "info": {
    "description": "YtDataGen Video Dataset",
    "version": "1.0",
    "date_created": "2024-01-01T00:00:00"
  },
  "categories": [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "pet"},
    {"id": 2, "name": "car"}
  ],
  "videos": [
    {
      "id": 0,
      "name": "video_id",
      "fps": 30.0,
      "frames": 900,
      "width": 1920,
      "height": 1080
    }
  ],
  "images": [
    {
      "id": 1,
      "width": 1920,
      "height": 1080,
      "file_name": "video_id_00000.jpg",
      "frame_id": 0
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 100, 200, 300],
      "area": 60000,
      "segmentation": [[x1,y1,x2,y2,...]],
      "track_id": 1,
      "score": 0.85
    }
  ]
}
```

#### 7. `report_generator.py` - Documentation
**Purpose**: Automated report generation

**Report Sections**:
- Configuration overview
- Dataset statistics  
- Per-video analysis
- Processing performance

#### 8. `utils.py` - Utility Functions
**Purpose**: Common functionality and helpers

#### 9. `visualization.py` - Visual Validation
**Purpose**: Create annotated videos for validation

#### 10. `download_models.py` - Model Management
**Purpose**: Automated model downloading and setup
- YOLO model variants (n, s, m, l, x)
- SAM model options
- Tracker configuration generation
- Dependency verification

#### 11. `cvat_converter.py` - CVAT Xml creator
**Purpose**: Rebuild COCO json into XML format for uploading annotations into CVAT

#### 12. `fifty_one_visualizing.py` - FiftyOne launcher
**Purpose**: Parse all files in data, create new dataset and use labels_final as annotation file

## Output Structure

```
project_root/
├── dataset/
│   ├── videos/                     # Downloaded videos
│   │   ├── VIDEO_ID1.mp4
│   │   └── VIDEO_ID2.mp4
│   ├── data/                       # Extracted frames
│   │   ├── VIDEO_ID1/
│   │   │   ├── VIDEO_ID1_00000.jpg
│   │   │   └── VIDEO_ID1_00001.jpg
│   │   └── VIDEO_ID2/
│   ├── annotations_per_videos/     # Individual annotations
│   │   ├── VIDEO_ID1_annotations.json
│   │   └── VIDEO_ID2_annotations.json
│   ├── labels_final.json          # Combined COCO dataset
│   └── report.pdf                 # Processing report
├── models/
│   ├── yolo_det/                  # YOLO models
│   ├── sam/                       # SAM models
│   └── trackers/                  # Tracker configs
├── logs/
│   └── ytdatagen.log             # Processing logs
└── visualized_videos/             # Annotated videos (optional)
│   ├── VIDEO_ID1_visualized.mp4
│   └── VIDEO_ID2_visualized.mp4
└── cvat_annotations/             # CVAT for video 1.1 ver.
    ├── VIDEO_ID1.xml
    └── VIDEO_ID2.xml
```

## Visualization

Create annotated videos to validate your dataset:

### Basic Visualization
```bash
python visualization.py --all
```

### Custom Visualization Options
```bash
python visualization.py -f VIDEO_ID_annotations.json \
  --no-masks \
  --no-confidence
```

### Visualization Options
- `--no-boxes`: Hide bounding boxes
- `--no-masks`: Hide segmentation masks  
- `--no-tracks`: Hide tracking IDs
- `--no-labels`: Hide class labels
- `--no-confidence`: Hide confidence scores

## CVAT Export
Convert your COCO annotations to CVAT format for advanced annotation editing and review.

### Advanced Keyframing Options
```bash
# Smart FPS-based keyframes (every 2 seconds)
python cvat_converter.py --all --keyframe-mode fps --fps-multiplier 2.0

# Fixed interval keyframes (every 50 frames)
python cvat_converter.py --all --keyframe-mode interval --keyframe-interval 50

# Auto-adaptive keyframes
python cvat_converter.py --all --keyframe-mode auto
```

### Keyframe Modes
- `fps`: Keyframes every N seconds based on video FPS
- `interval`: Fixed frame intervals  
- `auto`: Intelligent adaptive keyframing

## Model Management

### Download All Models
```bash
python download_models.py
```

### Available Models

**YOLO Detection Models**:
- `yolov8` series

**SAM Segmentation Models**:
- `sam2.1` series
- `mobile_sam.pt`

### Custom Models
Place custom models in appropriate directories:
- YOLO: `models/yolo_det/your_model.pt`
- SAM: `models/sam/your_model.pt`

## Performance Comparison

This comparison analyzes the performance of different YOLO and SAM model configurations on the same 3-second video (NBWd_5AZ79E) with 75 frames at 25 FPS.

### Test Configurations

| Configuration | YOLO Resolution | SAM Model | SAM Resolution | Input Video Resolution |
|---------------|----------------|-----------|----------------|----------------------|
| Config 1 | 640x640 | mobile_sam.pt | 640x640 | 640x360 |
| Config 2 | 640x640 | sam2.1_l.pt | 640x640 | 640x360 |
| Config 3 | 1024x1024 | mobile_sam.pt | 1024x1024 | 1920x1080 |
| Config 4 | 1024x1024 | sam2.1_l.pt | 1024x1024 | 1920x1080 |

### Performance Metrics

#### Detection Accuracy

| Metric | 640 + Mobile SAM | 640 + SAM2.1-L | 1024 + Mobile SAM | 1024 + SAM2.1-L |
|--------|------------------|-----------------|-------------------|------------------|
| **Total Detections** | 6,349 | 6,349 | 10,987 | 10,987 |
| **Average Confidence** | 0.713 | 0.713 | 0.737 | 0.737 |
| **People Detections** | 2,270 | 2,270 | 5,391 | 5,391 |
| **Car Detections** | 4,079 | 4,079 | 5,596 | 5,596 |
| **People Tracks** | 73 | 73 | 158 | 158 |
| **Car Tracks** | 73 | 73 | 98 | 98 |
| **Total Tracks** | 146 | 146 | 256 | 256 |

#### Processing Performance

| Metric | 640 + Mobile SAM | 640 + SAM2.1-L | 1024 + Mobile SAM | 1024 + SAM2.1-L |
|--------|------------------|-----------------|-------------------|------------------|
| **Inference Speed** | 5.02 FPS | 4.10 FPS | 1.03 FPS | 0.85 FPS |
| **Processing Time** | 14s | 18s | 1m 12s | 1m 28s |
| **Speed vs Real-time** | 5.02/25 = 20% | 4.10/25 = 16% | 1.03/25 = 4% | 0.85/25 = 3% |

| 640 + Mobile SAM | 640 + SAM2.1-L |
|------------------|----------------|
| <img width="760" alt="640_yolol_mobile_640" src="https://github.com/user-attachments/assets/11590288-9b30-4d85-bd1e-6b1bdb729993" /> | <img width="760" alt="640_yolol_saml_640" src="https://github.com/user-attachments/assets/c0aef4d5-e8de-4b88-976e-06740c5af50a" /> |

| 1024 + Mobile SAM | 1024 + SAM2.1-L |
|-------------------|-----------------|
| <img width="760" alt="1080_yolol_mobile_1024" src="https://github.com/user-attachments/assets/4237192f-560f-4342-ab2a-2347f585a209" /> | <img width="760" alt="1080_yolol_saml_1024" src="https://github.com/user-attachments/assets/35e07773-8077-4989-a471-70468d0a30e6" /> |


###

## Configuration Details

All tests used consistent parameters:
- YOLO Confidence: 0.4
- YOLO IoU: 0.6  
- SAM Confidence: 0.4
- SAM IoU: 0.5
- Tracking: BotSORT with ReID enabled
- Device: CUDA A100 gpu
- Half Precision: Disabled

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO and SAM implementations
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube downloading
- [COCO API](https://github.com/cocodataset/cocoapi) for annotation format
- [ReportLab](https://www.reportlab.com/) for PDF generation
