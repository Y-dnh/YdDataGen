# YtDataGen - YouTube Video Dataset Generation Tool

YtDataGen is a comprehensive tool for generating computer vision datasets from YouTube videos. It automates the entire pipeline from video downloading to creating COCO-format annotations with object detection, tracking, and segmentation capabilities.

## Features

- **Automated Video Processing**: Download YouTube videos and extract frames
- **Multi-Model Inference**: YOLO for object detection + SAM for segmentation
- **Object Tracking**: Support for BoT-SORT and ByteTrack trackers
- **Static Car Detection**: Specialized detection for stationary vehicles
- **COCO Format Output**: Industry-standard annotation format
- **Comprehensive Reporting**: Automated PDF report generation
- **Visualization Tools**: Create annotated videos for validation
- **Flexible Configuration**: Extensive customization options

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Arguments](#command-line-arguments)
- [Configuration](#configuration)
- [Module Architecture](#module-architecture)
- [Usage Examples](#usage-examples)
- [Output Structure](#output-structure)
- [Visualization](#visualization)
- [Model Management](#model-management)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8+
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
python src/download_models.py
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

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--urls`, `-u` | str | **Required** | Path to text file containing YouTube URLs |
| `--output-dir`, `-o` | str | Current dir | Output directory for all results |

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--yolo-model` | str | `yolov8n.pt` | YOLO model path for object detection |
| `--sam-model` | str | `sam2.1_t.pt` | SAM model path for segmentation |
| `--tracker` | str | `botsort.yaml` | Tracker configuration file |

### Detection Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--confidence` | float | 0.5 | YOLO confidence threshold |
| `--iou` | float | 0.5 | YOLO IoU threshold for NMS |
| `--sam-conf` | float | 0.5 | SAM confidence threshold |

### Feature Toggles

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no-sam` | flag | False | Disable SAM segmentation |
| `--static-cars` | flag | True | Enable static car detection |
| `--no-static-cars` | flag | False | Disable static car detection |
| `--no-report` | flag | False | Skip PDF report generation |

### Processing Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--skip-download` | flag | False | Use existing videos, skip download |
| `--skip-frames` | flag | False | Use existing frames, skip extraction |
| `--max-points` | int | 100 | Maximum polygon points for segmentation |

### System Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | `auto` | Device: `cpu`, `cuda`, or `auto` |
| `--half-precision` | flag | False | Use FP16 inference for speed |

### Logging Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose`, `-v` | flag | False | Verbose logging output |
| `--debug` | flag | False | Enable debug-level logging |
| `--quiet`, `-q` | flag | False | Suppress most output |

### Example Commands

```bash
# Basic usage with default settings
python main.py --urls urls.txt

# High-quality processing with large models
python main.py --urls urls.txt --yolo-model yolov8x.pt --sam-model sam2.1_l.pt

# Fast processing without segmentation
python main.py --urls urls.txt --no-sam --yolo-model yolov8n.pt

# Custom output directory with verbose logging
python main.py --urls urls.txt --output-dir /path/to/output --verbose

# Process existing videos without downloading
python main.py --urls urls.txt --skip-download --confidence 0.7
```

## Configuration

The system uses a centralized configuration in `src/config.py`. Key settings include:

### Custom Classes
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
https://youtube.com/watch?v=VIDEO_ID 00:05:00           # From 5 minutes
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
5. **Static Analysis**: Identifies non-moving cars

**Advanced Features**:
- **Batch SAM Processing**: Optimized segmentation for multiple objects
- **Polygon Simplification**: Douglas-Peucker algorithm for efficient masks
- **Memory Management**: Automatic cleanup for long video processing
- **Dynamic Tracking**: Configurable tracker types and parameters

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

## Usage Examples

### Basic Processing
```bash
python main.py --urls urls.txt
```

### High-Quality Dataset Generation
```bash
python main.py --urls urls.txt \
  --yolo-model yolov8x.pt \
  --sam-model sam2.1_l.pt \
  --confidence 0.7 \
  --output-dir /data/my_dataset
```

### Fast Prototyping (No Segmentation)
```bash
python main.py --urls urls.txt \
  --yolo-model yolov8n.pt \
  --no-sam \
  --confidence 0.6
```

### Processing Existing Videos
```bash
python main.py --urls urls.txt \
  --skip-download \
  --confidence 0.8 \
  --static-cars
```

### Custom Tracker Configuration
```bash
python main.py --urls urls.txt \
  --tracker bytetrack.yaml \
  --confidence 0.6
```

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

## Model Management

### Download All Models
```bash
python src/download_models.py
```

### Available Models

**YOLO Detection Models**:
- `yolov8n.pt`: Nano (fastest)
- `yolov8s.pt`: Small 
- `yolov8m.pt`: Medium
- `yolov8l.pt`: Large
- `yolov8x.pt`: Extra Large (most accurate)
- `Your custom yolo model`

**SAM Segmentation Models**:
- `sam2.1_t.pt`: Tiny (fastest)
- `sam2.1_s.pt`: Small
- `sam2.1_b.pt`: Base
- `sam2.1_l.pt`: Large (most accurate)
- `mobile_sam.pt`: Mobile optimized

### Custom Models
Place custom models in appropriate directories:
- YOLO: `models/yolo_det/your_model.pt`
- SAM: `models/sam/your_model.pt`


## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO and SAM implementations
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube downloading
- [COCO API](https://github.com/cocodataset/cocoapi) for annotation format
- [ReportLab](https://www.reportlab.com/) for PDF generation
