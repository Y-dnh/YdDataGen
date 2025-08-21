#!/usr/bin/env python3
"""
YtDataGen - Optimized YouTube Video Dataset Generation Tool
Streamlined main entry point for generating a consolidated report.
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Dict, Any

from src.config import CONFIG
from src.utils import setup_project, get_video_info
from src.download import download_videos
from src.extract_frames import extract_frames
from src.inference import InferenceEngine
from src.annotations import COCOAnnotationGenerator
from src.report_generator import ConsolidatedReportGenerator

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='YtDataGen - Generate video datasets with object detection and tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --urls urls.txt
  %(prog)s --urls urls.txt --yolo-model yolo8s.pt --confidence 0.7
  %(prog)s --urls urls.txt --output-dir /path/to/output --device cuda
  %(prog)s --urls urls.txt --no-sam --static-cars
        """
    )

    parser.add_argument('--urls', '-u', type=str, required=True,
                        help='Path to text file containing YouTube URLs')

    parser.add_argument('--yolo-model', type=str, default=CONFIG.yolo_model_path,
                        help=f'YOLO model path (default: {CONFIG.yolo_model_path})')
    parser.add_argument('--sam-model', type=str, default=CONFIG.sam_model_path,
                        help=f'SAM model path (default: {CONFIG.sam_model_path})')
    parser.add_argument('--tracker', type=str, default=CONFIG.tracker_type,
                        help=f'Tracker type (default: {CONFIG.tracker_type})')

    parser.add_argument('--confidence', type=float, default=CONFIG.yolo_confidence,
                        help=f'YOLO confidence threshold (default: {CONFIG.yolo_confidence})')
    parser.add_argument('--iou', type=float, default=CONFIG.yolo_iou,
                        help=f'YOLO IoU threshold (default: {CONFIG.yolo_iou})')
    parser.add_argument('--sam-conf', type=float, default=CONFIG.sam_confidence,
                        help=f'SAM confidence threshold (default: {CONFIG.sam_confidence})')

    parser.add_argument('--no-sam', action='store_true', help='Disable SAM segmentation')
    parser.add_argument('--static-cars', action='store_true', default=CONFIG.static_car_enabled,
                        help='Enable static car detection')
    parser.add_argument('--no-static-cars', action='store_true', help='Disable static car detection')
    parser.add_argument('--max-points', type=int, default=CONFIG.max_points,
                        help=f'Max polygon points (default: {CONFIG.max_points})')

    parser.add_argument('--output-dir', '-o', type=str, help='Output directory')
    parser.add_argument('--no-report', action='store_true', help='Skip consolidated report generation')  # <<< ЗМІНЕНО

    parser.add_argument('--skip-download', action='store_true', help='Skip video download')
    parser.add_argument('--skip-frames', action='store_true', help='Skip frame extraction')

    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                        help='Device to use for inference')
    parser.add_argument('--half-precision', action='store_true', default=CONFIG.half_precision,
                        help='Use FP16 inference')

    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--debug', action='store_true', help='Debug logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')

    return parser.parse_args()


def update_config_from_args(args: argparse.Namespace):
    CONFIG.paths.urls_file = Path(args.urls)
    if args.output_dir:
        CONFIG.paths.root = Path(args.output_dir)
        CONFIG.paths = CONFIG.paths.__class__()

    CONFIG.yolo_model_path = args.yolo_model
    CONFIG.sam_model_path = args.sam_model
    CONFIG.tracker_type = args.tracker
    CONFIG.yolo_confidence = args.confidence
    CONFIG.yolo_iou = args.iou
    CONFIG.sam_confidence = args.sam_conf
    CONFIG.max_points = args.max_points

    if args.no_sam:
        CONFIG.sam_enabled = False
    if args.no_static_cars:
        CONFIG.static_car_enabled = False
    elif args.static_cars:
        CONFIG.static_car_enabled = True


    CONFIG.half_precision = args.half_precision

    if args.debug:
        CONFIG.log_level = 'DEBUG'
    elif args.verbose:
        CONFIG.log_level = 'INFO'
    elif args.quiet:
        CONFIG.log_level = 'ERROR'


def process_video(
        video_id: str,
        video_info: Dict,
        inference_engine: InferenceEngine,
) -> Dict:
    logger.info(f"Processing video: {video_id}")
    annotation_generator = COCOAnnotationGenerator()
    try:
        video_results = inference_engine.process_video(video_id, video_info)
        if not video_results or not video_results.get("annotations"):
            logger.warning(f"No results returned from inference for video {video_id}")
            return None

        annotation_file = annotation_generator.save_video_annotations(
            video_results, video_id, video_info
        )
        video_results['annotation_file_path'] = annotation_file

        inference_engine.clear_memory()

        logger.info(f"Completed processing for {video_id}")
        return video_results

    except Exception as e:
        logger.error(f"Failed to process video {video_id}: {e}", exc_info=True)
        inference_engine.clear_memory()
        return None


def main():
    args = parse_arguments()

    update_config_from_args(args)
    setup_project()
    logger.info("=" * 60)
    logger.info("YtDataGen - Optimized Configuration")
    logger.info(f"YOLO Model: {CONFIG.yolo_model_path}, Device: {CONFIG.device}")
    logger.info("=" * 60)

    try:

        video_info_dict = {}
        if not args.skip_download:
            logger.info("Step 1/4: Downloading videos")
            video_info_dict = download_videos()
            if not video_info_dict:
                logger.error("No videos were downloaded. Exiting.")
                return 1
        else:
            logger.info("Step 1/4: Skipped download, using existing videos.")
            existing_videos = list(CONFIG.paths.videos_dir.glob("*.mp4"))
            if not existing_videos:
                logger.error(f"No videos found in {CONFIG.paths.videos_dir} to process.")
                return 1
            for video_file in existing_videos:
                video_info_dict[video_file.stem] = {"path": str(video_file)}

        if not args.skip_frames:
            logger.info("Step 2/4: Extracting frames")
            extract_frames(video_info_dict)
        else:
            logger.info("Step 2/4: Skipped frame extraction")

        logger.info("Gathering detailed video metadata...")
        for video_id, info in video_info_dict.items():
            if "path" in info and Path(info["path"]).exists():
                detailed_info = get_video_info(Path(info["path"]))
                if detailed_info:
                    info.update(detailed_info)
                else:
                    logger.warning(f"Could not get metadata for {video_id}")
            else:
                logger.warning(f"Path for video {video_id} is missing or invalid.")

        logger.info("Step 3/4: Running inference and generating annotations")


        all_video_results = {}
        processed_annotation_files = []

        for video_id, video_info in video_info_dict.items():
            inference_engine = InferenceEngine()
            results = process_video(video_id, video_info, inference_engine)
            if results:
                all_video_results[video_id] = results
                if "annotation_file_path" in results:
                    processed_annotation_files.append(results["annotation_file_path"])


        if processed_annotation_files:
            logger.info("Step 4/4: Creating final combined annotations file...")

            final_annotator = COCOAnnotationGenerator()

            final_annotation_file = final_annotator.save_final_annotations(
                processed_annotation_files,
                video_info_dict
                # todo check video quality
            )

            if final_annotation_file:
                logger.info(f"Final annotations file created: {final_annotation_file}")

                logger.info("Generating consolidated final report...")
                report_generator = ConsolidatedReportGenerator()
                report_path = report_generator.generate_consolidated_report(
                    all_video_results,
                    video_info_dict
                )
                logger.info(f"Successfully generated consolidated report: {report_path}")
            else:
                logger.error("Failed to create the final annotations file. Report generation skipped.")

        else:
            logger.warning("No videos were processed successfully. Skipping final report and annotation generation.")

        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total videos processed successfully: {len(all_video_results)}")
        logger.info(f"Final combined annotation file is at: {final_annotation_file}")
        if not args.no_report and 'report_path' in locals():
            logger.info(f"Consolidated PDF report is at: {report_path}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user.")
        return 1
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.argv = [
        "main.py",
        "--urls", "urls.txt",
        "--yolo-model", "yolo8n_pt_512_coco_skiped_crowd.pt",
        "--sam-model", "mobile_sam.pt",
        "--tracker", "botsort.yaml",
    ]
    sys.exit(main())