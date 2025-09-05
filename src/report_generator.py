from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors

from src.config import CONFIG

logger = logging.getLogger(__name__)


class ConsolidatedReportGenerator:
    """Generates comprehensive PDF reports with dataset overview and per-video statistics."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Configure custom paragraph styles for consistent report formatting."""
        font_name = 'Helvetica'
        font_name_bold = 'Helvetica-Bold'

        # Section headers for main report sections
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['h2'],
            fontSize=16,
            spaceBefore=30,
            spaceAfter=20,
            textColor=colors.black,
            fontName=font_name_bold
        ))

        # Individual video titles within sections
        self.styles.add(ParagraphStyle(
            name='VideoTitle',
            parent=self.styles['h3'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.black,
            fontName=font_name_bold
        ))

        # Standardize body text formatting across the report
        body_style = self.styles['BodyText']
        body_style.fontName = font_name
        body_style.fontSize = 11
        body_style.spaceBefore = 8
        body_style.spaceAfter = 12
        body_style.textColor = colors.black
        body_style.leading = 16

    def generate_consolidated_report(self, all_video_results: Dict, video_info_dict: Dict = None) -> Path:
        """
        Generate a comprehensive PDF report combining configuration, dataset overview, and per-video statistics.

        Args:
            all_video_results: Dictionary mapping video_id to processing results
            video_info_dict: Optional dictionary with additional video metadata

        Returns:
            Path to the generated PDF report
        """
        logger.info("Generating consolidated PDF report...")

        report_path = CONFIG.paths.report_path
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72
        )

        content = []

        # Build report sections with page breaks between major sections
        content.extend(self._create_configuration_section())
        content.append(PageBreak())

        content.extend(self._create_dataset_overview(all_video_results, video_info_dict))

        # Add individual video sections
        for video_id, results in all_video_results.items():
            content.append(PageBreak())

            # Prioritize external video_info_dict over embedded results
            if video_info_dict and video_id in video_info_dict:
                video_info = video_info_dict[video_id]
            else:
                video_info = results.get("video_info", {})
            statistics = results.get("statistics", {})
            content.extend(self._create_video_section(video_id, video_info, statistics))

        content.extend(self._create_timestamp_section())

        doc.build(content)
        logger.info(f"Consolidated report generated: {report_path}")
        return report_path

    def _create_configuration_section(self) -> List:
        """Build comprehensive configuration overview section with all processing parameters."""
        content = [
            Paragraph("Configuration Overview", self.styles['SectionHeader']),
            Spacer(1, 0.2 * inch)
        ]

        # Format configuration as structured HTML for better readability
        config_text = f"""
        <b>Model Configuration:</b><br/>
        • YOLO Model: {CONFIG.yolo_model_path}<br/>
        • SAM Model: {CONFIG.sam_model_path if CONFIG.sam_enabled else 'Disabled'}<br/>
        • Tracker: {CONFIG.tracker_type}<br/>
        • Device: {CONFIG.device}<br/>

        <b>Resolution Settings:</b><br/>
        • YOLO Inference: {CONFIG.yolo_imgsz}<br/>
        • SAM Inference: {CONFIG.sam_imgsz}<br/><br/>

        <b>Detection Parameters:</b><br/>
        • YOLO Confidence: {CONFIG.yolo_confidence}<br/>
        • YOLO IoU Threshold: {CONFIG.yolo_iou}<br/>
        • Max Detections per Frame: {CONFIG.yolo_max_det}<br/>
        • Rectangular Inference: {CONFIG.yolo_rect}<br/>
        • Agnostic NMS: {CONFIG.yolo_agnostic_nms}<br/>
        • YOLO Halh: {CONFIG.yolo_half}<br/>
        • Augmented Inference: {CONFIG.yolo_augment}<br/><br/>

        <b>Tracking Parameters:</b><br/>
        • High Confidence Threshold: {CONFIG.track_high_thresh}<br/>
        • Low Confidence Threshold: {CONFIG.track_low_thresh}<br/>
        • New Track Threshold: {CONFIG.new_track_thresh}<br/>
        • Match Threshold: {CONFIG.match_thresh}<br/>
        • Track Buffer Frames: {CONFIG.track_buffer}<br/>
        • Fuse Detection Scores: {'Yes' if CONFIG.fuse_score else 'No'}<br/>
        • Global Motion Compensation: {CONFIG.gmc_method}<br/>
        • Proximity Threshold (ReID): {CONFIG.proximity_thresh}<br/>
        • Appearance Threshold (ReID): {CONFIG.appearance_thresh}<br/>
        • Re-Identification: {CONFIG.with_reid}<br/><br/>

        <b>Track Smoothing:</b><br/>
        • Smoothing Enabled: {CONFIG.track_smoothing_enabled}<br/>
        """

        # Add smoothing parameters only if enabled
        if CONFIG.track_smoothing_enabled:
            config_text += f"""
        • Minimum Track Length: {CONFIG.min_track_length} frames<br/>
        • Max Gap Fill: {CONFIG.max_gap_frames} frames<br/>
        • Min Confidence for Gap Fill: {CONFIG.min_confidence_for_gap_fill}<br/>
        • Class Smoothing Window: {CONFIG.class_smoothing_window} frames<br/>
        • Class Confidence Threshold: {CONFIG.class_confidence_threshold}<br/>
        • Gap Interpolation: {CONFIG.interpolate_missing_detections}<br/><br/>
            """
        else:
            config_text += "<br/>"

        # SAM Configuration
        config_text += f"""
        <b>SAM Segmentation:</b><br/>
        • SAM Enabled: {CONFIG.sam_enabled}<br/>
        """

        if CONFIG.sam_enabled:
            config_text += f"""
        • SAM Confidence: {CONFIG.sam_confidence}<br/>
        • SAM IoU Threshold: {CONFIG.sam_iou}<br/>
        • SAM Halh: {CONFIG.sam_half}<br/>
        • Retina Masks: {CONFIG.sam_retina_masks}<br/><br/>
            """
        else:
            config_text += "<br/>"

        # Static Object Detection
        config_text += f"""
        <b>Static Object Detection:</b><br/>
        • Static Detection: {CONFIG.static_car_enabled}<br/>
        """

        if CONFIG.static_car_enabled:
            config_text += f"""
        • Movement Threshold: {CONFIG.movement_threshold} pixels<br/>
        • Min Static Duration: {CONFIG.min_static_duration} frames<br/>
        • Static Check Interval: {CONFIG.static_check_interval} frames<br/><br/>
            """
        else:
            config_text += "<br/>"

        # Polygon and Segmentation Settings
        config_text += f"""
        <b>Polygon Configuration:</b><br/>
        • Max Points per Polygon: {CONFIG.max_points}<br/>
        • Minimum Area: {CONFIG.min_area} pixels²<br/>
        • Fill Polygon Holes: {CONFIG.fill_holes}<br/>
        • Approximation Method: {CONFIG.approximation_method.replace('_', ' ').title()}<br/><br/>

        <b>Class Mapping:</b><br/>
        """

        # Add class mapping
        for class_id, class_name in CONFIG.custom_classes.items():
            config_text += f"• Class {class_id}: {class_name}<br/>"

        content.append(Paragraph(config_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.6 * inch))
        return content

    def _create_dataset_overview(self, all_video_results: Dict, video_info_dict: Dict = None) -> List:
        """
        Generate comprehensive dataset-level statistics by aggregating all video results.

        Calculates totals, averages, and performance metrics across the entire dataset.
        """
        content = [
            Paragraph("Dataset Overview", self.styles['SectionHeader']),
            Spacer(1, 0.2 * inch)
        ]

        # Initialize accumulators for dataset-wide statistics
        total_videos = len(all_video_results)
        total_duration = 0
        total_frames = 0
        total_detections = 0
        confidence_sum = 0

        # Dynamic class counters based on CONFIG.custom_classes
        class_counters = {class_name: 0 for class_name in CONFIG.custom_classes.values()}
        track_counters = {class_name: 0 for class_name in CONFIG.custom_classes.values()}
        static_counters = {}
        if CONFIG.static_car_enabled:
            static_counters = {class_name: 0 for class_name in CONFIG.custom_classes.values()}

        total_processing_time = 0

        # Aggregate statistics from all processed videos
        for video_id, results in all_video_results.items():
            # Prioritize external video info over embedded results
            if video_info_dict and video_id in video_info_dict:
                video_info = video_info_dict[video_id]
            else:
                video_info = results.get("video_info", {})

            statistics = results.get("statistics", {})

            total_duration += video_info.get("duration", 0)
            total_frames += statistics.get("total_frames", 0)
            total_detections += statistics.get("total_detections", 0)

            # Weight confidence by detection count for accurate average
            confidence_sum += statistics.get("avg_confidence", 0) * statistics.get("total_detections", 0)
            total_processing_time += statistics.get("processing_time", 0)

            # Aggregate class-specific counts
            for class_name in CONFIG.custom_classes.values():
                class_key = f"{class_name}_count"
                static_key = f"static_{class_name}_count"

                class_counters[class_name] += statistics.get(class_key, 0)
                if CONFIG.static_car_enabled:
                    static_counters[class_name] += statistics.get(static_key, 0)

            # Merge unique track IDs across all videos
            unique_tracks = statistics.get("unique_tracks", {})
            if isinstance(unique_tracks, dict):
                for class_name in CONFIG.custom_classes.values():
                    track_counters[class_name] += len(unique_tracks.get(class_name, set()))

        # Calculate derived metrics
        total_tracks = sum(track_counters.values())
        avg_confidence = confidence_sum / total_detections if total_detections > 0 else 0
        avg_duration = total_duration / total_videos if total_videos > 0 else 0
        avg_processing_fps = (total_frames / total_processing_time) if total_processing_time > 0 else 0

        # Build dynamic overview text
        overview_text = f"""
        <b>Total Videos:</b> {total_videos}<br/><br/>
        <b>Total Duration:</b> {self._format_duration(total_duration)} ({total_duration:.1f} seconds)<br/><br/>
        <b>Average Video Duration:</b> {self._format_duration(avg_duration)}<br/><br/>
        <b>Total Frames:</b> {total_frames:,}<br/><br/>
        <b>Total Detections:</b> {total_detections:,}<br/><br/>
        <b>Total Tracks:</b> {total_tracks:,}<br/><br/>
        <b>Average Confidence:</b> {avg_confidence:.3f}<br/><br/>

        <b>Object Counts:</b><br/>
        """

        # Add only classes with detections or tracks
        found_objects = False
        for class_name in CONFIG.custom_classes.values():
            detection_count = class_counters[class_name]
            track_count = track_counters[class_name]
            if detection_count > 0 or track_count > 0:
                overview_text += f"- {class_name.title()} (detections: {detection_count:,}, tracks: {track_count:,})<br/>"
                found_objects = True

        if not found_objects:
            overview_text += "- No objects detected<br/>"

        # Add static object counts if enabled and found
        if CONFIG.static_car_enabled:
            static_found = any(static_counters[class_name] > 0 for class_name in CONFIG.custom_classes.values())
            if static_found:
                overview_text += "<br/><b>Static Objects:</b><br/>"
                for class_name in CONFIG.custom_classes.values():
                    static_count = static_counters[class_name]
                    if static_count > 0:
                        overview_text += f"- Static {class_name.title()}: {static_count:,}<br/>"

        overview_text += f"""
        <br/><b>Processing Performance:</b><br/>
        - Average Inference Speed: {avg_processing_fps:.2f} FPS<br/>
        - Total Processing Time: {self._format_duration(total_processing_time)}<br/>
        """

        content.append(Paragraph(overview_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.6 * inch))
        return content

    def _create_video_section(self, video_id: str, video_info: Dict, statistics: Dict) -> List:
        """
        Generate detailed statistics section for individual video.

        Handles video metadata extraction with fallbacks for missing data.
        """
        # Truncate very long video IDs for display
        display_id = video_id[:50] + "..." if len(video_id) > 53 else video_id

        logger.debug(f"Video {video_id}: video_info keys = {list(video_info.keys())}")
        logger.debug(f"Video {video_id}: statistics keys = {list(statistics.keys())}")

        content = [
            Paragraph(f"Video: {display_id}", self.styles['VideoTitle']),
            Spacer(1, 0.1 * inch)
        ]

        # Extract video properties with fallback values
        duration = video_info.get("duration", 0) or video_info.get("total_duration", 0)
        frames = statistics.get("total_frames", 0) or statistics.get("processed_frames", 0)
        fps = video_info.get("fps", 0) or statistics.get("fps", 0)
        resolution = video_info.get("resolution", "N/A")

        # Construct resolution from width/height if not directly available
        if resolution == "N/A":
            width = video_info.get("width", 0)
            height = video_info.get("height", 0)
            if width and height:
                resolution = f"{width}x{height}"

        # Extract detection statistics dynamically
        total_detections = statistics.get("total_detections", 0)
        avg_confidence = statistics.get("avg_confidence", 0)

        # Calculate unique track counts with type safety
        unique_tracks = statistics.get("unique_tracks", {})
        track_counts = {}
        detection_counts = {}
        static_counts = {}

        for class_name in CONFIG.custom_classes.values():
            # Detection counts
            detection_key = f"{class_name}_count"
            detection_counts[class_name] = statistics.get(detection_key, 0)

            # Track counts
            if isinstance(unique_tracks, dict):
                track_counts[class_name] = len(unique_tracks.get(class_name, set()))
            else:
                track_counts[class_name] = 0

            # Static object counts
            if CONFIG.static_car_enabled:
                static_key = f"static_{class_name}_count"
                static_counts[class_name] = statistics.get(static_key, 0)

        # Calculate total tracks
        total_tracks = sum(track_counts.values())

        # Calculate actual processing performance metrics
        processing_time = statistics.get("processing_time", 0)
        actual_inference_fps = frames / processing_time if processing_time > 0 else 0

        video_text = f"""
        <b>Video Properties:</b><br/>
        • Duration: {self._format_duration(duration)} ({duration:.1f} seconds)<br/>
        • Frame Count: {frames:,}<br/>
        • Video FPS: {fps:.1f}<br/>
        • Resolution: {resolution}<br/><br/>

        <b>Results:</b><br/>
        • Total Detections: {total_detections:,}<br/>
        • Total Tracks: {total_tracks:,}<br/>
        • Average Confidence: {avg_confidence:.3f}<br/><br/>
        """

        # Add only classes with detections or tracks
        found_objects = False
        for class_name in CONFIG.custom_classes.values():
            detection_count = detection_counts[class_name]
            track_count = track_counts[class_name]
            if detection_count > 0 or track_count > 0:
                video_text += f"- {class_name.title()} (detections: {detection_count:,}, tracks: {track_count:,})<br/>"
                found_objects = True

        if not found_objects:
            video_text += "- No objects detected<br/>"

        video_text += "<br/>"

        # Add static object counts if enabled
        if CONFIG.static_car_enabled:
            static_found = any(static_counts[class_name] > 0 for class_name in CONFIG.custom_classes.values())
            if static_found:
                video_text += "<b>Static Objects:</b><br/>"
                for class_name in CONFIG.custom_classes.values():
                    static_count = static_counts[class_name]
                    if static_count > 0:
                        video_text += f"- Static {class_name.title()}: {static_count:,}<br/>"
            else:
                video_text += "<b>Static Objects:</b> None<br/>"
        else:
            video_text += "<b>Static Objects:</b> Disabled<br/>"

        video_text += f"""
        <br/><b>Processing Performance:</b><br/>
        • Inference Speed: {actual_inference_fps:.2f} FPS<br/>
        • Processing Time: {self._format_duration(processing_time)}<br/>
        """

        content.append(Paragraph(video_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.6 * inch))
        return content


    def _create_timestamp_section(self) -> List:
        """Add generation timestamp to report footer."""
        content = [
            Spacer(1, 2 * inch),  # Larger spacer to push timestamp to bottom
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['BodyText']),
        ]
        return content


    def _format_duration(self, seconds: float) -> str:
        """Convert seconds to human-readable duration format (hours, minutes, seconds)."""
        if seconds <= 0:
            return "0s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"