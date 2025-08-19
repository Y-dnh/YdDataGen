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
    """
    Generates a single, consolidated PDF report with configuration overview,
    dataset summary, and detailed per-video analysis.
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Setup custom styles for the report."""
        font_name = 'Helvetica'  # Arial not available by default, Helvetica is similar
        font_name_bold = 'Helvetica-Bold'

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['h2'],
            fontSize=16,
            spaceBefore=30,
            spaceAfter=20,
            textColor=colors.black,
            fontName=font_name_bold
        ))

        self.styles.add(ParagraphStyle(
            name='VideoTitle',
            parent=self.styles['h3'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.black,
            fontName=font_name_bold
        ))

        # Update body text style with more spacing
        body_style = self.styles['BodyText']
        body_style.fontName = font_name
        body_style.fontSize = 11
        body_style.spaceBefore = 8
        body_style.spaceAfter = 12
        body_style.textColor = colors.black
        body_style.leading = 16  # Increased line spacing

    def generate_consolidated_report(self, all_video_results: Dict, video_info_dict: Dict = None) -> Path:
        """Generate the final consolidated report for all videos."""
        logger.info("Generating consolidated PDF report...")

        report_path = CONFIG.paths.report_path
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72
        )

        content = []

        # Page 1: Configuration
        content.extend(self._create_configuration_section())
        content.append(PageBreak())  # Force new page after config

        # Page 2: Dataset Overview
        content.extend(self._create_dataset_overview(all_video_results, video_info_dict))

        # Individual video sections
        for video_id, results in all_video_results.items():
            content.append(PageBreak())
            # Try to get video info from separate dict first, then from results
            if video_info_dict and video_id in video_info_dict:
                video_info = video_info_dict[video_id]
            else:
                video_info = results.get("video_info", {})
            statistics = results.get("statistics", {})
            content.extend(self._create_video_section(video_id, video_info, statistics))

        # Add generation timestamp at the end
        content.extend(self._create_timestamp_section())

        doc.build(content)
        logger.info(f"Consolidated report generated: {report_path}")
        return report_path

    def _create_timestamp_section(self) -> List:
        """Create timestamp section at the end of the document."""
        content = [
            Spacer(1, 0.5 * inch),
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['BodyText']),
        ]
        return content

    def _create_configuration_section(self) -> List:
        """Create configuration overview section."""
        content = [
            Paragraph("Configuration Overview", self.styles['SectionHeader']),
            Spacer(1, 0.2 * inch)
        ]

        config_text = f"""
        <b>YOLO Model:</b> {CONFIG.yolo_model_path}<br/>
        <b>Sam Model:</b> {CONFIG.sam_model_path}<br/>
        <b>Tracker:</b> {CONFIG.tracker_type}<br/>
        <b>Device:</b> {CONFIG.device}<br/>
        <b>Inference Resolution:</b> {CONFIG.inference_resolution[0]}x{CONFIG.inference_resolution[1]}<br/>
        <b>Half Precision:</b> {'Enabled' if CONFIG.half_precision else 'Disabled'}<br/>

        <b>Detection Parameters:</b><br/>
        • YOLO Confidence: {CONFIG.yolo_confidence}<br/>
        • YOLO IoU: {CONFIG.yolo_iou}<br/>
        • Max Detections: {CONFIG.yolo_max_det}<br/>
        • Min Tracking Confidence: {CONFIG.min_confidence_for_tracking}<br/><br/>

        <b>Tracking Settings:</b><br/>
        • Tracker Type: {CONFIG.tracker_type}<br/>
        • Track High Thresh: {CONFIG.track_high_thresh}<br/>
        • Track Low Thresh: {CONFIG.track_low_thresh}<br/>
        • Match Thresh: {CONFIG.match_thresh}<br/><br/>

        <b>SAM Segmentation:</b><br/>
        • SAM Enabled: {'Yes' if CONFIG.sam_enabled else 'No'}<br/>
        """

        if CONFIG.sam_enabled:
            config_text += f"""
        • SAM Model: {CONFIG.sam_model_path}<br/>
        • SAM Confidence: {CONFIG.sam_confidence}<br/>
        • SAM IoU: {CONFIG.sam_iou}<br/>
            """
        else:
            config_text += "<br/>"

        config_text += f"""
        <b>Static Car Detection:</b><br/>
        • Enabled: {'Yes' if CONFIG.static_car_enabled else 'No'}<br/>
        """

        if CONFIG.static_car_enabled:
            config_text += f"""
        • Movement Threshold: {CONFIG.movement_threshold} pixels<br/>
            """
        else:
            config_text += "<br/>"

        config_text += f"""
        <b>Polygon Settings:</b><br/>
        • Max Points: {CONFIG.max_points}<br/>
        • Simplify Tolerance: {CONFIG.simplify_tolerance}<br/>
        • Min Area: {CONFIG.min_area}<br/>
        • Smoothing: {'Enabled' if CONFIG.smoothing else 'Disabled'}<br/>
        • Fill Holes: {'Enabled' if CONFIG.fill_holes else 'Disabled'}<br/>
        """

        content.append(Paragraph(config_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.6 * inch))  # Larger spacing before page break
        return content


    def _create_dataset_overview(self, all_video_results: Dict, video_info_dict: Dict = None) -> List:
        """Create dataset overview section."""
        content = [
            Paragraph("Dataset Overview", self.styles['SectionHeader']),
            Spacer(1, 0.2 * inch)
        ]

        # Calculate totals
        total_videos = len(all_video_results)
        total_duration = 0
        total_frames = 0
        total_people = 0
        total_cars = 0
        total_pets = 0
        total_static_cars = 0
        total_detections = 0
        confidence_sum = 0
        total_people_tracks = set()
        total_car_tracks = set()
        total_pet_tracks = set()
        total_processing_time = 0

        for video_id, results in all_video_results.items():
            # Try to get video info from separate dict first, then from results
            if video_info_dict and video_id in video_info_dict:
                video_info = video_info_dict[video_id]
            else:
                video_info = results.get("video_info", {})

            statistics = results.get("statistics", {})

            total_duration += video_info.get("duration", 0)
            total_frames += statistics.get("total_frames", 0)
            total_people += statistics.get("people_count", 0)
            total_cars += statistics.get("cars_count", 0)
            total_pets += statistics.get("pets_count", 0)
            total_static_cars += statistics.get("static_cars_count", 0)
            total_detections += statistics.get("total_detections", 0)
            confidence_sum += statistics.get("avg_confidence", 0) * statistics.get("total_detections", 0)
            total_processing_time += statistics.get("processing_time", 0)

            # Add unique tracks
            unique_tracks = statistics.get("unique_tracks", {})
            if isinstance(unique_tracks, dict):
                total_people_tracks.update(unique_tracks.get("person", set()))
                total_car_tracks.update(unique_tracks.get("car", set()))
                total_pet_tracks.update(unique_tracks.get("pet", set()))

        # Calculate aggregated statistics
        total_tracks = len(total_people_tracks) + len(total_car_tracks) + len(total_pet_tracks)
        avg_confidence = confidence_sum / total_detections if total_detections > 0 else 0
        avg_duration = total_duration / total_videos if total_videos > 0 else 0
        avg_processing_fps = (total_frames / total_processing_time) if total_processing_time > 0 else 0

        overview_text = f"""
        <b>Total Videos:</b> {total_videos}<br/><br/>
        <b>Total Duration:</b> {self._format_duration(total_duration)} ({total_duration:.1f} seconds)<br/><br/>
        <b>Average Video Duration:</b> {self._format_duration(avg_duration)}<br/><br/>
        <b>Total Frames:</b> {total_frames:,}<br/><br/>
        <b>Total Detections:</b> {total_detections:,}<br/><br/>
        <b>Average Confidence:</b> {avg_confidence:.3f}<br/><br/>

        <b>Object Counts:</b><br/>
        - People (detections: {total_people:,}, tracks: {len(total_people_tracks):,})<br/>
        - Cars (detections: {total_cars:,}, tracks: {len(total_car_tracks):,})<br/>
        - Pets (detections: {total_pets:,}, tracks: {len(total_pet_tracks):,})<br/>
        - Static Cars: {total_static_cars:,}<br/>
        - <b>Total Tracks: {total_tracks:,}</b><br/><br/>

        <b>Processing Performance:</b><br/>
        - Average Inference Speed: {avg_processing_fps:.2f} FPS<br/>
        - Total Processing Time: {self._format_duration(total_processing_time)}<br/>
        """

        content.append(Paragraph(overview_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.6 * inch))  # Larger spacing before page break
        return content


    def _create_video_section(self, video_id: str, video_info: Dict, statistics: Dict) -> List:
        """Create detailed section for individual video."""
        display_id = video_id[:50] + "..." if len(video_id) > 53 else video_id

        # Debug logging
        logger.debug(f"Video {video_id}: video_info keys = {list(video_info.keys())}")
        logger.debug(f"Video {video_id}: statistics keys = {list(statistics.keys())}")

        content = [
            Paragraph(f"Video: {display_id}", self.styles['VideoTitle']),
            Spacer(1, 0.1 * inch)
        ]

        # Video properties - try different field names
        duration = video_info.get("duration", 0) or video_info.get("total_duration", 0)
        frames = statistics.get("total_frames", 0) or statistics.get("processed_frames", 0)
        fps = video_info.get("fps", 0) or statistics.get("fps", 0)
        resolution = video_info.get("resolution", "N/A")

        # If no resolution in video_info, try to construct from width/height
        if resolution == "N/A":
            width = video_info.get("width", 0)
            height = video_info.get("height", 0)
            if width and height:
                resolution = f"{width}x{height}"

        # Detection results
        people_count = statistics.get("people_count", 0)
        cars_count = statistics.get("cars_count", 0)
        pets_count = statistics.get("pets_count", 0)
        static_cars = statistics.get("static_cars_count", 0)
        total_detections = statistics.get("total_detections", 0)
        avg_confidence = statistics.get("avg_confidence", 0)

        # Get unique track counts
        unique_tracks = statistics.get("unique_tracks", {})
        people_tracks = len(unique_tracks.get("person", set())) if isinstance(unique_tracks, dict) else 0
        car_tracks = len(unique_tracks.get("car", set())) if isinstance(unique_tracks, dict) else 0
        pet_tracks = len(unique_tracks.get("pet", set())) if isinstance(unique_tracks, dict) else 0

        # Calculate actual processing performance
        processing_time = statistics.get("processing_time", 0)
        actual_inference_fps = frames / processing_time if processing_time > 0 else 0

        video_text = f"""
        <b>Video Properties:</b><br/>
        • Duration: {self._format_duration(duration)} ({duration:.1f} seconds)<br/>
        • Frame Count: {frames:,}<br/>
        • Video FPS: {fps:.1f}<br/>
        • Resolution: {resolution}<br/><br/>

        <b>Detection Results:</b><br/>
        - People (detections: {people_count:,}, tracks: {people_tracks:,})<br/>
        - Cars (detections: {cars_count:,}, tracks: {car_tracks:,})<br/>
        - Pets (detections: {pets_count:,}, tracks: {pet_tracks:,})<br/>
        - Static Cars: {static_cars:,}<br/>
        - Total Detections: {total_detections:,}<br/>
        - Average Confidence: {avg_confidence:.3f}<br/><br/>

        <b>Processing Performance:</b><br/>
        • Inference Speed: {actual_inference_fps:.2f} FPS<br/>
        • Processing Time: {self._format_duration(processing_time)}<br/>
        """

        content.append(Paragraph(video_text, self.styles['BodyText']))
        content.append(Spacer(1, 0.6 * inch))  # Larger spacing before page break
        return content


    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable format."""
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