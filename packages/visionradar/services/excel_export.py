"""
VisionRadar Enterprise Excel Intelligence Export Engine (Phase 5).
Generates professional, multi-sheet .xlsx workbooks using openpyxl for both
Traffic Intelligence and Examination Intelligence domains.
Supports both 'final' (completed analysis) and 'snapshot' (live in-flight) export modes.
"""

import io
import datetime
import hashlib
from typing import Dict, Any, List, Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from sqlalchemy.orm import Session

from visionradar.models.entities import (
    ProcessingJob, Video, Calibration, Track, SpeedMeasurement, Violation, Evidence,
    ExaminationTrackRecord, ExaminationEventRecord, ExaminationZoneRecord,
    ResearchExperimentDB, BenchmarkExperimentDB
)

# ---------------------------------------------------------------------------
# Styling Tokens (Professional Corporate Palette)
# ---------------------------------------------------------------------------
NAVY_HEADER_FILL = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
INDIGO_ACCENT_FILL = PatternFill(start_color="4F46E5", end_color="4F46E5", fill_type="solid")
SUBHEADER_FILL = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
ZEBRA_FILL = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
ALERT_RED_FILL = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
ALERT_AMBER_FILL = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
SUCCESS_GREEN_FILL = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")

WHITE_BOLD_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
DARK_BOLD_FONT = Font(name="Calibri", size=11, bold=True, color="0F172A")
TITLE_FONT = Font(name="Calibri", size=16, bold=True, color="1E293B")
SUBTITLE_FONT = Font(name="Calibri", size=11, italic=True, color="64748B")
REGULAR_FONT = Font(name="Calibri", size=10, color="1E293B")
BOLD_REGULAR_FONT = Font(name="Calibri", size=10, bold=True, color="1E293B")
MUTED_FONT = Font(name="Calibri", size=9, color="64748B")

THIN_BORDER_SIDE = Side(border_style="thin", color="CBD5E1")
TABLE_BORDER = Border(left=THIN_BORDER_SIDE, right=THIN_BORDER_SIDE, top=THIN_BORDER_SIDE, bottom=THIN_BORDER_SIDE)
BOTTOM_DOUBLE_SIDE = Side(border_style="double", color="1E293B")
SUMMARY_BORDER = Border(bottom=BOTTOM_DOUBLE_SIDE, top=THIN_BORDER_SIDE)

ALIGN_LEFT = Alignment(horizontal="left", vertical="center")
ALIGN_CENTER = Alignment(horizontal="center", vertical="center")
ALIGN_RIGHT = Alignment(horizontal="right", vertical="center")
ALIGN_HEADER = Alignment(horizontal="center", vertical="center", wrap_text=True)

def utcnow_str() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def auto_fit_columns(ws, min_width=12, max_width=50, max_sample_rows=80):
    """Dynamically calculates column widths with padding, using bounded sampling for high performance."""
    merged_coord_set = set()
    for rng in ws.merged_cells.ranges:
        for r in range(rng.min_row, rng.max_row + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                merged_coord_set.add((r, c))

    limit_row = min(ws.max_row, max_sample_rows)
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        for row_idx in range(1, limit_row + 1):
            if (row_idx, col_idx) in merged_coord_set:
                continue
            cell = ws.cell(row=row_idx, column=col_idx)
            val = cell.value
            if val is not None:
                first_line = str(val).split("\n")[0]
                max_len = max(max_len, len(first_line))
        ws.column_dimensions[col_letter].width = max(min_width, min(max_len + 4, max_width))



class ExcelExportEngine:
    """
    Enterprise-grade Excel Export Engine for VisionRadar.
    Produces rich, formatted, multi-tab workbooks containing analytical summaries,
    per-track metrics, raw detections, speed forensics, exam proctoring events, and data dictionaries.
    """

    @staticmethod
    def generate_workbook(job_id: int, db: Session, mode: str = "final") -> io.BytesIO:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            raise ValueError(f"ProcessingJob #{job_id} not found in database.")

        video = db.query(Video).filter(Video.id == job.video_id).first()
        calibration = db.query(Calibration).filter(Calibration.id == job.calibration_id).first() if job.calibration_id else None
        if not calibration and video:
            calibration = db.query(Calibration).filter(Calibration.video_id == video.id).order_by(Calibration.version.desc()).first()

        config = job.config_json or {}
        telemetry = job.telemetry_json or {}
        domain = config.get("module", "traffic").lower()
        if "exam" in domain:
            domain = "examination"

        wb = openpyxl.Workbook()
        # Remove default empty sheet; we create named sheets deterministically
        wb.remove(wb.active)

        # 1. Executive Summary Sheet
        ExcelExportEngine._build_executive_summary(wb, job, video, calibration, domain, mode, db)

        # 2. Processing Metadata Sheet
        ExcelExportEngine._build_processing_metadata(wb, job, video, calibration, domain, mode)

        # 3. Domain Specific Sheets
        if domain == "traffic":
            ExcelExportEngine._build_traffic_unique_tracks(wb, job, db)
            ExcelExportEngine._build_traffic_events(wb, job, db)
            ExcelExportEngine._build_speed_analytics(wb, job, db)
            if calibration:
                ExcelExportEngine._build_calibration_metadata(wb, calibration, video)
        else:
            ExcelExportEngine._build_examination_tracks(wb, job, db)
            ExcelExportEngine._build_examination_events(wb, job, db)
            ExcelExportEngine._build_examination_zones(wb, video, db)

        # 4. Sampled Detection Records Sheet
        ExcelExportEngine._build_detection_records(wb, job, domain, db)

        # 5. Research & Experiments (if experiments exist)
        ExcelExportEngine._build_research_experiments(wb, video, db)

        # 6. Data Dictionary Sheet
        ExcelExportEngine._build_data_dictionary(wb, domain)

        # 7. Export Metadata Sheet
        ExcelExportEngine._build_export_metadata(wb, job, domain, mode)

        output_stream = io.BytesIO()
        wb.save(output_stream)
        output_stream.seek(0)
        return output_stream

    # -----------------------------------------------------------------------
    # Sheet Builders
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_executive_summary(wb, job, video, calibration, domain, mode, db: Session):
        ws = wb.create_sheet(title="Executive Summary")
        ws.views.sheetView[0].showGridLines = True

        # Header Title
        ws.merge_cells("A1:G1")
        ws["A1"] = "VISIONRADAR — UNIVERSAL VIDEO INTELLIGENCE PLATFORM"
        ws["A1"].font = TITLE_FONT
        ws["A1"].alignment = ALIGN_LEFT
        ws.row_dimensions[1].height = 28

        ws.merge_cells("A2:G2")
        domain_label = "TRAFFIC INTELLIGENCE & SPEED ESTIMATION" if domain == "traffic" else "EXAMINATION PROCTORING & BEHAVIORAL INTELLIGENCE"
        mode_label = "FINAL AUDIT EXPORT" if mode == "final" else "LIVE IN-FLIGHT SNAPSHOT"
        ws["A2"] = f"Executive Analytical Report • Domain: {domain_label} • Mode: {mode_label}"
        ws["A2"].font = SUBTITLE_FONT
        ws["A2"].alignment = ALIGN_LEFT
        ws.row_dimensions[2].height = 20

        # Provenance Card
        ws.append([]) # Row 3 empty
        ws.append(["SESSION PROVENANCE & IDENTIFICATION", "", "", "", "", "", ""])
        header_row = 4
        ws.merge_cells(f"A{header_row}:G{header_row}")
        ws[f"A{header_row}"].fill = NAVY_HEADER_FILL
        ws[f"A{header_row}"].font = WHITE_BOLD_FONT
        ws[f"A{header_row}"].alignment = ALIGN_LEFT
        ws.row_dimensions[header_row].height = 22

        prov_rows = [
            ("Job Identifier:", f"#{job.id}", "Processing Status:", job.status),
            ("Analysis Domain:", domain.upper(), "Pipeline Stage:", job.stage),
            ("Source Video:", video.filename if video else "N/A", "Video Duration / FPS:", f"{video.duration_sec:.1f}s @ {video.fps:.1f} FPS" if video else "N/A"),
            ("Video SHA-256:", video.sha256_hash if video else "N/A", "Resolution:", f"{video.width}x{video.height}" if video else "N/A"),
            ("Started At:", job.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if job.started_at else "N/A", "Completed At:", job.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if job.completed_at else "IN-PROGRESS"),
            ("Calibration Reference:", f"Calibration #{calibration.id} (v{calibration.version})" if calibration else "UNCALIBRATED (Uncalibrated Motion)", "Calibration RMSE:", f"{calibration.reprojection_rmse_m:.3f} m" if calibration else "N/A")
        ]

        for r in prov_rows:
            ws.append([r[0], r[1], "", r[2], r[3], "", ""])
            curr = ws.max_row
            ws[f"A{curr}"].font = DARK_BOLD_FONT
            ws[f"B{curr}"].font = REGULAR_FONT
            ws[f"D{curr}"].font = DARK_BOLD_FONT
            ws[f"E{curr}"].font = REGULAR_FONT
            ws.row_dimensions[curr].height = 19

        # Metrics KPI Section
        ws.append([])
        kpi_title_row = ws.max_row + 1
        ws.append(["KEY PERFORMANCE INDICATORS (EMPIRICALLY VERIFIED)", "", "", "", "", "", ""])
        ws.merge_cells(f"A{kpi_title_row}:G{kpi_title_row}")
        ws[f"A{kpi_title_row}"].fill = INDIGO_ACCENT_FILL
        ws[f"A{kpi_title_row}"].font = WHITE_BOLD_FONT
        ws[f"A{kpi_title_row}"].alignment = ALIGN_LEFT
        ws.row_dimensions[kpi_title_row].height = 22

        if domain == "traffic":
            tracks_count = db.query(Track).filter(Track.job_id == job.id).count()
            viols_count = db.query(Violation).filter(Violation.job_id == job.id).count()
            accepted_viols = db.query(Violation).filter(Violation.job_id == job.id, Violation.review_status == "ACCEPTED").count()
            pending_viols = db.query(Violation).filter(Violation.job_id == job.id, Violation.review_status == "PENDING").count()
            telemetry = job.telemetry_json or {}
            speed_stats = telemetry.get("speed_statistics", {})

            kpis = [
                ("Total Unique Vehicles Tracked", tracks_count, "units"),
                ("Confirmed Speed Violations", viols_count, "events"),
                ("Human-Verified Violations (Accepted)", accepted_viols, "events"),
                ("Pending Violations Under Review", pending_viols, "events"),
                ("Fleet Mean Speed", f"{speed_stats.get('mean_speed_kmh', 'N/A')}", "km/h" if "mean_speed_kmh" in speed_stats else ""),
                ("P85 Traffic Speed (85th Percentile)", f"{speed_stats.get('p85_speed_kmh', 'N/A')}", "km/h" if "p85_speed_kmh" in speed_stats else ""),
                ("Maximum Speed Observed", f"{speed_stats.get('max_speed_kmh', 'N/A')}", "km/h" if "max_speed_kmh" in speed_stats else ""),
                ("Average Processing Throughput", f"{telemetry.get('average_fps', 'N/A')}", "FPS" if "average_fps" in telemetry else "")
            ]
        else:
            person_tracks = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == job.id, ExaminationTrackRecord.entity_type == "person").all()
            events = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job.id).all()
            total_persons = len(person_tracks)
            students = sum(1 for t in person_tracks if t.assigned_role == "STUDENT")
            staff = sum(1 for t in person_tracks if t.assigned_role in ["INVIGILATOR", "EXAM_COORDINATOR", "AUTHORIZED_STAFF"])
            unknown_roles = sum(1 for t in person_tracks if not t.assigned_role or t.assigned_role == "UNKNOWN")
            red_events = sum(1 for e in events if (e.supporting_objects_json or {}).get("priority") == "RED")
            accepted_events = sum(1 for e in events if e.review_status == "ACCEPTED")
            pending_events = sum(1 for e in events if e.review_status == "PENDING")

            kpis = [
                ("Total Monitored Persons", total_persons, "people"),
                ("Confirmed Students Enrolled", students, "students"),
                ("Authorized Staff (Invigilators/Coordinators)", staff, "staff"),
                ("Unclassified / Unknown Roles", unknown_roles, "unassigned"),
                ("Candidate Behavioral Events", len(events), "observations"),
                ("High-Priority Candidate Alerts (RED)", red_events, "alerts"),
                ("Proctor Accepted Evidence Findings", accepted_events, "confirmed"),
                ("Pending Proctor Review Items", pending_events, "pending review")
            ]

        ws.append(["Metric Name", "Observed Value", "Unit / Semantics", "", "", "", ""])
        table_hdr = ws.max_row
        for col_idx in [1, 2, 3]:
            cell = ws.cell(row=table_hdr, column=col_idx)
            cell.fill = SUBHEADER_FILL
            cell.font = DARK_BOLD_FONT
            cell.border = TABLE_BORDER
        ws.row_dimensions[table_hdr].height = 20

        for k in kpis:
            ws.append([k[0], k[1], k[2], "", "", "", ""])
            curr = ws.max_row
            ws[f"A{curr}"].font = REGULAR_FONT
            ws[f"A{curr}"].border = TABLE_BORDER
            ws[f"B{curr}"].font = DARK_BOLD_FONT
            ws[f"B{curr}"].border = TABLE_BORDER
            ws[f"B{curr}"].alignment = ALIGN_RIGHT if isinstance(k[1], (int, float)) else ALIGN_LEFT
            ws[f"C{curr}"].font = MUTED_FONT
            ws[f"C{curr}"].border = TABLE_BORDER
            ws.row_dimensions[curr].height = 19

        # Ethical & Legal Governance Notice
        ws.append([])
        gov_row = ws.max_row + 1
        ws.append(["LEGAL & SCIENTIFIC INTEGRITY GOVERNANCE", "", "", "", "", "", ""])
        ws.merge_cells(f"A{gov_row}:G{gov_row}")
        ws[f"A{gov_row}"].fill = SUBHEADER_FILL
        ws[f"A{gov_row}"].font = DARK_BOLD_FONT
        ws.row_dimensions[gov_row].height = 20

        disclaimer_text = (
            "1. PROCTOR/OPERATOR IN THE LOOP: Automated candidate alerts reflect objective algorithmic observations and do NOT declare guilt, innocence, or legal liability.\n"
            "2. UNCALIBRATED MEASUREMENTS: Monocular speeds lacking validated homography matrices are classified UNCALIBRATED_MOTION and never generate automatic citations.\n"
            "3. OBJECT TAXONOMY DISCLOSURE: Pretrained COCO 80 models lack explicit calculator classes. Handheld rectangular electronic objects are classified as uncertain devices requiring human physical confirmation."
        )
        ws.append([disclaimer_text, "", "", "", "", "", ""])
        note_row = ws.max_row
        ws.merge_cells(f"A{note_row}:G{note_row}")
        ws[f"A{note_row}"].font = MUTED_FONT
        ws[f"A{note_row}"].alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[note_row].height = 48

        auto_fit_columns(ws)

    @staticmethod
    def _build_processing_metadata(wb, job, video, calibration, domain, mode):
        ws = wb.create_sheet(title="Processing Metadata")
        ws.views.sheetView[0].showGridLines = True
        telemetry = job.telemetry_json or {}
        config = job.config_json or {}
        model_meta = telemetry.get("model_metadata", {})

        headers = ["Parameter Name", "Value", "Units / Specification", "Provenance Notes"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        meta_rows = [
            ("VisionRadar Software Version", "2.1.0", "Release Tag", "Universal Video Intelligence Platform"),
            ("Analysis Engine Domain", domain.upper(), "Pipeline Route", "Modular Independent Inference Engine"),
            ("Export Mode", mode.upper(), "Snapshot vs Final", "Live snapshot retains in-progress frame markers"),
            ("Job Database ID", str(job.id), "Integer Key", "Canonical primary key in processing_jobs table"),
            ("Job Status", job.status, "State", "QUEUED | RUNNING | SUCCEEDED | FAILED | CANCELLED"),
            ("Processing Pipeline Stage", job.stage, "Stage", "Active worker execution stage"),
            ("Progress Percentage", f"{job.progress_pct:.1f}%", "Percentage", "Completed frame ratio"),
            ("Video Processing Stride", str(config.get("stride", telemetry.get("stride", 2))), "Frames", "Frame downsampling step (1=forensic, 2=balanced, 3=high_speed)"),
            ("Average Throughput FPS", str(telemetry.get("average_fps", "N/A")), "Frames / Sec", "Measured live pipeline execution rate"),
            ("Average Inference Latency", f"{telemetry.get('average_inference_ms', 'N/A')}", "ms / Frame", "Neural network forward pass execution time"),
            ("Processed Frames Count", str(telemetry.get("processed_frames", 0)), "Frames", "Number of video frames passed through detector"),
            ("Total Video Frames", str(telemetry.get("total_frames", 0)), "Frames", "Length of source video container"),
            ("Neural Network Model", model_meta.get("model_name", "YOLOX-Nano ONNX"), "Architecture", "COCO 80 pretrained object detection backbone"),
            ("Input Resolution", f"{model_meta.get('input_size', [416, 416])[0]}x{model_meta.get('input_size', [416, 416])[1]}", "Pixels (WxH)", "Preprocessed tensor input dimensions"),
            ("Confidence Detection Floor", str(model_meta.get("confidence_threshold", 0.20)), "Probability [0, 1]", "Calibrated detection confidence cutoff"),
            ("Non-Maximum Suppression IoU", str(model_meta.get("nms_threshold", 0.45)), "IoU [0, 1]", "Box deduplication overlap threshold"),
            ("Camera Calibration ID", str(calibration.id) if calibration else "None", "Foreign Key", "Bound road-plane homography calibration record"),
            ("Calibration Version", str(calibration.version) if calibration else "N/A", "Version Integer", "Monotonically incrementing calibration version")
        ]

        for r in meta_rows:
            ws.append(list(r))
            curr = ws.max_row
            ws[f"A{curr}"].font = DARK_BOLD_FONT
            ws[f"A{curr}"].border = TABLE_BORDER
            ws[f"B{curr}"].font = REGULAR_FONT
            ws[f"B{curr}"].border = TABLE_BORDER
            ws[f"C{curr}"].font = MUTED_FONT
            ws[f"C{curr}"].border = TABLE_BORDER
            ws[f"D{curr}"].font = REGULAR_FONT
            ws[f"D{curr}"].border = TABLE_BORDER
            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:D{ws.max_row}"
        auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # Traffic Sheets
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_traffic_unique_tracks(wb, job, db: Session):
        ws = wb.create_sheet(title="Unique Tracks")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Track ID", "Vehicle Class", "Detection Conf", "First Frame", "Last Frame", "Duration (Frames)", "Duration (s)", "Trajectory Points", "Max Speed (km/h)", "Violation Logged"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        tracks = db.query(Track).filter(Track.job_id == job.id).order_id_asc() if hasattr(Track, "order_id_asc") else db.query(Track).filter(Track.job_id == job.id).all()
        violations = {v.track_id: v for v in db.query(Violation).filter(Violation.job_id == job.id).all()}
        fps = (job.video.fps or 30.0) if job.video else 30.0

        for t in tracks:
            dur_frames = max(1, t.last_frame - t.first_frame + 1)
            dur_sec = round(dur_frames / fps, 2)
            traj = t.trajectory_json or []
            max_spd = "N/A"
            if t.speed_measurements:
                spds = [sm.smoothed_kmh for sm in t.speed_measurements if sm.smoothed_kmh is not None]
                if spds:
                    max_spd = round(max(spds), 1)

            has_viol = "YES (VIOLATION)" if t.track_id in violations else "NO"

            ws.append([
                t.track_id,
                t.vehicle_class,
                round(t.confidence, 3),
                t.first_frame,
                t.last_frame,
                dur_frames,
                dur_sec,
                len(traj),
                max_spd,
                has_viol
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER
            if has_viol.startswith("YES"):
                ws.cell(row=curr, column=10).fill = ALERT_RED_FILL
                ws.cell(row=curr, column=10).font = DARK_BOLD_FONT
            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_traffic_events(wb, job, db: Session):
        ws = wb.create_sheet(title="Traffic Events")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Violation ID", "Track ID", "Vehicle Class", "Timestamp (s)", "Frame Index", "Est. Speed (km/h)", "Speed Limit (km/h)", "Over Limit (km/h)", "Uncertainty (±km/h)", "Review Status", "Reviewer Notes"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        violations = db.query(Violation).filter(Violation.job_id == job.id).all()
        for v in violations:
            over_lim = round(max(0.0, v.estimated_speed_kmh - v.speed_limit_kmh), 1)
            ws.append([
                v.id,
                v.track_id,
                v.vehicle_class,
                round(v.timestamp, 2),
                v.frame_index,
                round(v.estimated_speed_kmh, 1),
                round(v.speed_limit_kmh, 1),
                over_lim,
                round(v.uncertainty_kmh, 1),
                v.review_status,
                v.reviewer_notes or ""
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER
            st_cell = ws.cell(row=curr, column=10)
            if v.review_status == "ACCEPTED":
                st_cell.fill = SUCCESS_GREEN_FILL
                st_cell.font = DARK_BOLD_FONT
            elif v.review_status == "REJECTED":
                st_cell.fill = ALERT_RED_FILL
            else:
                st_cell.fill = ALERT_AMBER_FILL
            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_speed_analytics(wb, job, db: Session):
        ws = wb.create_sheet(title="Speed Analytics")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Measurement ID", "Track ID", "Frame Index", "Timestamp (s)", "Instantaneous (km/h)", "Smoothed Speed (km/h)", "Uncertainty (±km/h)", "Confidence Low (km/h)", "Confidence High (km/h)"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        measurements = (
            db.query(SpeedMeasurement)
            .join(Track, SpeedMeasurement.track_id == Track.id)
            .filter(Track.job_id == job.id)
            .limit(10000)
            .all()
        )

        for sm in measurements:
            track_obj = sm.track
            trk_label = track_obj.track_id if track_obj else sm.track_id
            inst = round(sm.instantaneous_kmh, 1) if sm.instantaneous_kmh is not None else "N/A"
            smooth = round(sm.smoothed_kmh, 1) if sm.smoothed_kmh is not None else "N/A"
            unc = round(sm.uncertainty_kmh, 1) if sm.uncertainty_kmh is not None else "N/A"
            low = round(sm.confidence_low_kmh, 1) if sm.confidence_low_kmh is not None else "N/A"
            high = round(sm.confidence_high_kmh, 1) if sm.confidence_high_kmh is not None else "N/A"

            ws.append([
                sm.id,
                trk_label,
                sm.frame_index,
                round(sm.timestamp, 2),
                inst,
                smooth,
                unc,
                low,
                high
            ])

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_calibration_metadata(wb, calibration: Calibration, video: Optional[Video]):
        ws = wb.create_sheet(title="Calibration Metadata")
        ws.views.sheetView[0].showGridLines = True

        ws.merge_cells("A1:E1")
        ws["A1"] = f"CAMERA ROAD-PLANE CALIBRATION SPECIFICATION (ID #{calibration.id} · v{calibration.version})"
        ws["A1"].font = TITLE_FONT
        ws.row_dimensions[1].height = 24

        ws.append([])
        ws.append(["Parameter", "Value", "Units", "Integrity Evaluation"])
        hdr_row = 3
        for i in range(1, 5):
            c = ws.cell(row=hdr_row, column=i)
            c.fill = NAVY_HEADER_FILL
            c.font = WHITE_BOLD_FONT
            c.alignment = ALIGN_HEADER

        cal_rows = [
            ("Reprojection RMSE", f"{calibration.reprojection_rmse_m:.4f}", "meters", "Pass (< 0.50m)" if calibration.reprojection_rmse_m < 0.50 else "High Error"),
            ("Camera Height", f"{calibration.camera_height_m:.2f}", "meters", "Physically plausible mounting height"),
            ("Camera Pitch Angle", f"{calibration.pitch_deg:.2f}", "degrees", "Downward optical tilt"),
            ("Created At", calibration.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if calibration.created_at else "N/A", "Timestamp", "Monotonically versioned")
        ]
        for r in cal_rows:
            ws.append(list(r))
            curr = ws.max_row
            for c_idx in range(1, 5):
                ws.cell(row=curr, column=c_idx).border = TABLE_BORDER
                ws.cell(row=curr, column=c_idx).font = REGULAR_FONT

        # Homography Matrix display
        ws.append([])
        ws.append(["HOMOGRAPHY MATRIX H (3x3 Road Plane Transform)", "", "", ""])
        h_row = ws.max_row
        ws.merge_cells(f"A{h_row}:D{h_row}")
        ws[f"A{h_row}"].fill = SUBHEADER_FILL
        ws[f"A{h_row}"].font = DARK_BOLD_FONT

        H = calibration.h_matrix_json or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for row_idx, h_r in enumerate(H):
            ws.append([f"H[{row_idx},0]: {h_r[0]:.6e}", f"H[{row_idx},1]: {h_r[1]:.6e}", f"H[{row_idx},2]: {h_r[2]:.6e}", ""])
            curr = ws.max_row
            for c_idx in range(1, 4):
                ws.cell(row=curr, column=c_idx).border = TABLE_BORDER
                ws.cell(row=curr, column=c_idx).font = REGULAR_FONT

        auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # Examination Sheets
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_examination_tracks(wb, job, db: Session):
        ws = wb.create_sheet(title="Monitored Persons & Tracks")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Track ID", "Entity Type", "Assigned Role", "Role Source", "First Frame", "Last Frame", "Duration (Frames)", "Duration (s)", "Assigned Zone", "Trajectory Observations", "Continuity Ratio"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        tracks = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == job.id).all()
        fps = (job.video.fps or 30.0) if job.video else 30.0

        for t in tracks:
            dur_frames = max(1, t.last_frame - t.first_frame + 1)
            dur_s = round(dur_frames / fps, 2)
            hist = t.trajectory_json or []
            coverage_pct = f"{round(min(100.0, (len(hist) / dur_frames) * 100.0), 1)}%"

            ws.append([
                t.track_id,
                t.entity_type,
                t.assigned_role or "UNKNOWN",
                t.role_source or "DEFAULT",
                t.first_frame,
                t.last_frame,
                dur_frames,
                dur_s,
                t.assigned_zone or "UNASSIGNED",
                len(hist),
                coverage_pct
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER
            role_cell = ws.cell(row=curr, column=3)
            if t.assigned_role in ["INVIGILATOR", "EXAM_COORDINATOR", "AUTHORIZED_STAFF"]:
                role_cell.fill = SUCCESS_GREEN_FILL
                role_cell.font = DARK_BOLD_FONT
            elif t.assigned_role == "STUDENT":
                role_cell.fill = SUBHEADER_FILL
            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_examination_events(wb, job, db: Session):
        ws = wb.create_sheet(title="Examination Events")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Event ID", "Candidate Event Type", "Priority", "Evidence Strength", "Track ID", "Assigned Role", "Start Time (s)", "End Time (s)", "Duration (s)", "Confidence", "Factual Description", "Proctor Review", "Reviewer Notes"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        events = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job.id).all()
        for e in events:
            details = e.supporting_objects_json if isinstance(e.supporting_objects_json, dict) else {}
            priority = details.get("priority", "AMBER")
            strength = details.get("evidence_strength", "MEDIUM")
            dur = round(max(0.1, e.end_time - e.start_time), 2)

            ws.append([
                e.event_id,
                e.event_type,
                priority,
                strength,
                f"#{e.person_track_id}" if e.person_track_id else "N/A",
                e.assigned_role or "UNKNOWN",
                round(e.start_time, 2),
                round(e.end_time, 2),
                dur,
                round(e.confidence, 3),
                e.description,
                e.review_status,
                e.reviewer_notes or ""
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER

            p_cell = ws.cell(row=curr, column=3)
            if priority == "RED":
                p_cell.fill = ALERT_RED_FILL
                p_cell.font = DARK_BOLD_FONT
            elif priority == "AMBER":
                p_cell.fill = ALERT_AMBER_FILL

            rev_cell = ws.cell(row=curr, column=12)
            if e.review_status == "ACCEPTED":
                rev_cell.fill = SUCCESS_GREEN_FILL
                rev_cell.font = DARK_BOLD_FONT
            elif e.review_status == "REJECTED":
                rev_cell.fill = ALERT_RED_FILL
            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_examination_zones(wb, video: Optional[Video], db: Session):
        if not video:
            return
        zones = db.query(ExaminationZoneRecord).filter(ExaminationZoneRecord.video_id == video.id).all()
        if not zones:
            return

        ws = wb.create_sheet(title="Seating Zones")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Zone ID", "Descriptive Name", "Zone Type", "Polygon Vertices Count", "Polygon JSON Coordinates"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        for z in zones:
            poly = z.polygon_json or []
            ws.append([
                z.zone_id,
                z.name,
                z.zone_type,
                len(poly),
                str(poly)
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER
            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # Detections Sheet
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_detection_records(wb, job, domain, db: Session):
        ws = wb.create_sheet(title="Detection Records")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Frame Index", "Timestamp (s)", "Track ID", "Entity / Class", "Confidence", "BBox X1", "BBox Y1", "BBox X2", "BBox Y2", "BBox Width", "BBox Height"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        fps = (job.video.fps or 30.0) if job.video else 30.0

        # Sample detections from persisted trajectories (capped at 5,000 for excel responsiveness)
        sampled_rows = []
        if domain == "traffic":
            tracks = db.query(Track).filter(Track.job_id == job.id).all()
            for t in tracks:
                for h in (t.trajectory_json or [])[:200]:
                    f_idx = h.get("frame", 0)
                    bbox = h.get("bbox", [0, 0, 0, 0])
                    w = round(bbox[2] - bbox[0], 1)
                    h_val = round(bbox[3] - bbox[1], 1)
                    sampled_rows.append((
                        f_idx, round(f_idx / fps, 2), t.track_id, t.vehicle_class, round(t.confidence, 3),
                        round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1), w, h_val
                    ))
        else:
            tracks = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == job.id).all()
            for t in tracks:
                for h in (t.trajectory_json or [])[:200]:
                    f_idx = h.get("frame", 0)
                    bbox = h.get("bbox", [0, 0, 0, 0])
                    w = round(bbox[2] - bbox[0], 1)
                    h_val = round(bbox[3] - bbox[1], 1)
                    sampled_rows.append((
                        f_idx, round(f_idx / fps, 2), t.track_id, t.entity_type, round(h.get("conf", 0.90), 3),
                        round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1), w, h_val
                    ))

        # Sort chronologically
        sampled_rows.sort(key=lambda x: (x[0], x[2]))
        for r in sampled_rows[:3000]:
            ws.append(list(r))

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # Research & Experiments Sheet
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_research_experiments(wb, video: Optional[Video], db: Session):
        exps = db.query(ResearchExperimentDB).limit(25).all()
        if not exps:
            return

        ws = wb.create_sheet(title="Research & Experiments")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Experiment ID", "Name", "Dataset", "Model", "Pipeline Ver", "Protocol", "Result Status", "Ablations Count", "Failures Count", "Completed At"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        for e in exps:
            ws.append([
                e.experiment_id,
                e.name,
                e.dataset_name,
                e.model_id,
                e.pipeline_version,
                e.evaluation_protocol,
                e.result_status,
                len(e.ablations or []),
                len(e.failure_cases or []),
                e.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if e.completed_at else "N/A"
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER
            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # Data Dictionary Sheet
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_data_dictionary(wb, domain: str):
        ws = wb.create_sheet(title="Data Dictionary")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Target Sheet", "Field / Column", "Data Type", "Units", "Operational Meaning & Semantics", "Engineering Caveats & Safe Interpretation"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        dict_rows = [
            ("All Sheets", "Job ID", "Integer", "None", "Persistent unique identifier of processing task in processing_jobs table", "Stable cross-module identifier"),
            ("All Sheets", "Track ID", "Integer", "None", "Visual continuity identifier maintained by tracking filter", "Temporary tracking token; not a biometric identity or license plate"),
            ("Executive Summary", "Calibration Status", "String", "None", "Indicates whether homography road-plane projection was verified", "Uncalibrated cameras cannot support certified speed measurements"),
            ("Unique Tracks", "Duration (Frames)", "Integer", "Frames", "Total span between first and last observed frame", "Subject to visual occlusions"),
            ("Unique Tracks", "Continuity Ratio", "Percentage", "%", "Ratio of frames with active detections over total track duration", "Dips below 80% indicate frequent visual occlusions"),
            ("Detection Records", "Confidence", "Float", "[0.0 - 1.0]", "Raw object detector output score", "Detector score reflects visual match to training distribution, not truth"),
            ("Detection Records", "BBox X1/Y1/X2/Y2", "Float", "Pixels", "Bounding box coordinates in video pixel space", "Relative to video native width and height")
        ]

        if domain == "traffic":
            dict_rows.extend([
                ("Traffic Events", "Est. Speed", "Float", "km/h", "Homography-projected road-plane vehicle speed estimate", "Certified only when calibration is valid and error is bounded"),
                ("Traffic Events", "Speed Limit", "Float", "km/h", "Configured legal speed limit for enforcement segment", "Derived from user configuration or road signage"),
                ("Traffic Events", "Uncertainty (±)", "Float", "km/h", "Combined error-decomposed interval accounting for pitch, homography, and tracking jitter", "Never zero; reflects fundamental optical precision"),
                ("Traffic Events", "Review Status", "String", "None", "PENDING | ACCEPTED | REJECTED human sign-off status", "Automated violations require human proctor/operator verification"),
                ("Speed Analytics", "Smoothed Speed", "Float", "km/h", "Kalman-smoothed trajectory speed estimation", "Attenuates frame-to-frame bounding box discretization jitter")
            ])
        else:
            dict_rows.extend([
                ("Monitored Persons", "Assigned Role", "String", "None", "STUDENT | INVIGILATOR | EXAM_COORDINATOR | AUTHORIZED_STAFF | UNKNOWN", "Assigned via config, zone, or proctor review. Never inferred from appearance alone"),
                ("Monitored Persons", "Role Source", "String", "None", "CONFIG | ZONE | HUMAN_REVIEW | DEFAULT", "Indicates provenance of assigned role"),
                ("Examination Events", "Priority", "String", "None", "RED | AMBER | INFO alert severity hierarchy", "RED priority indicates candidate requiring immediate proctor inspection"),
                ("Examination Events", "Evidence Strength", "String", "None", "HIGH | MEDIUM | LOW heuristic robustness", "Synthesizes duration, object presence, and visibility stability"),
                ("Examination Events", "Candidate Event Type", "String", "None", "Neutral behavioral observation category", "Does NOT establish cheating or academic dishonesty without proctor corroboration")
            ])

        for r in dict_rows:
            ws.append(list(r))
            curr = ws.max_row
            ws[f"A{curr}"].font = DARK_BOLD_FONT
            ws[f"A{curr}"].border = TABLE_BORDER
            ws[f"B{curr}"].font = DARK_BOLD_FONT
            ws[f"B{curr}"].border = TABLE_BORDER
            ws[f"C{curr}"].font = REGULAR_FONT
            ws[f"C{curr}"].border = TABLE_BORDER
            ws[f"D{curr}"].font = MUTED_FONT
            ws[f"D{curr}"].border = TABLE_BORDER
            ws[f"E{curr}"].font = REGULAR_FONT
            ws[f"E{curr}"].border = TABLE_BORDER
            ws[f"F{curr}"].font = REGULAR_FONT
            ws[f"F{curr}"].border = TABLE_BORDER
            ws.row_dimensions[curr].height = 20

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:F{ws.max_row}"
        auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # Export Metadata Sheet
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_export_metadata(wb, job, domain: str, mode: str):
        ws = wb.create_sheet(title="Export Metadata")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Metadata Attribute", "Value", "Specification"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, 4):
            c = ws.cell(row=1, column=i)
            c.fill = NAVY_HEADER_FILL
            c.font = WHITE_BOLD_FONT
            c.alignment = ALIGN_HEADER

        timestamp_str = utcnow_str()
        seal_raw = f"{job.id}-{domain}-{mode}-{timestamp_str}"
        seal = hashlib.sha256(seal_raw.encode("utf-8")).hexdigest()

        rows = [
            ("Export Generation Timestamp", timestamp_str, "ISO 8601 UTC Timestamp"),
            ("Export Engine Version", "VisionRadar Excel Intelligence Engine v2.1.0", "openpyxl 3.1.5 based generator"),
            ("Target Job ID", f"#{job.id}", "Database primary key"),
            ("Export Mode", mode.upper(), "FINAL EXPORT or LIVE IN-FLIGHT SNAPSHOT"),
            ("Domain Route", domain.upper(), "Domain-specific sheets included"),
            ("Cryptographic Export Seal", f"VR-XLSX-SEAL-{seal[:32].upper()}", "SHA-256 seal for tamper detection"),
            ("Non-Blocking Execution", "Guaranteed", "Export generated independently of video processing loop"),
            ("Export Compliance Standard", "Strict Non-Accusation & Non-Fabrication Protocol", "No empty metrics fabricated as zero")
        ]

        for r in rows:
            ws.append(list(r))
            curr = ws.max_row
            ws[f"A{curr}"].font = DARK_BOLD_FONT
            ws[f"A{curr}"].border = TABLE_BORDER
            ws[f"B{curr}"].font = REGULAR_FONT
            ws[f"B{curr}"].border = TABLE_BORDER
            ws[f"C{curr}"].font = MUTED_FONT
            ws[f"C{curr}"].border = TABLE_BORDER
            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        auto_fit_columns(ws)
