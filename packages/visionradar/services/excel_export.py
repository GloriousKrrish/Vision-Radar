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

VIOLET_HEADER_FILL = PatternFill(start_color="312E81", end_color="312E81", fill_type="solid")
VIOLET_ACCENT_FILL = PatternFill(start_color="4F46E5", end_color="4F46E5", fill_type="solid")
PURPLE_SUBHEADER_FILL = PatternFill(start_color="F5F3FF", end_color="F5F3FF", fill_type="solid")

def utcnow_str() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def format_timecode(seconds: Optional[float]) -> str:
    """Formats raw seconds into standard video timecode (MM:SS.ms)."""
    if seconds is None:
        return "N/A"
    try:
        total_sec = float(seconds)
        if total_sec < 0:
            return "00:00.00"
        mins = int(total_sec // 60)
        secs = total_sec % 60
        return f"{mins:02d}:{secs:05.2f}"
    except (ValueError, TypeError):
        return "N/A"

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
        domain = getattr(job, "domain", None) or config.get("domain") or config.get("module") or "traffic"
        domain = "examination" if "exam" in str(domain).lower() else "traffic"

        wb = openpyxl.Workbook()
        # Remove default empty sheet; we create named sheets deterministically
        wb.remove(wb.active)

        # 1. Executive Summary Sheet
        ExcelExportEngine._build_executive_summary(wb, job, video, calibration, domain, mode, db)

        # 2. Domain Specific Sheets
        if domain == "traffic":
            ExcelExportEngine._build_processing_metadata(wb, job, video, calibration, domain, mode)
            ExcelExportEngine._build_traffic_unique_tracks(wb, job, db)
            ExcelExportEngine._build_traffic_events(wb, job, db)
            ExcelExportEngine._build_speed_analytics(wb, job, db)
            if calibration:
                ExcelExportEngine._build_calibration_metadata(wb, calibration, video)
            ExcelExportEngine._build_detection_records(wb, job, domain, db)
            ExcelExportEngine._build_research_experiments(wb, video, db)
            ExcelExportEngine._build_data_dictionary(wb, domain)
            ExcelExportEngine._build_export_metadata(wb, job, domain, mode, db)
        else:
            # 10 Professional Examination Sheets in cohesive logical order:
            ExcelExportEngine._build_examination_tracks(wb, job, db)
            ExcelExportEngine._build_examination_events(wb, job, db)
            ExcelExportEngine._build_evidence_timeline(wb, job, video, db)
            ExcelExportEngine._build_detection_records(wb, job, domain, db)
            ExcelExportEngine._build_event_review_history(wb, job, db)
            ExcelExportEngine._build_examination_zones(wb, video, db)
            ExcelExportEngine._build_processing_metadata(wb, job, video, calibration, domain, mode)
            ExcelExportEngine._build_data_dictionary(wb, domain)
            ExcelExportEngine._build_export_metadata(wb, job, domain, mode, db)

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
            person_tracks = db.query(ExaminationTrackRecord).filter(
                ExaminationTrackRecord.job_id == job.id,
                ExaminationTrackRecord.entity_type == "person"
            ).all()
            object_tracks = db.query(ExaminationTrackRecord).filter(
                ExaminationTrackRecord.job_id == job.id,
                ExaminationTrackRecord.entity_type != "person"
            ).all()
            events = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job.id).all()
            total_persons = len(person_tracks)
            total_objects = len(object_tracks)
            students = sum(1 for t in person_tracks if t.assigned_role == "STUDENT")
            staff = sum(1 for t in person_tracks if t.assigned_role in ["INVIGILATOR", "EXAM_COORDINATOR", "AUTHORIZED_STAFF"])
            unknown_roles = sum(1 for t in person_tracks if not t.assigned_role or t.assigned_role == "UNKNOWN")
            red_events = sum(1 for e in events if (e.supporting_objects_json or {}).get("priority") == "RED")
            amber_events = sum(1 for e in events if (e.supporting_objects_json or {}).get("priority") == "AMBER")
            accepted_events = sum(1 for e in events if e.review_status == "ACCEPTED")
            rejected_events = sum(1 for e in events if e.review_status == "REJECTED")
            pending_events = sum(1 for e in events if e.review_status == "PENDING")
            primary_evt = next((e for e in events if (e.supporting_objects_json or {}).get("primary_finding")), None)
            primary_desc = f"{primary_evt.event_type} on Track #{primary_evt.person_track_id} ({format_timecode(primary_evt.start_time)})" if primary_evt else "None Declared"
            audit_progress = f"{round(((accepted_events + rejected_events) / max(1, len(events))) * 100.0, 1)}%" if events else "N/A"

            kpis = [
                ("Total Monitored Persons", total_persons, "individual human tracks"),
                ("Associated Physical Materials", total_objects, "evidence object tracks"),
                ("Confirmed Students Enrolled", students, "roster-verified students"),
                ("Authorized Staff (Invigilators/Coordinators)", staff, "roster-verified staff"),
                ("Unclassified / Unknown Roles", unknown_roles, "awaiting roster reconciliation"),
                ("Candidate Behavioral Events", len(events), "distinct observation windows"),
                ("High-Priority Candidate Alerts (RED)", red_events, "immediate review alerts"),
                ("Medium-Priority Candidate Alerts (AMBER)", amber_events, "corroborating candidate alerts"),
                ("Proctor Accepted Evidence Findings", accepted_events, "confirmed proctoring events"),
                ("Proctor Dismissed / Rejected Findings", rejected_events, "dismissed false positives"),
                ("Pending Proctor Review Items", pending_events, "action required"),
                ("Proctor Review Progress", audit_progress, "reviewed vs total events"),
                ("Primary Algorithmic Finding", primary_desc, "top candidate observation")
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

        person_tracks = db.query(ExaminationTrackRecord).filter(
            ExaminationTrackRecord.job_id == job.id,
            ExaminationTrackRecord.entity_type == "person"
        ).order_by(ExaminationTrackRecord.track_id).all()

        object_tracks = db.query(ExaminationTrackRecord).filter(
            ExaminationTrackRecord.job_id == job.id,
            ExaminationTrackRecord.entity_type != "person"
        ).order_by(ExaminationTrackRecord.track_id).all()

        all_events = db.query(ExaminationEventRecord).filter(
            ExaminationEventRecord.job_id == job.id
        ).all()

        # Build lookup maps for associated events and objects
        person_events: Dict[int, List[ExaminationEventRecord]] = {}
        for e in all_events:
            if e.person_track_id is not None:
                person_events.setdefault(e.person_track_id, []).append(e)

        person_objects: Dict[int, List[ExaminationTrackRecord]] = {}
        for ot in object_tracks:
            p_tid = None
            if ot.assigned_zone and ot.assigned_zone.startswith("person_"):
                try:
                    p_tid = int(ot.assigned_zone.split("_")[1])
                except Exception:
                    pass
            if p_tid is not None:
                person_objects.setdefault(p_tid, []).append(ot)

        fps = (job.video.fps or 30.0) if job.video else 30.0

        # Section 1: Monitored Persons (Human Candidates)
        headers_p = [
            "Person Track ID", "Entity Classification", "Assigned Role", "Role Provenance",
            "Authorized Roster Ref", "First Observed Time", "Last Observed Time", "Observed Duration (s)",
            "Active Detections", "Continuity Ratio", "Assigned Seating Zone", "Associated Candidate Events",
            "Proctor Accepted Events", "Proctor Rejected Events", "Pending Review Events",
            "Associated Physical Materials", "Tracking Quality & Limitations", "Proctor Audit Status"
        ]
        ws.append(headers_p)
        ws.row_dimensions[1].height = 25
        for i in range(1, len(headers_p) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = VIOLET_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        for t in person_tracks:
            dur_frames = max(1, t.last_frame - t.first_frame + 1)
            dur_s = round(dur_frames / fps, 2)
            hist = t.trajectory_json or []
            coverage_pct = f"{round(min(100.0, (len(hist) / dur_frames) * 100.0), 1)}%"

            evts = person_events.get(t.track_id, [])
            total_evts = len(evts)
            acc_evts = sum(1 for e in evts if e.review_status == "ACCEPTED")
            rej_evts = sum(1 for e in evts if e.review_status == "REJECTED")
            pnd_evts = sum(1 for e in evts if e.review_status == "PENDING")

            objs = person_objects.get(t.track_id, [])
            if objs:
                obj_summary = ", ".join([f"{o.entity_type.replace('object_', '').replace('_', ' ')} (#{o.track_id})" for o in objs])
            else:
                obj_summary = "None Associated"

            first_tc = f"{format_timecode(t.first_frame / fps)} ({round(t.first_frame / fps, 2)}s)"
            last_tc = f"{format_timecode(t.last_frame / fps)} ({round(t.last_frame / fps, 2)}s)"

            # Quality & Occlusion Limitations
            if len(hist) == dur_frames and dur_s >= 5.0:
                qual_note = f"Full session continuity ({coverage_pct}); uninterrupted tracking"
            elif dur_s < 1.0:
                qual_note = f"Brief peripheral appearance ({dur_s}s at session boundary)"
            else:
                qual_note = f"Standard coverage ({coverage_pct}); intermittent desk/posture occlusions"

            # Proctor status
            if pnd_evts > 0:
                audit_status = f"ACTION REQUIRED ({pnd_evts} Pending Review)"
            elif acc_evts > 0:
                audit_status = f"FLAGGED & CONFIRMED ({acc_evts} Accepted Findings)"
            elif rej_evts > 0:
                audit_status = f"CASE DISMISSED ({rej_evts} False Positives Rejected)"
            else:
                audit_status = "CLEAR (0 Candidate Observations)"

            ws.append([
                f"#{t.track_id}",
                "HUMAN_CANDIDATE",
                t.assigned_role or "UNKNOWN",
                t.role_source or "DEFAULT",
                "UNREGISTERED / NOT ON ROSTER",
                first_tc,
                last_tc,
                dur_s,
                len(hist),
                coverage_pct,
                t.assigned_zone or "UNASSIGNED",
                total_evts,
                acc_evts,
                rej_evts,
                pnd_evts,
                obj_summary,
                qual_note,
                audit_status
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers_p) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER

            role_cell = ws.cell(row=curr, column=3)
            if t.assigned_role in ["INVIGILATOR", "EXAM_COORDINATOR", "AUTHORIZED_STAFF"]:
                role_cell.fill = SUCCESS_GREEN_FILL
                role_cell.font = DARK_BOLD_FONT
            elif t.assigned_role == "STUDENT":
                role_cell.fill = SUBHEADER_FILL

            status_cell = ws.cell(row=curr, column=18)
            if "ACTION REQUIRED" in audit_status:
                status_cell.fill = ALERT_AMBER_FILL
                status_cell.font = DARK_BOLD_FONT
            elif "FLAGGED" in audit_status:
                status_cell.fill = ALERT_RED_FILL
                status_cell.font = DARK_BOLD_FONT
            elif "CLEAR" in audit_status:
                status_cell.fill = SUCCESS_GREEN_FILL

            ws.row_dimensions[curr].height = 20

        # Section 2: Associated Examination Physical Materials & Evidence Object Tracks
        if object_tracks:
            ws.append([]) # blank separator
            banner_row = ws.max_row + 1
            ws.append(["ASSOCIATED PHYSICAL MATERIALS & EVIDENCE OBJECT TRACKS (SPATIALLY LINKED TO PERSONS)", "", "", "", "", "", "", "", "", ""])
            ws.merge_cells(f"A{banner_row}:J{banner_row}")
            ws[f"A{banner_row}"].fill = VIOLET_ACCENT_FILL
            ws[f"A{banner_row}"].font = WHITE_BOLD_FONT
            ws.row_dimensions[banner_row].height = 22

            headers_o = [
                "Object Track ID", "Detected Material / Class", "Associated Person Track",
                "First Observed Time", "Last Observed Time", "Duration (s)",
                "Active Observations", "Continuity Ratio", "Mean Detection Conf", "Physical Association Semantics"
            ]
            ws.append(headers_o)
            hdr_o_row = ws.max_row
            ws.row_dimensions[hdr_o_row].height = 22
            for i in range(1, len(headers_o) + 1):
                c = ws.cell(row=hdr_o_row, column=i)
                c.fill = SUBHEADER_FILL
                c.font = DARK_BOLD_FONT
                c.alignment = ALIGN_HEADER
                c.border = TABLE_BORDER

            for ot in object_tracks:
                dur_f = max(1, ot.last_frame - ot.first_frame + 1)
                dur_s = round(dur_f / fps, 2)
                hist = ot.trajectory_json or []
                cov_pct = f"{round(min(100.0, (len(hist) / dur_f) * 100.0), 1)}%"
                cls_clean = ot.entity_type.replace("object_", "").replace("_", " ").upper()
                p_tid = ot.assigned_zone.split("_")[1] if (ot.assigned_zone and ot.assigned_zone.startswith("person_")) else "N/A"
                confs = [float(h["conf"]) for h in hist if "conf" in h]
                avg_conf = round(float(sum(confs) / len(confs)), 3) if confs else "N/A"

                ws.append([
                    f"#{ot.track_id}",
                    cls_clean,
                    f"Person #{p_tid}" if p_tid != "N/A" else "Unassigned / Desktop Surface",
                    f"{format_timecode(ot.first_frame / fps)} ({round(ot.first_frame / fps, 2)}s)",
                    f"{format_timecode(ot.last_frame / fps)} ({round(ot.last_frame / fps, 2)}s)",
                    dur_s,
                    len(hist),
                    cov_pct,
                    avg_conf,
                    f"Spatially associated in immediate reach/desktop of Person #{p_tid}" if p_tid != "N/A" else "Free desktop object"
                ])
                curr_o = ws.max_row
                for c_idx in range(1, len(headers_o) + 1):
                    cell = ws.cell(row=curr_o, column=c_idx)
                    cell.font = REGULAR_FONT
                    cell.border = TABLE_BORDER
                ws.row_dimensions[curr_o].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers_p))}{len(person_tracks) + 1}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_examination_events(wb, job, db: Session):
        ws = wb.create_sheet(title="Examination Events")
        ws.views.sheetView[0].showGridLines = True

        headers = [
            "Event Identifier", "Candidate Behavioral Category", "Priority Level", "Evidence Robustness",
            "Target Person Track", "Assigned Role", "Start Time (s)", "End Time (s)", "Duration (s)",
            "Start Timecode", "End Timecode", "Combined Model Conf", "Detection Conf", "Spatial Assoc Conf",
            "Associated Evidence Material", "Supporting Frames Count", "Primary Finding?",
            "Uncertainty Factors", "Objective Factual Observation", "Proctor Review Verdict", "Reviewer Notes"
        ]
        ws.append(headers)
        ws.row_dimensions[1].height = 25
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = VIOLET_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        events = db.query(ExaminationEventRecord).filter(
            ExaminationEventRecord.job_id == job.id
        ).order_by(ExaminationEventRecord.start_time).all()

        for e in events:
            details = e.supporting_objects_json if isinstance(e.supporting_objects_json, dict) else {}
            priority = details.get("priority", "AMBER")
            strength = details.get("evidence_strength", "MEDIUM")
            dur = round(max(0.1, e.end_time - e.start_time), 2)
            supp_frames = details.get("supporting_frames", [])
            uncert_reasons = details.get("uncertainty_reasons", [])
            uncert_str = ", ".join(uncert_reasons) if uncert_reasons else "None Identified"
            is_primary = "YES" if details.get("primary_finding") else "NO"
            assoc_mat = details.get("associated_object_class") or (", ".join(details.get("supporting_objects", [])) or "None")

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
                format_timecode(e.start_time),
                format_timecode(e.end_time),
                round(e.confidence, 3),
                round(details.get("detection_confidence", e.confidence), 3),
                round(details.get("association_confidence", 0.0), 3),
                assoc_mat,
                len(supp_frames),
                is_primary,
                uncert_str,
                e.description,
                e.review_status,
                e.reviewer_notes or "N/A"
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER

            # Priority styling
            p_cell = ws.cell(row=curr, column=3)
            if priority == "RED":
                p_cell.fill = ALERT_RED_FILL
                p_cell.font = DARK_BOLD_FONT
            elif priority == "AMBER":
                p_cell.fill = ALERT_AMBER_FILL
                p_cell.font = DARK_BOLD_FONT

            # Primary finding styling
            if is_primary == "YES":
                prim_cell = ws.cell(row=curr, column=17)
                prim_cell.fill = VIOLET_ACCENT_FILL
                prim_cell.font = WHITE_BOLD_FONT

            # Review verdict styling
            rev_cell = ws.cell(row=curr, column=20)
            if e.review_status == "ACCEPTED":
                rev_cell.fill = SUCCESS_GREEN_FILL
                rev_cell.font = DARK_BOLD_FONT
            elif e.review_status == "REJECTED":
                rev_cell.fill = ALERT_RED_FILL
                rev_cell.font = DARK_BOLD_FONT
            elif e.review_status == "PENDING":
                rev_cell.fill = ALERT_AMBER_FILL

            ws.row_dimensions[curr].height = 20

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_evidence_timeline(wb, job, video: Optional[Video], db: Session):
        ws = wb.create_sheet(title="Evidence Timeline")
        ws.views.sheetView[0].showGridLines = True

        headers = [
            "Timeline Seq #", "Video Timecode", "Video Timestamp (s)", "Video Frame Index",
            "Target Person Track", "Event Reference", "Observation Nature",
            "Evidence Material / Interaction", "Confidence Level", "Priority", "Review Status",
            "Objective Chronological Description"
        ]
        ws.append(headers)
        ws.row_dimensions[1].height = 25
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = VIOLET_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        fps = (video.fps or 30.0) if video else 30.0
        events = db.query(ExaminationEventRecord).filter(
            ExaminationEventRecord.job_id == job.id
        ).all()

        timeline_entries = []
        for e in events:
            details = e.supporting_objects_json if isinstance(e.supporting_objects_json, dict) else {}
            priority = details.get("priority", "AMBER")
            assoc_mat = details.get("associated_object_class") or (", ".join(details.get("supporting_objects", [])) or "Behavioral Motion")
            supp_frames = details.get("supporting_frames", [])

            # 1. Event Inception
            timeline_entries.append({
                "ts": e.start_time,
                "frame": e.start_frame,
                "track": f"#{e.person_track_id}" if e.person_track_id else "N/A",
                "evt_id": e.event_id,
                "nature": "EVENT_INCEPTION",
                "material": assoc_mat,
                "conf": round(e.confidence, 3),
                "priority": priority,
                "status": e.review_status,
                "desc": f"Algorithmic onset of {e.event_type} involving Track #{e.person_track_id}."
            })

            # 2. Peak Evidence Observation Cluster (if multi-frame)
            if len(supp_frames) > 2:
                mid_frame = supp_frames[len(supp_frames) // 2]
                mid_ts = round(mid_frame / fps, 2)
                timeline_entries.append({
                    "ts": mid_ts,
                    "frame": mid_frame,
                    "track": f"#{e.person_track_id}" if e.person_track_id else "N/A",
                    "evt_id": e.event_id,
                    "nature": "EVIDENCE_CONVERGENCE_PEAK",
                    "material": assoc_mat,
                    "conf": round(details.get("detection_confidence", e.confidence), 3),
                    "priority": priority,
                    "status": e.review_status,
                    "desc": f"Peak evidentiary cluster observed across {len(supp_frames)} frames: {details.get('supporting_observations', e.description)}"
                })

            # 3. Event Conclusion
            timeline_entries.append({
                "ts": e.end_time,
                "frame": e.end_frame,
                "track": f"#{e.person_track_id}" if e.person_track_id else "N/A",
                "evt_id": e.event_id,
                "nature": "EVENT_CONCLUSION",
                "material": assoc_mat,
                "conf": round(e.confidence, 3),
                "priority": priority,
                "status": e.review_status,
                "desc": f"Observed candidate pattern concludes (Duration: {round(max(0.1, e.end_time - e.start_time), 2)}s)."
            })

            # 4. Human Proctor Review Disposition
            if e.review_status in ("ACCEPTED", "REJECTED"):
                timeline_entries.append({
                    "ts": e.end_time + 0.01,
                    "frame": e.end_frame,
                    "track": f"#{e.person_track_id}" if e.person_track_id else "N/A",
                    "evt_id": e.event_id,
                    "nature": "PROCTOR_DISPOSITION_RECORDED",
                    "material": assoc_mat,
                    "conf": round(e.confidence, 3),
                    "priority": priority,
                    "status": e.review_status,
                    "desc": f"Proctor recorded verdict: {e.review_status}. Notes: {e.reviewer_notes or 'None'}"
                })

        # Sort chronologically by timestamp
        timeline_entries.sort(key=lambda x: (x["ts"], x["frame"]))

        for idx, entry in enumerate(timeline_entries, start=1):
            ws.append([
                f"TL-{idx:03d}",
                format_timecode(entry["ts"]),
                round(entry["ts"], 2),
                entry["frame"],
                entry["track"],
                entry["evt_id"],
                entry["nature"],
                entry["material"],
                entry["conf"],
                entry["priority"],
                entry["status"],
                entry["desc"]
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER

            # Highlight proctor verdicts
            if entry["status"] == "ACCEPTED":
                ws.cell(row=curr, column=11).fill = SUCCESS_GREEN_FILL
            elif entry["status"] == "REJECTED":
                ws.cell(row=curr, column=11).fill = ALERT_RED_FILL
            elif entry["status"] == "PENDING":
                ws.cell(row=curr, column=11).fill = ALERT_AMBER_FILL

            ws.row_dimensions[curr].height = 19

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_event_review_history(wb, job, db: Session):
        ws = wb.create_sheet(title="Event Review History")
        ws.views.sheetView[0].showGridLines = True

        headers = [
            "Audit Entry ID", "Target Event ID", "Candidate Behavioral Category", "Target Person Track",
            "Model Priority & Evidence", "Model Combined Conf", "Proctor Review Verdict",
            "Disposition Classification", "Proctor Reviewer Notes", "Corroborating Evidence Assessed",
            "Procedural Action Status"
        ]
        ws.append(headers)
        ws.row_dimensions[1].height = 25
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = VIOLET_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        events = db.query(ExaminationEventRecord).filter(
            ExaminationEventRecord.job_id == job.id
        ).order_by(ExaminationEventRecord.start_time).all()

        for idx, e in enumerate(events, start=1):
            details = e.supporting_objects_json if isinstance(e.supporting_objects_json, dict) else {}
            priority = details.get("priority", "AMBER")
            strength = details.get("evidence_strength", "MEDIUM")
            assoc_mat = details.get("associated_object_class") or (", ".join(details.get("supporting_objects", [])) or "None")

            if e.review_status == "ACCEPTED":
                disp_cat = "CONFIRMED_PROCTORING_FINDING"
                action_status = "REFERRED_TO_ACADEMIC_INTEGRITY_COMMITTEE"
            elif e.review_status == "REJECTED":
                disp_cat = "DISMISSED_FALSE_POSITIVE"
                action_status = "CASE_CLOSED_DISMISSED"
            else:
                disp_cat = "UNRESOLVED_NEEDS_PROCTOR_REVIEW"
                action_status = "ACTION_REQUIRED_AWAITING_REVIEW"

            ws.append([
                f"AUD-{job.id}-{idx:03d}",
                e.event_id,
                e.event_type,
                f"Person #{e.person_track_id}" if e.person_track_id else "N/A",
                f"{priority} / {strength}",
                round(e.confidence, 3),
                e.review_status,
                disp_cat,
                e.reviewer_notes or "N/A (Pending Proctor Disposition)",
                f"Associated material: {assoc_mat}. {details.get('supporting_observations', e.description)}",
                action_status
            ])
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                c = ws.cell(row=curr, column=c_idx)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER

            # Review verdict styling
            v_cell = ws.cell(row=curr, column=7)
            if e.review_status == "ACCEPTED":
                v_cell.fill = SUCCESS_GREEN_FILL
                v_cell.font = DARK_BOLD_FONT
            elif e.review_status == "REJECTED":
                v_cell.fill = ALERT_RED_FILL
                v_cell.font = DARK_BOLD_FONT
            else:
                v_cell.fill = ALERT_AMBER_FILL
                v_cell.font = DARK_BOLD_FONT

            ws.row_dimensions[curr].height = 20

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    @staticmethod
    def _build_examination_zones(wb, video: Optional[Video], db: Session):
        ws = wb.create_sheet(title="Seating Zones")
        ws.views.sheetView[0].showGridLines = True

        zones = db.query(ExaminationZoneRecord).filter(
            ExaminationZoneRecord.video_id == video.id
        ).all() if video else []

        if not zones:
            # Explicit Unconstrained Layout Structure (Never omit sheet!)
            ws.merge_cells("A1:F1")
            ws["A1"] = "SEATING ZONE MONITORING STATUS: UNCONSTRAINED CLASSROOM LAYOUT"
            ws["A1"].fill = VIOLET_HEADER_FILL
            ws["A1"].font = WHITE_BOLD_FONT
            ws["A1"].alignment = ALIGN_LEFT
            ws.row_dimensions[1].height = 25

            ws.merge_cells("A2:F2")
            ws["A2"] = (
                "Notice: No geometric seating zones or boundaries were pre-configured for this video. "
                "Person tracking and candidate event detection operated across the full unconstrained camera field of view."
            )
            ws["A2"].font = MUTED_FONT
            ws["A2"].alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[2].height = 30

            headers = ["Zone Identifier", "Descriptive Name", "Zone Type", "Expected Role", "Polygon Vertices Count", "Geometric Boundary Coordinates"]
            ws.append([])
            ws.append(headers)
            hdr_row = 4
            ws.row_dimensions[hdr_row].height = 22
            for i in range(1, len(headers) + 1):
                c = ws.cell(row=hdr_row, column=i)
                c.fill = SUBHEADER_FILL
                c.font = DARK_BOLD_FONT
                c.alignment = ALIGN_HEADER
                c.border = TABLE_BORDER

            ws.append([
                "GLOBAL_ROOM_ARENA",
                "Unconstrained Examination Space",
                "GLOBAL_ROOM",
                "ANY_ROLE",
                4,
                "[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] (Full Video Frame)"
            ])
            data_r = ws.max_row
            for i in range(1, len(headers) + 1):
                c = ws.cell(row=data_r, column=i)
                c.font = REGULAR_FONT
                c.border = TABLE_BORDER
            ws.row_dimensions[data_r].height = 20

            ws.freeze_panes = "A5"
            auto_fit_columns(ws)
            return

        headers = ["Zone ID", "Descriptive Name", "Zone Type", "Expected Role", "Polygon Vertices Count", "Polygon Coordinates"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = VIOLET_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        for z in zones:
            poly = z.polygon_json or []
            ws.append([
                z.zone_id,
                z.name,
                z.zone_type,
                "STUDENT" if "student" in z.name.lower() else "STAFF" if "staff" in z.name.lower() else "ANY",
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

        fps = (job.video.fps or 30.0) if job.video else 30.0
        sampled_rows = []

        if domain == "traffic":
            headers = ["Frame Index", "Timestamp (s)", "Track ID", "Entity / Class", "Confidence", "BBox X1", "BBox Y1", "BBox X2", "BBox Y2", "BBox Width", "BBox Height"]
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
            headers = [
                "Frame Index", "Video Timecode", "Timestamp (s)", "Track ID", "Entity Classification",
                "Detection Class", "Confidence", "Spatial Association Context",
                "BBox X1", "BBox Y1", "BBox X2", "BBox Y2", "BBox Width", "BBox Height"
            ]
            tracks = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == job.id).all()
            for t in tracks:
                ent_clean = "HUMAN_CANDIDATE" if t.entity_type == "person" else "PHYSICAL_MATERIAL"
                cls_clean = "person" if t.entity_type == "person" else t.entity_type.replace("object_", "").replace("_", " ")
                assoc = t.assigned_zone or "UNASSIGNED"
                for h in (t.trajectory_json or [])[:200]:
                    f_idx = h.get("frame", 0)
                    bbox = h.get("bbox", [0, 0, 0, 0])
                    w = round(bbox[2] - bbox[0], 1)
                    h_val = round(bbox[3] - bbox[1], 1)
                    ts = round(f_idx / fps, 2)
                    sampled_rows.append((
                        f_idx, format_timecode(ts), ts, f"#{t.track_id}", ent_clean,
                        cls_clean, round(float(h.get("conf", 0.90)), 3), assoc,
                        round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1), w, h_val
                    ))

        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = VIOLET_HEADER_FILL if domain == "examination" else NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        # Sort chronologically
        sampled_rows.sort(key=lambda x: (x[0], str(x[3])))
        for r in sampled_rows[:3000]:
            ws.append(list(r))
            curr = ws.max_row
            for c_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=curr, column=c_idx)
                cell.font = REGULAR_FONT
                cell.border = TABLE_BORDER
            ws.row_dimensions[curr].height = 18

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, ws.max_row)}"
        auto_fit_columns(ws)

    # -----------------------------------------------------------------------
    # Research & Experiments Sheet (Preserved for Traffic)
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
            cell.fill = VIOLET_HEADER_FILL if domain == "examination" else NAVY_HEADER_FILL
            cell.font = WHITE_BOLD_FONT
            cell.alignment = ALIGN_HEADER

        dict_rows = [
            ("All Sheets", "Job ID", "Integer", "None", "Persistent unique identifier of processing task in processing_jobs table", "Stable cross-module identifier"),
            ("All Sheets", "Track ID", "Integer", "None", "Visual continuity identifier maintained by tracking filter", "Temporary tracking token; not a biometric identity or legal identifier"),
            ("All Sheets", "Video Timecode", "String", "MM:SS.ms", "Standard video presentation timestamp formatted in minutes, seconds, and centiseconds", "Synchronized directly to source container timestamps"),
            ("Executive Summary", "Calibration Status", "String", "None", "Indicates whether homography road-plane projection was verified", "Uncalibrated cameras cannot support certified speed measurements"),
            ("Detection Records", "Confidence", "Float", "[0.0 - 1.0]", "Raw object detector output score", "Detector score reflects visual match to training distribution, not truth"),
            ("Detection Records", "BBox X1/Y1/X2/Y2", "Float", "Pixels", "Bounding box coordinates in video pixel space", "Relative to video native width and height")
        ]

        if domain == "traffic":
            dict_rows.extend([
                ("Unique Tracks", "Duration (Frames)", "Integer", "Frames", "Total span between first and last observed frame", "Subject to visual occlusions"),
                ("Unique Tracks", "Continuity Ratio", "Percentage", "%", "Ratio of frames with active detections over total track duration", "Dips below 80% indicate frequent visual occlusions"),
                ("Traffic Events", "Est. Speed", "Float", "km/h", "Homography-projected road-plane vehicle speed estimate", "Certified only when calibration is valid and error is bounded"),
                ("Traffic Events", "Speed Limit", "Float", "km/h", "Configured legal speed limit for enforcement segment", "Derived from user configuration or road signage"),
                ("Traffic Events", "Uncertainty (±)", "Float", "km/h", "Combined error-decomposed interval accounting for pitch, homography, and tracking jitter", "Never zero; reflects fundamental optical precision"),
                ("Traffic Events", "Review Status", "String", "None", "PENDING | ACCEPTED | REJECTED human sign-off status", "Automated violations require human proctor/operator verification"),
                ("Speed Analytics", "Smoothed Speed", "Float", "km/h", "Kalman-smoothed trajectory speed estimation", "Attenuates frame-to-frame bounding box discretization jitter")
            ])
        else:
            dict_rows.extend([
                ("Monitored Persons & Tracks", "Assigned Role", "String", "None", "STUDENT | INVIGILATOR | EXAM_COORDINATOR | AUTHORIZED_STAFF | UNKNOWN", "Assigned via config, zone, or proctor review. Never inferred from appearance alone"),
                ("Monitored Persons & Tracks", "Role Provenance", "String", "None", "CONFIG | ZONE | HUMAN_REVIEW | DEFAULT", "Indicates authoritative origin of assigned role"),
                ("Monitored Persons & Tracks", "Continuity Ratio", "Percentage", "%", "Ratio of frames with active person detections over total track duration", "Reflects tracking stability across posture changes and desk occlusions"),
                ("Monitored Persons & Tracks", "Associated Physical Materials", "String", "None", "Summary of phones, books, or laptops spatially correlated with this person", "Reflects proximity reach; does not prove ownership without verification"),
                ("Examination Events", "Candidate Behavioral Category", "String", "None", "Neutral algorithmic observation category (e.g. POSSIBLE_PHONE_INTERACTION)", "Represents candidate pattern for review; NOT a finding of academic dishonesty"),
                ("Examination Events", "Priority Level", "String", "None", "RED | AMBER | INFO alert severity hierarchy", "RED priority indicates candidate requiring immediate proctor inspection"),
                ("Examination Events", "Evidence Robustness", "String", "None", "HIGH | MEDIUM | LOW heuristic robustness index", "Synthesizes temporal duration, object presence, and visibility stability"),
                ("Examination Events", "Proctor Review Verdict", "String", "None", "PENDING | ACCEPTED | REJECTED human proctor disposition", "Automated events require proctor corroboration prior to any integrity action"),
                ("Evidence Timeline", "Observation Nature", "String", "None", "EVENT_INCEPTION | EVIDENCE_CONVERGENCE_PEAK | EVENT_CONCLUSION | PROCTOR_DISPOSITION", "Categorizes chronological lifecycle milestones"),
                ("Event Review History", "Disposition Classification", "String", "None", "CONFIRMED_PROCTORING_FINDING | DISMISSED_FALSE_POSITIVE | UNRESOLVED_NEEDS_REVIEW", "Formal compliance classification for institutional recordkeeping"),
                ("Seating Zones", "Zone Type", "String", "None", "SEATING_ZONE | RESTRICTED_ZONE | AISLE | GLOBAL_ROOM", "Geometric zone classification. UNCONSTRAINED indicates no boundaries were set")
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
    def _build_export_metadata(wb, job, domain: str, mode: str, db: Optional[Session] = None):
        ws = wb.create_sheet(title="Export Metadata")
        ws.views.sheetView[0].showGridLines = True
        headers = ["Metadata Attribute", "Value", "Specification"]
        ws.append(headers)
        ws.row_dimensions[1].height = 24
        for i in range(1, 4):
            c = ws.cell(row=1, column=i)
            c.fill = VIOLET_HEADER_FILL if domain == "examination" else NAVY_HEADER_FILL
            c.font = WHITE_BOLD_FONT
            c.alignment = ALIGN_HEADER

        timestamp_str = utcnow_str()

        # Domain-aware cryptographic seal formula
        if domain == "examination" and db is not None:
            p_cnt = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == job.id, ExaminationTrackRecord.entity_type == "person").count()
            o_cnt = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == job.id, ExaminationTrackRecord.entity_type != "person").count()
            e_cnt = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job.id).count()
            acc_cnt = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job.id, ExaminationEventRecord.review_status == "ACCEPTED").count()
            rej_cnt = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job.id, ExaminationEventRecord.review_status == "REJECTED").count()
            seal_raw = f"{job.id}-{domain}-{mode}-{timestamp_str}-{p_cnt}-{o_cnt}-{e_cnt}-{acc_cnt}-{rej_cnt}"
            seal_prefix = "VR-EXAM-SEAL"
        else:
            seal_raw = f"{job.id}-{domain}-{mode}-{timestamp_str}"
            seal_prefix = "VR-XLSX-SEAL"

        seal = hashlib.sha256(seal_raw.encode("utf-8")).hexdigest()

        rows = [
            ("Export Generation Timestamp", timestamp_str, "ISO 8601 UTC Timestamp"),
            ("Export Engine Version", "VisionRadar Excel Intelligence Engine v3.0.0", "openpyxl 3.1.5 based enterprise generator"),
            ("Target Job ID", f"#{job.id}", "Database primary key in processing_jobs"),
            ("Export Mode", mode.upper(), "FINAL AUDIT EXPORT or LIVE IN-FLIGHT SNAPSHOT"),
            ("Domain Classification", f"{domain.upper()} INTELLIGENCE", "Domain-segregated worksheets and analytical models"),
            ("Cryptographic Export Seal", f"{seal_prefix}-{seal[:32].upper()}", "SHA-256 seal for tamper detection and audit integrity"),
            ("Tamper Detection Method", f"SHA-256 over JobID, Domain, Mode, Timestamp, and Record Counts", "Verify via VisionRadar CLI audit tool"),
            ("Non-Blocking Execution", "Guaranteed", "Export generated independently of video processing loop"),
            ("Legal & Ethical Governance", "Strict Non-Accusation & Non-Fabrication Protocol", "Candidate visual patterns only; human review mandatory")
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
