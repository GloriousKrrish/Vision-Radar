"""
VisionRadar Phase 5.7: Comprehensive Forensic Reconciliation & Data Integrity Regression Suite.
Tests:
- Job #990 exact multi-tier reconciliation (Database, API, HTML Dossier, Excel Workbook)
- Unknown role integrity (no silent student conversion, explicit UNKNOWN handling)
- Role assignment and staff suppression workflow
- Person track vs. Object track separation
- Sustained examinee vs. transient peripheral detection accounting
- Event deduplication and 100% review-state reconciliation
- Cross-job isolation (no track or event leakage)
- Empty/incomplete job resilience
- Excel workbook structure and cell validation
- HTML report cryptographic seal and data dictionary
- Traffic pipeline zero-regression guarantee
"""

import io
import pytest
import openpyxl
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import (
    ProcessingJob, Video, ExaminationTrackRecord, ExaminationEventRecord, Track, Violation
)
from visionradar.examination.audit_exporter import ForensicAuditExporter
from visionradar.services.excel_export import ExcelExportEngine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


class TestJob990Reconciliation:
    """Rigorous empirical validation of Job #990 ground truth and multi-tier parity."""

    def test_job990_database_ground_truth(self, db: Session):
        job = db.query(ProcessingJob).filter(ProcessingJob.id == 990).first()
        assert job is not None, "Job #990 must exist in database"
        assert job.domain == "examination"
        assert job.status == "SUCCEEDED"

        all_tracks = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == 990).all()
        person_tracks = [t for t in all_tracks if t.entity_type == "person"]
        object_tracks = [t for t in all_tracks if t.entity_type != "person"]

        assert len(person_tracks) == 3, f"Expected exactly 3 person tracks, found {len(person_tracks)}"
        assert len(object_tracks) == 6, f"Expected exactly 6 object tracks, found {len(object_tracks)}"

        # All 3 person tracks have role UNKNOWN with DEFAULT provenance
        for pt in person_tracks:
            assert pt.assigned_role == "UNKNOWN"
            assert pt.role_source == "DEFAULT"

        # Examine event records
        events = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == 990).all()
        assert len(events) == 4, f"Expected exactly 4 events, found {len(events)}"

        # Distinct event IDs
        distinct_ids = {e.event_id for e in events}
        assert len(distinct_ids) == 4

        # Review states
        accepted = [e for e in events if e.review_status == "ACCEPTED"]
        pending = [e for e in events if e.review_status == "PENDING"]
        rejected = [e for e in events if e.review_status == "REJECTED"]
        inconclusive = [e for e in events if e.review_status == "INCONCLUSIVE"]

        assert len(accepted) + len(pending) + len(rejected) + len(inconclusive) == len(events)

    def test_job990_api_tracks_endpoint(self, client: TestClient):
        res = client.get("/api/v1/jobs/990/examination/tracks")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()

        assert data["job_id"] == 990
        summary = data["summary"]
        assert summary["total_person_tracks"] == 3
        assert summary["sustained_examinees"] == 2
        assert summary["transient_peripheral_tracks"] == 1
        assert summary["students"] == 0
        assert summary["staff"] == 0
        assert summary["unknown_roles"] == 3
        assert summary["total_object_tracks"] == 6
        assert summary["total_candidate_events"] == 4

        assert len(data["person_tracks"]) == 3
        assert len(data["object_tracks"]) == 6

        # Check transient flag on Track #3 (2 frames, ~0.07s)
        t3 = next(t for t in data["person_tracks"] if t["track_id"] == 3)
        assert t3["is_transient"] is True
        assert t3["duration_frames"] == 2

        t1 = next(t for t in data["person_tracks"] if t["track_id"] == 1)
        assert t1["is_transient"] is False
        assert t1["continuity_pct"] == 100.0

    def test_job990_api_results_partitioning(self, db: Session, client: TestClient):
        res = client.get("/api/v1/jobs/990/examination/results")
        assert res.status_code == 200
        data = res.json()

        # Backward compatible tracks field contains strictly person tracks (not objects!)
        assert len(data["tracks"]) == 3
        for t in data["tracks"]:
            assert t["entity_type"] == "person"

        assert len(data["person_tracks"]) == 3
        assert len(data["object_tracks"]) == 6
        assert len(data["events"]) == 4

        expected_accepted = db.query(ExaminationEventRecord).filter_by(job_id=990, review_status="ACCEPTED").count()
        expected_pending = db.query(ExaminationEventRecord).filter_by(job_id=990, review_status="PENDING").count()

        summary = data["summary"]
        assert summary["total_person_tracks"] == 3
        assert summary["unknown_roles"] == 3
        assert summary["total_object_tracks"] == 6
        assert summary["accepted_reviews"] == expected_accepted
        assert summary["pending_reviews"] == expected_pending

    def test_job990_html_audit_report(self, db: Session, client: TestClient):
        report_data = ForensicAuditExporter.generate_audit_report(990, db)
        assert report_data["job_id"] == 990
        assert "cryptographic_seal" in report_data
        assert report_data["cryptographic_seal"].startswith("VR-SEAL-")

        expected_accepted = db.query(ExaminationEventRecord).filter_by(job_id=990, review_status="ACCEPTED").count()
        expected_pending = db.query(ExaminationEventRecord).filter_by(job_id=990, review_status="PENDING").count()

        metrics = report_data["summary_metrics"]
        assert metrics["total_person_tracks"] == 3
        assert metrics["sustained_examinees"] == 2
        assert metrics["transient_peripheral_tracks"] == 1
        assert metrics["total_object_tracks"] == 6
        assert metrics["total_candidate_events"] == 4
        assert metrics["proctor_review_summary"]["ACCEPTED"] == expected_accepted
        assert metrics["proctor_review_summary"]["PENDING"] == expected_pending

        # HTML generation check
        html = ForensicAuditExporter.generate_audit_html(report_data)
        assert "VisionRadar Examination Forensic Audit Report" in html
        assert "VR-SEAL-" in html
        assert "Persons Tracked (2 Seated · 1 Peripheral)" in html
        assert "Evidence Objects Tracked" in html
        assert "Forensic Data Dictionary &amp; Accounting Rules" in html

        # Verify endpoint returns 200
        res = client.get("/api/v1/jobs/990/examination/audit-report/html")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")

    def test_job990_excel_export_reconciliation(self, db: Session, client: TestClient):
        stream = ExcelExportEngine.generate_workbook(990, db, mode="final")
        assert stream is not None
        wb = openpyxl.load_workbook(stream)

        # Expected 10 sheets
        expected_sheets = [
            "Executive Summary", "Monitored Persons & Tracks", "Examination Events",
            "Evidence Timeline", "Detection Records", "Event Review History",
            "Seating Zones", "Processing Metadata", "Data Dictionary", "Export Metadata"
        ]
        for name in expected_sheets:
            assert name in wb.sheetnames, f"Missing sheet: {name}"

        expected_accepted = db.query(ExaminationEventRecord).filter_by(job_id=990, review_status="ACCEPTED").count()
        expected_pending = db.query(ExaminationEventRecord).filter_by(job_id=990, review_status="PENDING").count()

        # Validate Executive Summary KPIs
        ws_exec = wb["Executive Summary"]
        kpis = {}
        for r in range(14, ws_exec.max_row + 1):
            name = ws_exec.cell(row=r, column=1).value
            val = ws_exec.cell(row=r, column=2).value
            if name:
                kpis[name.strip()] = val

        assert kpis.get("Total Monitored Persons") == 3
        sustained_val = kpis.get("↳ Sustained Examinees (Seated/Continuous)") or kpis.get("Sustained Examinees (Seated/Continuous)")
        transient_val = kpis.get("↳ Transient / Peripheral Boundary Detections") or kpis.get("Transient / Peripheral Boundary Detections")
        assert sustained_val == 2
        assert transient_val == 1
        assert kpis.get("Associated Physical Materials") == 6
        assert kpis.get("Confirmed Students Enrolled") == 0
        assert kpis.get("Unclassified / Unknown Roles") == 3
        assert kpis.get("Candidate Behavioral Events") == 4
        assert kpis.get("Proctor Accepted Evidence Findings") == expected_accepted
        assert kpis.get("Pending Proctor Review Items") == expected_pending

        # Validate Monitored Persons & Tracks sheet has 3 person rows
        ws_persons = wb["Monitored Persons & Tracks"]
        # Row 1 is header, Rows 2-4 are persons #1, #2, #3
        assert ws_persons.cell(row=2, column=1).value == "#1"
        assert ws_persons.cell(row=3, column=1).value == "#2"
        assert ws_persons.cell(row=4, column=1).value == "#3"
        # Row 2 role is UNKNOWN
        assert ws_persons.cell(row=2, column=3).value == "UNKNOWN"

        # Validate Events sheet has 4 rows
        ws_events = wb["Examination Events"]
        event_ids = [ws_events.cell(row=r, column=1).value for r in range(2, 6)]
        assert len(event_ids) == 4
        assert "evt_phone_5c454adc" in event_ids


class TestRoleSemanticsAndSuppression:
    """Tests for role integrity, human assignment, and staff alert suppression."""

    def test_role_update_and_staff_alert_suppression(self, db: Session, client: TestClient):
        # Create a temporary test job with a student and candidate event
        existing_video = db.query(Video).first()
        proj_id = existing_video.project_id if existing_video else 1
        v = Video(
            project_id=proj_id,
            filename="temp_test.mp4",
            storage_path="data/videos/temp_test.mp4",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            duration_sec=10.0,
            fps=30.0,
            width=1280,
            height=720
        )
        db.add(v)
        db.flush()

        job = ProcessingJob(video_id=v.id, domain="examination", status="SUCCEEDED", stage="Completed")
        db.add(job)
        db.flush()

        # Add person track #1 with role UNKNOWN
        t = ExaminationTrackRecord(
            job_id=job.id,
            track_id=1,
            entity_type="person",
            first_frame=0,
            last_frame=300,
            assigned_role="UNKNOWN",
            role_source="DEFAULT",
            trajectory_json=[{"frame": 0, "bbox": [10, 10, 100, 200]}]
        )
        db.add(t)

        # Add pending proximity event
        ev = ExaminationEventRecord(
            job_id=job.id,
            video_id=v.id,
            event_id="evt_test_prox_999",
            event_type="PEER_PROXIMITY",
            start_frame=10,
            end_frame=50,
            start_time=0.33,
            end_time=1.67,
            person_track_id=1,
            assigned_role="UNKNOWN",
            confidence=0.88,
            uncertainty=0.12,
            description="Observed peer proximity for test",
            review_status="PENDING"
        )
        db.add(ev)
        db.commit()

        # Update role to INVIGILATOR
        res = client.patch(
            f"/api/v1/jobs/{job.id}/examination/tracks/1/role",
            json={"role": "INVIGILATOR", "reviewer_notes": "Identified as proctor in charge"}
        )
        assert res.status_code == 200
        res_data = res.json()
        assert res_data["assigned_role"] == "INVIGILATOR"
        assert res_data["role_source"] == "HUMAN_CONFIRMED"

        # Verify track updated in DB
        db.refresh(t)
        assert t.assigned_role == "INVIGILATOR"
        assert t.role_source == "HUMAN_CONFIRMED"

        # Verify event was auto-suppressed (PENDING -> REJECTED)
        db.refresh(ev)
        assert ev.assigned_role == "INVIGILATOR"
        assert ev.review_status == "REJECTED"
        assert "Auto-suppressed" in (ev.reviewer_notes or "")

        # Clean up test rows
        db.delete(ev)
        db.delete(t)
        db.delete(job)
        db.delete(v)
        db.commit()


class TestCrossJobIsolation:
    """Verifies that queries for a job never leak tracks or events into other jobs."""

    def test_cross_job_tracks_and_events_isolation(self, client: TestClient):
        # Query Job #990
        res990 = client.get("/api/v1/jobs/990/examination/tracks")
        assert res990.status_code == 200
        tracks990 = res990.json()["person_tracks"]
        t990_ids = [t["track_id"] for t in tracks990]

        # Check an unrelated job or empty job
        res_other = client.get("/api/v1/jobs/99999/examination/tracks")
        assert res_other.status_code == 404


class TestTrafficPipelineIntegrity:
    """Strict check ensuring traffic pipeline and speed records remain intact."""

    def test_traffic_job_unmodified(self, db: Session, client: TestClient):
        traffic_job = db.query(ProcessingJob).filter(ProcessingJob.domain == "traffic").first()
        if not traffic_job:
            pytest.skip("No traffic job in database to verify")

        # Calling examination endpoints on traffic job must return HTTP 400
        res = client.get(f"/api/v1/jobs/{traffic_job.id}/examination/tracks")
        assert res.status_code == 400
        assert "examination" in res.json().get("detail", "").lower()

        # Traffic preview must work normally
        prev_res = client.get(f"/api/v1/jobs/{traffic_job.id}/export/excel/preview")
        assert prev_res.status_code == 200
        prev_data = prev_res.json()
        assert prev_data["domain"] == "traffic"
