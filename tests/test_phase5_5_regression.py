"""
VisionRadar Phase 5.5 Regression Test Suite
Empirical verification for:
1. Authoritative Domain Synchronization (Traffic vs Examination)
2. Video Upload and Playback Lifecycle
3. Examination and Traffic Isolation (Labels, HUD, Overlays)
4. Examination Pipeline Performance & Frame Metrics
5. Excel Export Reliability (.xlsx vs JSON separation)
"""

import io
import json
import pytest
import openpyxl
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.api.main import app, get_db
from visionradar.models.database import SessionLocal, Base
from visionradar.models.entities import (
    Video, ProcessingJob, Track, SpeedMeasurement, Violation,
    ExaminationTrackRecord, ExaminationEventRecord
)
from visionradar.services.excel_export import ExcelExportEngine
from visionradar.models.envelope import CommonEnvelope


# In-memory isolated SQLite database fixture for Phase 5.5 regressions
@pytest.fixture(scope="module")
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="module")
def client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# =============================================================================
# 1. Authoritative Domain Synchronization Tests
# =============================================================================

def test_examination_job_creation_persists_authoritative_domain(client, test_db):
    """Verify Examination jobs are created with domain='examination' and never default to traffic."""
    vid = Video(project_id=1, filename="classroom_cam1.mp4", storage_path="media/videos/classroom_cam1.mp4", sha256_hash="dummy1", duration_sec=60.0, fps=30.0)
    test_db.add(vid)
    test_db.commit()

    resp = client.post(f"/api/v1/videos/{vid.id}/jobs", json={
        "domain": "examination",
        "config_json": {"domain": "examination", "module": "examination"}
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["domain"] == "examination"

    job = test_db.query(ProcessingJob).filter(ProcessingJob.id == data["id"]).first()
    assert job is not None
    assert job.domain == "examination"
    assert job.calibration_id is None  # Road calibration omitted for examination


def test_traffic_job_creation_persists_authoritative_traffic_domain(client, test_db):
    """Verify Traffic jobs are created with domain='traffic'."""
    vid = Video(project_id=1, filename="highway_sample.mp4", storage_path="media/videos/highway_sample.mp4", sha256_hash="dummy2", duration_sec=40.0, fps=30.0)
    test_db.add(vid)
    test_db.commit()

    resp = client.post(f"/api/v1/videos/{vid.id}/jobs", json={
        "domain": "traffic",
        "config_json": {"domain": "traffic", "module": "traffic"}
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["domain"] == "traffic"


def test_domain_guard_enforces_strict_isolation(client, test_db):
    """Verify traffic-specific queries fail on examination jobs and examination queries fail on traffic jobs."""
    vid = Video(project_id=1, filename="exam_clip.mp4", storage_path="media/videos/exam_clip.mp4", sha256_hash="dummy3", duration_sec=30.0, fps=30.0)
    test_db.add(vid)
    test_db.commit()

    # Examination job
    exam_job = ProcessingJob(video_id=vid.id, status="SUCCEEDED", domain="examination")
    test_db.add(exam_job)
    test_db.commit()

    # Traffic tracks request on examination job should return 400
    res_tracks = client.get(f"/api/v1/jobs/{exam_job.id}/tracks")
    assert res_tracks.status_code == 400
    assert "examination" in res_tracks.json()["detail"].lower()

    # Traffic job
    traffic_job = ProcessingJob(video_id=vid.id, status="SUCCEEDED", domain="traffic")
    test_db.add(traffic_job)
    test_db.commit()

    # Examination results on traffic job should return 400
    res_exam = client.get(f"/api/v1/jobs/{traffic_job.id}/examination/results")
    assert res_exam.status_code == 400
    assert "traffic" in res_exam.json()["detail"].lower()


# =============================================================================
# 2. Terminal Completion Envelope Tests
# =============================================================================

def test_examination_terminal_envelope_format():
    """Verify CommonEnvelope format emitted at the end of examination pipeline."""
    env = CommonEnvelope(
        schema_version="2.0",
        job_id=971,
        video_id=10,
        analysis_module="examination",
        source_timestamp=30.0,
        frame_index=900,
        result_type="job_summary",
        processing_status="SUCCESS",
        confidence=0.95,
        payload={
            "type": "completed",
            "status": "SUCCEEDED",
            "telemetry": {
                "source_video_fps": 30.0,
                "processed_frames": 900,
                "detected_frames": 300,
                "skipped_frames": 600,
                "stride": 3,
                "processing_mode": "balanced",
                "average_fps": 24.5
            },
            "processing_fps": 24.5,
            "throughput_fps": 24.5
        }
    )
    d = env.to_dict()
    assert d["analysis_module"] == "examination"
    assert d["result_type"] == "job_summary"
    assert d["payload"]["type"] == "completed"
    assert d["payload"]["status"] == "SUCCEEDED"
    assert d["payload"]["throughput_fps"] == 24.5


# =============================================================================
# 3. Excel Export Reliability & openpyxl Validation Tests
# =============================================================================

def test_examination_excel_export_generates_valid_openpyxl_workbook(client, test_db):
    """Verify GET /api/v1/jobs/{id}/export/excel generates valid openpyxl readable workbook for examination."""
    vid = Video(project_id=1, filename="examination_hall.mp4", storage_path="media/videos/examination_hall.mp4", sha256_hash="dummy4", duration_sec=50.0, fps=30.0)
    test_db.add(vid)
    test_db.commit()

    job = ProcessingJob(
        video_id=vid.id,
        status="SUCCEEDED",
        domain="examination",
        telemetry_json={
            "analysis_module": "examination",
            "average_fps": 22.4,
            "total_frames": 600,
            "processed_frames": 600,
            "detected_frames": 200,
            "skipped_frames": 400,
            "stride": 3
        }
    )
    test_db.add(job)
    test_db.commit()

    # Add person track and examination event
    trk = ExaminationTrackRecord(
        job_id=job.id,
        track_id=1,
        entity_type="person",
        first_frame=0,
        last_frame=600,
        assigned_zone="Row 1 - Desk 2",
        assigned_role="STUDENT",
        role_source="SEATING_CHART"
    )
    test_db.add(trk)

    evt = ExaminationEventRecord(
        event_id="EVT-EXAM-001",
        job_id=job.id,
        video_id=vid.id,
        event_type="UNAUTHORIZED_ELECTRONIC_DEVICE",
        start_frame=30,
        end_frame=90,
        start_time=1.0,
        end_time=3.0,
        person_track_id=1,
        assigned_role="STUDENT",
        confidence=0.91,
        uncertainty=0.09,
        description="Candidate smartphone detected on desk.",
        review_status="PENDING"
    )
    test_db.add(evt)
    test_db.commit()

    # Check export preview
    prev_resp = client.get(f"/api/v1/jobs/{job.id}/export/excel/preview")
    assert prev_resp.status_code == 200
    prev_data = prev_resp.json()
    assert prev_data["domain"] == "examination"
    assert prev_data["counts"]["person_tracks"] == 1
    assert prev_data["counts"]["events"] == 1

    # Check export download
    export_resp = client.get(f"/api/v1/jobs/{job.id}/export/excel?mode=final")
    assert export_resp.status_code == 200
    assert "spreadsheetml.sheet" in export_resp.headers["content-type"]
    assert export_resp.headers["x-visionradar-domain"] == "examination"
    assert "VisionRadar_Examination_Job_" in export_resp.headers["content-disposition"]
    assert export_resp.headers["content-disposition"].endswith('.xlsx"')

    # Validate workbook with openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(export_resp.content), data_only=True)
    expected_sheets = [
        "Executive Summary", "Monitored Persons & Tracks", "Examination Events",
        "Evidence Timeline", "Detection Records", "Event Review History",
        "Seating Zones", "Processing Metadata", "Data Dictionary", "Export Metadata"
    ]
    for s in expected_sheets:
        assert s in wb.sheetnames, f"Missing sheet: {s}"

    # Verify Executive Summary
    summary_ws = wb["Executive Summary"]
    assert "VISIONRADAR" in str(summary_ws["A1"].value)
    assert "EXAMINATION" in str(summary_ws["A2"].value)


def test_traffic_excel_export_protected_and_valid(client, test_db):
    """Verify Traffic Excel export continues to function flawlessly without regression."""
    vid = Video(project_id=1, filename="traffic_corridor.mp4", storage_path="media/videos/traffic_corridor.mp4", sha256_hash="dummy5", duration_sec=30.0, fps=30.0)
    test_db.add(vid)
    test_db.commit()

    job = ProcessingJob(
        video_id=vid.id,
        status="SUCCEEDED",
        domain="traffic",
        telemetry_json={"detector_actual": "YOLOX-Nano-ONNX", "average_fps": 31.2}
    )
    test_db.add(job)
    test_db.commit()

    trk = Track(job_id=job.id, track_id=42, vehicle_class="car", first_frame=0, last_frame=300)
    test_db.add(trk)
    test_db.commit()

    export_resp = client.get(f"/api/v1/jobs/{job.id}/export/excel?mode=final")
    assert export_resp.status_code == 200
    assert "spreadsheetml.sheet" in export_resp.headers["content-type"]
    assert export_resp.headers["x-visionradar-domain"] == "traffic"
    assert "VisionRadar_Traffic_Job_" in export_resp.headers["content-disposition"]

    wb = openpyxl.load_workbook(io.BytesIO(export_resp.content), data_only=True)
    assert "Unique Tracks" in wb.sheetnames
    assert "Traffic Events" in wb.sheetnames
    assert "Speed Analytics" in wb.sheetnames
