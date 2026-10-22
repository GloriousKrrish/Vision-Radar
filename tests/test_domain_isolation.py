import pytest
from fastapi.testclient import TestClient
from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Video, Calibration, Track, ExaminationEventRecord
from visionradar.services.excel_export import ExcelExportEngine
import openpyxl
import io

client = TestClient(app)

@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_missing_or_invalid_domain_rejected_with_400():
    """Requirement 16: Ensure an invalid or missing domain fails explicitly instead of defaulting to Traffic."""
    # 1. Missing domain completely
    res_missing = client.post("/api/v1/videos/1/jobs", json={"calibration_id": None})
    assert res_missing.status_code == 400
    assert "Invalid or missing domain" in res_missing.json()["detail"]

    # 2. Invalid/unrecognized domain
    res_invalid = client.post("/api/v1/videos/1/jobs", json={"domain": "medical", "calibration_id": None})
    assert res_invalid.status_code == 400
    assert "Explicit domain ('traffic' or 'examination') is strictly required" in res_invalid.json()["detail"]

def test_create_examination_job_persists_domain_and_omits_calibration(db_session):
    """Requirement 1, 2, 3: Create examination job, verify persisted domain and no traffic calibration."""
    # Create synthetic video if needed
    v = db_session.query(Video).first()
    if not v:
        pytest.skip("No video available in database to test job creation")

    # Post job with domain="examination"
    res = client.post(f"/api/v1/videos/{v.id}/jobs", json={
        "domain": "examination",
        "calibration_id": 9999 # Attempting to bind calibration to examination
    })
    assert res.status_code == 200
    job_data = res.json()
    job_id = job_data["id"]

    # Authoritative response check
    assert job_data["domain"] == "examination"
    assert job_data["calibration_id"] is None # Traffic calibration must be stripped

    # Database record check
    db_job = db_session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    assert db_job is not None
    assert db_job.domain == "examination"
    assert db_job.calibration_id is None
    assert db_job.config_json.get("domain") == "examination"

def test_create_traffic_job_persists_traffic_domain(db_session):
    """Requirement 6: Create traffic job and verify domain is persisted as traffic."""
    v = db_session.query(Video).first()
    if not v:
        pytest.skip("No video available in database to test job creation")

    res = client.post(f"/api/v1/videos/{v.id}/jobs", json={
        "domain": "traffic"
    })
    assert res.status_code == 200
    job_data = res.json()
    job_id = job_data["id"]

    assert job_data["domain"] == "traffic"

    db_job = db_session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    assert db_job is not None
    assert db_job.domain == "traffic"
    assert db_job.config_json.get("domain") == "traffic"

def test_domain_guard_blocks_cross_domain_track_and_speed_requests(db_session):
    """Requirement 11: Traffic endpoints return 400 if accessed for an examination job."""
    # Find an examination job or create a mock one
    exam_job = db_session.query(ProcessingJob).filter(ProcessingJob.domain == "examination").first()
    if not exam_job:
        exam_job = ProcessingJob(
            video_id=1,
            domain="examination",
            status="SUCCEEDED",
            stage="Done",
            progress_pct=100.0,
            config_json={"domain": "examination"}
        )
        db_session.add(exam_job)
        db_session.commit()
        db_session.refresh(exam_job)

    # 1. /tracks
    res_tracks = client.get(f"/api/v1/jobs/{exam_job.id}/tracks")
    assert res_tracks.status_code == 400
    assert "belongs to 'examination' domain" in res_tracks.json()["detail"]
    assert "Vehicle tracks are only available for traffic jobs" in res_tracks.json()["detail"]

    # 2. /vehicle-speeds
    res_speeds = client.get(f"/api/v1/jobs/{exam_job.id}/vehicle-speeds")
    assert res_speeds.status_code == 400
    assert "Vehicle speeds are only available for traffic jobs" in res_speeds.json()["detail"]

def test_domain_guard_blocks_cross_domain_examination_requests(db_session):
    """Requirement 12: Examination endpoints return 400 if accessed for a traffic job."""
    traffic_job = db_session.query(ProcessingJob).filter(ProcessingJob.domain == "traffic").first()
    if not traffic_job:
        traffic_job = ProcessingJob(
            video_id=1,
            domain="traffic",
            status="SUCCEEDED",
            stage="Done",
            progress_pct=100.0,
            config_json={"domain": "traffic"}
        )
        db_session.add(traffic_job)
        db_session.commit()
        db_session.refresh(traffic_job)

    # 1. /examination/results
    res_exam_res = client.get(f"/api/v1/jobs/{traffic_job.id}/examination/results")
    assert res_exam_res.status_code == 400
    assert "belongs to 'traffic' domain" in res_exam_res.json()["detail"]

    # 2. /examination/report
    res_exam_rep = client.get(f"/api/v1/jobs/{traffic_job.id}/examination/report")
    assert res_exam_rep.status_code == 400
    assert "belongs to 'traffic' domain" in res_exam_rep.json()["detail"]

    # 3. /examination/events
    res_exam_evt = client.get(f"/api/v1/jobs/{traffic_job.id}/examination/events")
    assert res_exam_evt.status_code == 400
    assert "belongs to 'traffic' domain" in res_exam_evt.json()["detail"]

def test_excel_export_preview_derives_domain_from_persisted_job(db_session):
    """Requirement 15: Excel preview uses persisted job domain and rejects domain tampering."""
    exam_job = db_session.query(ProcessingJob).filter(ProcessingJob.domain == "examination").first()
    traffic_job = db_session.query(ProcessingJob).filter(ProcessingJob.domain == "traffic").first()

    if exam_job:
        # Requesting preview without query param
        res_e = client.get(f"/api/v1/jobs/{exam_job.id}/export/excel/preview")
        assert res_e.status_code == 200
        data_e = res_e.json()
        assert data_e["domain"] == "examination"
        sheet_names_e = data_e["planned_sheets"]
        assert "Monitored Persons & Tracks" in sheet_names_e
        assert "Examination Events" in sheet_names_e
        assert "Unique Tracks" not in sheet_names_e
        assert "Speed Analytics" not in sheet_names_e

        # Attempting query parameter override
        res_e_tamper = client.get(f"/api/v1/jobs/{exam_job.id}/export/excel/preview?domain=traffic")
        assert res_e_tamper.status_code == 200
        # Authoritative backend must enforce persisted domain
        assert res_e_tamper.json()["domain"] == "examination"

    if traffic_job:
        res_t = client.get(f"/api/v1/jobs/{traffic_job.id}/export/excel/preview")
        assert res_t.status_code == 200
        data_t = res_t.json()
        assert data_t["domain"] == "traffic"
        sheet_names_t = data_t["planned_sheets"]
        assert "Unique Tracks" in sheet_names_t
        assert "Speed Analytics" in sheet_names_t
        assert "Monitored Persons & Tracks" not in sheet_names_t
        assert "Examination Events" not in sheet_names_t

def test_excel_export_download_generates_domain_isolated_workbooks(db_session):
    """Requirement 15 & Phase G: Excel binary workbook sheets match authoritative job domain."""
    exam_job = db_session.query(ProcessingJob).filter(ProcessingJob.domain == "examination").first()
    traffic_job = db_session.query(ProcessingJob).filter(ProcessingJob.domain == "traffic").first()

    if exam_job:
        res = client.get(f"/api/v1/jobs/{exam_job.id}/export/excel?mode=final")
        assert res.status_code == 200
        wb = openpyxl.load_workbook(io.BytesIO(res.content))
        sheets = wb.sheetnames
        assert "Monitored Persons & Tracks" in sheets
        assert "Examination Events" in sheets
        assert "Unique Tracks" not in sheets
        assert "Speed Analytics" not in sheets

    if traffic_job:
        res = client.get(f"/api/v1/jobs/{traffic_job.id}/export/excel?mode=final")
        assert res.status_code == 200
        wb = openpyxl.load_workbook(io.BytesIO(res.content))
        sheets = wb.sheetnames
        assert "Executive Summary" in sheets
        assert "Unique Tracks" in sheets
        assert "Speed Analytics" in sheets
        assert "Monitored Persons & Tracks" not in sheets
        assert "Examination Events" not in sheets

def test_job_switching_and_isolation(db_session):
    """Requirement 7, 8: Switching between traffic and examination jobs guarantees isolated domain metadata."""
    jobs = client.get("/api/v1/jobs").json()
    assert isinstance(jobs, list)
    for j in jobs:
        assert "domain" in j
        assert j["domain"] in ("traffic", "examination")

def test_backward_compatibility_job_entity_defaults(db_session):
    """Requirement 17: ProcessingJob entity defaults domain to 'traffic' when added/flushed."""
    legacy_job = ProcessingJob(
        video_id=1,
        status="SUCCEEDED",
        stage="Done"
    )
    db_session.add(legacy_job)
    db_session.flush()
    # Model default is 'traffic'
    assert legacy_job.domain == "traffic"
    db_session.rollback()
