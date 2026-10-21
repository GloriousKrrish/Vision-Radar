"""
VisionRadar Phase 5 Automated Validation Test Suite:
Enterprise Excel Intelligence Exports & Cross-Workbench Shared State.
Tests workbook generation, multi-sheet integrity, formatting, domain segregation,
snapshot vs final modes, data dictionaries, error handling, and API endpoints.
"""

import os
import io
import pytest
import openpyxl
from fastapi.testclient import TestClient

from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Video, Track, Violation, SpeedMeasurement, ExaminationTrackRecord, ExaminationEventRecord
from visionradar.services.excel_export import ExcelExportEngine

client = TestClient(app)

@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_excel_export_traffic_job(db_session):
    """Verify complete enterprise Excel export for a traffic job."""
    # Find a traffic job or fallback to Job 733
    job = db_session.query(ProcessingJob).filter(ProcessingJob.id == 733).first()
    if not job:
        job = db_session.query(ProcessingJob).filter(ProcessingJob.status == "SUCCEEDED").first()

    assert job is not None, "Test requires at least one existing job"

    stream = ExcelExportEngine.generate_workbook(job.id, db_session, mode="final")
    assert stream is not None
    data = stream.getvalue()
    assert len(data) > 1000, "Excel output file should contain data"

    # Open with openpyxl to validate workbook integrity
    wb = openpyxl.load_workbook(io.BytesIO(data))
    sheet_names = wb.sheetnames

    # Check common sheets
    assert "Executive Summary" in sheet_names
    assert "Processing Metadata" in sheet_names
    assert "Data Dictionary" in sheet_names
    assert "Export Metadata" in sheet_names

    # Check traffic domain sheets
    assert "Unique Tracks" in sheet_names
    assert "Traffic Events" in sheet_names
    assert "Speed Analytics" in sheet_names

    # Validate Executive Summary contents
    ws_exec = wb["Executive Summary"]
    assert "VISIONRADAR — UNIVERSAL VIDEO INTELLIGENCE PLATFORM" in str(ws_exec["A1"].value)

    # Validate Data Dictionary
    ws_dict = wb["Data Dictionary"]
    assert ws_dict.max_row >= 8, "Data dictionary should contain field definitions"
    headers = [cell.value for cell in ws_dict[1]]
    assert "Field / Column" in headers
    assert "Operational Meaning & Semantics" in headers

    # Validate Export Metadata
    ws_meta = wb["Export Metadata"]
    meta_keys = [ws_meta.cell(row=r, column=1).value for r in range(2, ws_meta.max_row + 1)]
    assert "Cryptographic Export Seal" in meta_keys
    assert "Export Mode" in meta_keys


def test_excel_export_examination_job(db_session):
    """Verify complete enterprise Excel export for an examination job."""
    # Find examination job (e.g. Job 732)
    job = db_session.query(ProcessingJob).filter(ProcessingJob.id == 732).first()
    if not job:
        all_jobs = db_session.query(ProcessingJob).all()
        for j in all_jobs:
            if "exam" in str(j.config_json or {}):
                job = j
                break

    if not job:
        pytest.skip("No examination job in database to evaluate")

    stream = ExcelExportEngine.generate_workbook(job.id, db_session, mode="final")
    wb = openpyxl.load_workbook(stream)
    sheet_names = wb.sheetnames

    # Verify domain segregation
    assert "Monitored Persons & Tracks" in sheet_names
    assert "Examination Events" in sheet_names
    # Traffic sheets must NOT be in examination export
    assert "Speed Analytics" not in sheet_names


def test_excel_export_modes_final_vs_snapshot(db_session):
    """Verify live snapshot and final export produce distinct mode labels and metadata."""
    job = db_session.query(ProcessingJob).filter(ProcessingJob.id == 733).first()
    if not job:
        job = db_session.query(ProcessingJob).first()

    # Final Mode
    stream_final = ExcelExportEngine.generate_workbook(job.id, db_session, mode="final")
    wb_final = openpyxl.load_workbook(stream_final)
    ws_meta_final = wb_final["Export Metadata"]
    mode_val_final = None
    for r in range(2, ws_meta_final.max_row + 1):
        if ws_meta_final.cell(row=r, column=1).value == "Export Mode":
            mode_val_final = ws_meta_final.cell(row=r, column=2).value

    assert mode_val_final == "FINAL"

    # Snapshot Mode
    stream_snap = ExcelExportEngine.generate_workbook(job.id, db_session, mode="snapshot")
    wb_snap = openpyxl.load_workbook(stream_snap)
    ws_meta_snap = wb_snap["Export Metadata"]
    mode_val_snap = None
    for r in range(2, ws_meta_snap.max_row + 1):
        if ws_meta_snap.cell(row=r, column=1).value == "Export Mode":
            mode_val_snap = ws_meta_snap.cell(row=r, column=2).value

    assert mode_val_snap == "SNAPSHOT"


def test_excel_export_empty_dataset_handling(db_session):
    """Verify that an empty job without tracks/events generates without errors."""
    # Create a synthetic empty job record for test
    video = db_session.query(Video).first()
    if not video:
        pytest.skip("No video in database")

    empty_job = ProcessingJob(
        video_id=video.id,
        status="SUCCEEDED",
        stage="Completed",
        progress_pct=100.0,
        config_json={"module": "traffic"}
    )
    db_session.add(empty_job)
    db_session.commit()
    db_session.refresh(empty_job)

    try:
        stream = ExcelExportEngine.generate_workbook(empty_job.id, db_session, mode="final")
        assert stream is not None
        wb = openpyxl.load_workbook(stream)
        assert "Executive Summary" in wb.sheetnames
        assert "Unique Tracks" in wb.sheetnames
        assert "Data Dictionary" in wb.sheetnames
    finally:
        db_session.delete(empty_job)
        db_session.commit()


def test_api_excel_export_preview_endpoint():
    """Verify GET /api/v1/jobs/{job_id}/export/excel/preview returns structured metadata."""
    resp = client.get("/api/v1/jobs/733/export/excel/preview")
    if resp.status_code == 404:
        # Fallback to latest job
        resp_latest = client.get("/api/v1/jobs/latest")
        if resp_latest.status_code == 200:
            jid = resp_latest.json()["id"]
            resp = client.get(f"/api/v1/jobs/{jid}/export/excel/preview")

    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert "domain" in data
    assert "counts" in data
    assert "planned_sheets" in data
    assert len(data["planned_sheets"]) >= 5


def test_api_excel_export_download_endpoint():
    """Verify GET /api/v1/jobs/{job_id}/export/excel returns application/vnd.openxmlformats."""
    resp = client.get("/api/v1/jobs/733/export/excel?mode=final")
    if resp.status_code == 404:
        resp_latest = client.get("/api/v1/jobs/latest")
        if resp_latest.status_code == 200:
            jid = resp_latest.json()["id"]
            resp = client.get(f"/api/v1/jobs/{jid}/export/excel?mode=final")

    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    assert "spreadsheetml.sheet" in content_type or "application/vnd" in content_type
    disposition = resp.headers.get("content-disposition", "")
    assert "attachment" in disposition
    assert ".xlsx" in disposition

    # Validate that downloaded payload is a valid zip/openpyxl file
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    assert len(wb.sheetnames) >= 4
