import pytest
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from visionradar.models.database import Base
from visionradar.models.entities import (
    Project, ProcessingJob, Video, ExaminationTrackRecord, ExaminationEventRecord, ExaminationZoneRecord
)
from visionradar.examination.audit_exporter import ForensicAuditExporter

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()

def test_audit_report_generation(test_db):
    proj = Project(id=1, name="Exam Project")
    test_db.add(proj)
    test_db.commit()

    video = Video(
        project_id=1,
        filename="exam_session_01.mp4",
        storage_path="data/videos/exam_session_01.mp4",
        duration_sec=60.0,
        fps=30.0,
        width=1920,
        height=1080,
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    test_db.add(video)
    test_db.commit()

    job = ProcessingJob(
        video_id=video.id,
        status="SUCCEEDED",
        stage="Completed",
        progress_pct=100.0,
        config_json={"module": "examination", "processing_mode": "balanced"},
        telemetry_json={"stride": 2, "avg_fps": 18.2}
    )
    test_db.add(job)
    test_db.commit()

    track = ExaminationTrackRecord(
        job_id=job.id,
        track_id=1,
        entity_type="person",
        first_frame=0,
        last_frame=600,
        assigned_zone="Seat-01",
        trajectory_json=[]
    )
    test_db.add(track)

    event = ExaminationEventRecord(
        job_id=job.id,
        video_id=video.id,
        event_id="evt_phone_test001",
        event_type="POSSIBLE_PHONE_INTERACTION",
        start_frame=30,
        end_frame=90,
        start_time=1.0,
        end_time=3.0,
        person_track_id=1,
        confidence=0.88,
        uncertainty=0.12,
        supporting_objects_json={"priority": "RED", "evidence_strength": "HIGH", "supporting_objects": ["cell phone"]},
        description="Candidate phone interaction near Track #1",
        review_status="PENDING"
    )
    test_db.add(event)

    zone = ExaminationZoneRecord(
        video_id=video.id,
        zone_id="Seat-01",
        name="Desk Row 1",
        zone_type="SEATING_ZONE",
        polygon_json=[[100, 100], [200, 100], [200, 200], [100, 200]]
    )
    test_db.add(zone)
    test_db.commit()

    report = ForensicAuditExporter.generate_audit_report(job.id, test_db)

    assert report["job_id"] == job.id
    assert report["video_metadata"]["filename"] == "exam_session_01.mp4"
    assert report["summary_metrics"]["total_person_tracks"] == 1
    assert report["summary_metrics"]["total_candidate_events"] == 1
    assert report["summary_metrics"]["priority_breakdown"]["RED"] == 1
    assert "VR-SEAL-" in report["cryptographic_seal"]
    assert report["seal_algorithm"] == "SHA-256 / Canonical JSON"

    # Verify HTML generation
    html = ForensicAuditExporter.generate_audit_html(report)
    assert "<!DOCTYPE html>" in html
    assert "VisionRadar Examination Forensic Audit Report" in html
    assert report["cryptographic_seal"] in html
    assert "POSSIBLE_PHONE_INTERACTION" in html
    assert "Print / Save as PDF" in html
