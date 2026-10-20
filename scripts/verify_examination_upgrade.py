import sys
import os
sys.path.insert(0, "packages")
sys.path.insert(0, ".")

import datetime
from visionradar.models.database import SessionLocal
from visionradar.models.entities import Video, ProcessingJob, ExaminationEventRecord, ExaminationTrackRecord
from visionradar.examination.pipeline import run_examination_job

def verify():
    db = SessionLocal()
    # Find video for exam cheat
    video = db.query(Video).filter(Video.filename.like("%cheat%")).order_by(Video.id.desc()).first()
    if not video:
        print("Error: No exam video found in DB")
        return False

    print(f"Target Examination Video: ID={video.id}, File={video.filename}, Duration={video.duration_sec}s, Path={video.storage_path}")

    # Create a fresh test job
    job = ProcessingJob(
        video_id=video.id,
        status="PENDING",
        stage="Queued",
        progress_pct=0.0,
        config_json={"module": "examination"}
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id
    print(f"Created Verification Job #{job_id} for Video #{video.id}")
    db.close()

    # Execute the upgraded examination pipeline
    print("Running upgraded examination pipeline...")
    run_examination_job(job_id)

    # Inspect results
    db = SessionLocal()
    completed_job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    print(f"Job #{job_id} status: {completed_job.status}, stage: {completed_job.stage}, progress: {completed_job.progress_pct}%")

    tracks = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == job_id).all()
    events = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job_id).all()

    print(f"Monitored Tracks: {len(tracks)}")
    for t in tracks:
        print(f"  Track #{t.track_id}: type={t.entity_type}, frames={t.first_frame}-{t.last_frame}, assigned_zone={t.assigned_zone}")

    print(f"\nCandidate Events Emitted: {len(events)}")
    for e in events:
        details = e.supporting_objects_json if isinstance(e.supporting_objects_json, dict) else {}
        print(f"  Event [{e.event_type}]:")
        print(f"    Event ID: {e.event_id}")
        print(f"    Person Track: #{e.person_track_id}")
        print(f"    Time: {e.start_time:.1f}s - {e.end_time:.1f}s (Frames #{e.start_frame} - #{e.end_frame})")
        print(f"    Priority: {details.get('priority')}")
        print(f"    Evidence Strength: {details.get('evidence_strength')} (Score: {details.get('event_confidence')})")
        print(f"    Supporting Frames: {details.get('supporting_frames')}")
        print(f"    Primary Finding: {details.get('primary_finding')}")
        print(f"    Observations: {details.get('supporting_observations')}")
        print(f"    Description: {e.description}")

    # Check for presence of phone interaction
    phone_evts = [e for e in events if e.event_type == "POSSIBLE_PHONE_INTERACTION"]
    leaving_evts = [e for e in events if e.event_type == "LEAVING_SEATING_ZONE"]

    print("\n--- Empirical Verification Summary ---")
    print(f"POSSIBLE_PHONE_INTERACTION detected: {len(phone_evts) > 0} (Count: {len(phone_evts)})")
    print(f"Spurious LEAVING_SEATING_ZONE eliminated: {len(leaving_evts) == 0} (Count: {len(leaving_evts)})")

    # Test report endpoint
    from fastapi.testclient import TestClient
    from apps.api.main import app
    client = TestClient(app)
    rep_res = client.get(f"/api/v1/jobs/{job_id}/examination/report")
    print(f"Report endpoint status: {rep_res.status_code}")
    if rep_res.status_code == 200:
        rep_data = rep_res.json()
        print(f"Report ID: {rep_data.get('report_id')}")
        print(f"Primary Candidate Finding: {rep_data.get('primary_candidate_finding', {}).get('event_type')}")
        print(f"Ethical disclosure present: {'proctor_in_the_loop' in rep_data.get('ethical_and_privacy_disclosures', {})}")

    db.close()
    return len(phone_evts) > 0 and len(leaving_evts) == 0

if __name__ == "__main__":
    success = verify()
    print(f"\nVerification Result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
