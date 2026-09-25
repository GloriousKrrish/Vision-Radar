import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from apps.api.main import app
from visionradar.models.database import get_db, SessionLocal
from visionradar.models.entities import ProcessingJob, Video, Track, SpeedMeasurement

client = TestClient(app)

def test_job_specific_track_isolation_and_lifecycle():
    db = SessionLocal()
    try:
        # Create Video record
        video = Video(
            project_id=1,
            filename="LifecycleTest.mp4",
            storage_path="data/videos/LifecycleTest.mp4",
            sha256_hash="lifecycle_hash_123",
            duration_sec=10.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264"
        )
        db.add(video)
        db.commit()

        # Job A: SUCCEEDED with 2 tracks
        job_a = ProcessingJob(
            video_id=video.id,
            calibration_id=1,
            status="SUCCEEDED",
            stage="Complete",
            progress_pct=100.0
        )
        db.add(job_a)
        db.commit()

        track_a1 = Track(
            job_id=job_a.id,
            track_id=101,
            vehicle_class="Car",
            confidence=0.95,
            first_frame=0,
            last_frame=10,
            trajectory_json=[{"frame_index": 5, "bbox": [100, 100, 200, 200], "anchor_pixel": [150, 200]}]
        )
        db.add(track_a1)
        db.commit()

        # Job B: RUNNING with 0 tracks persisted
        job_b = ProcessingJob(
            video_id=video.id,
            calibration_id=1,
            status="RUNNING",
            stage="Detection & Tracking",
            progress_pct=40.0
        )
        db.add(job_b)
        db.commit()

        # 1. Query Job B status & tracks while RUNNING
        res_b_status = client.get(f"/api/v1/jobs/{job_b.id}")
        assert res_b_status.status_code == 200
        data_b = res_b_status.json()
        assert data_b["status"] == "RUNNING"
        assert data_b["progress_pct"] == 40.0

        res_b_tracks = client.get(f"/api/v1/jobs/{job_b.id}/tracks")
        assert res_b_tracks.status_code == 200
        tracks_b_list = res_b_tracks.json()
        # RUNNING Job B MUST HAVE ZERO TRACKS
        assert len(tracks_b_list) == 0

        # Verify Job A tracks are isolated to Job A
        res_a_tracks = client.get(f"/api/v1/jobs/{job_a.id}/tracks")
        assert res_a_tracks.status_code == 200
        tracks_a_list = res_a_tracks.json()
        assert len(tracks_a_list) == 1
        assert tracks_a_list[0]["track_id"] == 101

        # 2. Transition Job B to SUCCEEDED and persist Job B tracks
        job_b.status = "SUCCEEDED"
        job_b.stage = "Complete"
        job_b.progress_pct = 100.0

        track_b1 = Track(
            job_id=job_b.id,
            track_id=202,
            vehicle_class="Truck",
            confidence=0.98,
            first_frame=0,
            last_frame=15,
            trajectory_json=[{"frame_index": 5, "bbox": [300, 300, 450, 450], "anchor_pixel": [375, 450]}]
        )
        db.add(track_b1)
        db.commit()

        # 3. Query Job B tracks after SUCCEEDED
        res_b_tracks_after = client.get(f"/api/v1/jobs/{job_b.id}/tracks")
        assert res_b_tracks_after.status_code == 200
        tracks_b_after = res_b_tracks_after.json()
        assert len(tracks_b_after) == 1
        assert tracks_b_after[0]["track_id"] == 202
        # Ensure Job B tracks contain ONLY track 202 and NOT track 101 from Job A
        assert all(t["track_id"] != 101 for t in tracks_b_after)

    finally:
        db.close()
