import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("packages"))
from visionradar.models.database import SessionLocal
from visionradar.models.init_db import create_tables
from visionradar.models.entities import Project, Video, Calibration, ProcessingJob, Experiment
from visionradar.cv.calibration import HomographyCalibrator

def seed_database():
    create_tables()
    db = SessionLocal()

    # Seed Project
    project = db.query(Project).filter(Project.name == "Highway Speed & Flow Study").first()
    if not project:
        project = Project(
            name="Highway Speed & Flow Study",
            description="Monocular speed estimation benchmark on 3-lane synthetic highway video."
        )
        db.add(project)
        db.commit()
        db.refresh(project)

    # Seed Video
    video = db.query(Video).filter(Video.filename == "synthetic_highway.mp4").first()
    if not video:
        video = Video(
            project_id=project.id,
            filename="synthetic_highway.mp4",
            storage_path="data/videos/synthetic_highway.mp4",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            duration_sec=10.0,
            fps=30.0,
            width=800,
            height=450,
            codec="h264"
        )
        db.add(video)
        db.commit()
        db.refresh(video)

    # Seed Calibration
    calib = db.query(Calibration).filter(Calibration.video_id == video.id).first()
    if not calib:
        img_pts = [(330.0, 160.0), (470.0, 160.0), (748.0, 435.0), (51.0, 435.0)]
        world_pts = [(-1.0, 150.0), (13.0, 150.0), (13.0, 0.5), (-1.0, 0.5)]
        calibrator = HomographyCalibrator(image_points=img_pts, world_points=world_pts)

        calib = Calibration(
            video_id=video.id,
            version=1,
            h_matrix_json=calibrator.H.tolist(),
            image_points_json=img_pts,
            world_points_json=world_pts,
            reprojection_rmse_m=calibrator.compute_reprojection_rmse(),
            camera_height_m=9.0,
            pitch_deg=14.0
        )
        db.add(calib)
        db.commit()
        db.refresh(calib)

    # Seed Job
    job = db.query(ProcessingJob).filter(ProcessingJob.video_id == video.id).first()
    if not job:
        job = ProcessingJob(
            video_id=video.id,
            calibration_id=calib.id,
            status="QUEUED",
            stage="Queued",
            progress_pct=0.0
        )
        db.add(job)
        db.commit()

    # Seed Experiments
    if db.query(Experiment).count() == 0:
        b1 = Experiment(
            name="BrnoCompSpeed Benchmark",
            dataset_name="BrnoCompSpeed",
            detector_name="YOLOX-Nano",
            tracker_name="ByteTrack",
            speed_method="Linear Regression",
            mae_kmh=1.92,
            rmse_kmh=2.45,
            r2_score=0.9812
        )
        b2 = Experiment(
            name="UA-DETRAC Benchmark",
            dataset_name="UA-DETRAC",
            detector_name="YOLOv8-Small",
            tracker_name="SORT",
            speed_method="Instantaneous DLT",
            mae_kmh=3.41,
            rmse_kmh=4.12,
            r2_score=0.9450
        )
        b3 = Experiment(
            name="Synthetic Highway Baseline",
            dataset_name="Synthetic Highway",
            detector_name="LightweightDetector",
            tracker_name="ByteTrack",
            speed_method="Windowed Linear Regression",
            mae_kmh=0.35,
            rmse_kmh=0.48,
            r2_score=0.9985
        )
        db.add_all([b1, b2, b3])
        db.commit()

    db.close()
    print("Database successfully seeded with initial project, video, calibration, and benchmark experiments.")

if __name__ == "__main__":
    seed_database()
