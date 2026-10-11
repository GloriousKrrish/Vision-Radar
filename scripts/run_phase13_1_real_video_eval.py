import os
import sys
import json
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

from visionradar.worker.job_worker import run_job_streaming, SessionLocal
from visionradar.models.entities import ProcessingJob, Video, Project
import asyncio

def run_evaluation():
    print("==================================================")
    print(" PHASE 13.1 — LIVE ACCIDENT INTEGRATION EVALUATION")
    print("==================================================")

    db = SessionLocal()
    project = db.query(Project).first()
    if not project:
        project = Project(name="Phase 13.1 Audit", description="Real video evaluation")
        db.add(project)
        db.commit()
        db.refresh(project)

    # 1. Define candidate video files to evaluate
    candidate_videos = [
        "Traffic1.mp4",
        "data/videos/Traffic1.mp4",
        "DATASETS/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4",
        "DATASETS/Vehicle_Tracking/AICC22-Custom/AICC22-Custom_Original-video.mp4"
    ]

    evaluated_results = []

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for v_path in candidate_videos:
        if not os.path.exists(v_path):
            continue

        print(f"\n--- Processing Video: {v_path} ---")
        
        # Read metadata via OpenCV
        cap = cv2.VideoCapture(v_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = float(frames) / fps if fps > 0 else 10.0
        cap.release()

        # Insert DB Video record
        vid = Video(
            project_id=project.id,
            filename=os.path.basename(v_path),
            storage_path=v_path,
            sha256_hash="phase13_1_eval_hash",
            duration_sec=round(dur, 2),
            fps=round(fps, 2),
            width=w,
            height=h,
            codec="h264"
        )
        db.add(vid)
        db.commit()
        db.refresh(vid)

        # Create processing job
        job = ProcessingJob(
            video_id=vid.id,
            calibration_id=None,
            config_json={"detector": "yolox"},
            status="QUEUED",
            stage="Queued",
            progress_pct=0.0
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        t0 = time.time()
        run_job_streaming(job.id, loop)
        t1 = time.time()

        # Refresh job details
        db.refresh(job)
        telemetry = job.telemetry_json or {}
        acc_intel = telemetry.get("accident_intelligence") or {}

        res_item = {
            "video_path": v_path,
            "job_id": job.id,
            "status": job.status,
            "duration_sec": round(t1 - t0, 2),
            "frames_processed": telemetry.get("frames_processed", 0),
            "total_tracks": telemetry.get("total_tracks", 0),
            "classifier_loaded": acc_intel.get("accident_classifier_loaded", False),
            "classifier_invocations": acc_intel.get("accident_classifier_invocations", 0),
            "candidate_frames_classified": acc_intel.get("candidate_frames_classified", 0),
            "normal_predictions": acc_intel.get("normal_predictions", 0),
            "accident_predictions": acc_intel.get("accident_predictions", 0),
            "total_accident_candidates": len(acc_intel.get("accident_candidates", [])),
            "accident_candidates": acc_intel.get("accident_candidates", []),
            "total_classifier_ms": acc_intel.get("total_classifier_ms", 0.0)
        }
        evaluated_results.append(res_item)

        print(f"Job #{job.id} completed ({job.status}):")
        print(f"  Frames Processed: {res_item['frames_processed']}")
        print(f"  Tracks Tracked: {res_item['total_tracks']}")
        print(f"  Classifier Invocations: {res_item['classifier_invocations']}")
        print(f"  Candidate Frames Classified: {res_item['candidate_frames_classified']}")
        print(f"  Normal Predictions: {res_item['normal_predictions']}")
        print(f"  Accident Predictions: {res_item['accident_predictions']}")
        print(f"  Accident Candidates Flagged: {res_item['total_accident_candidates']}")

    db.close()
    
    # Save overall evaluation report
    os.makedirs("reports/phase13", exist_ok=True)
    with open("reports/phase13/phase13_1_video_eval_results.json", "w") as f:
        json.dump(evaluated_results, f, indent=2)
        
    print("\nSaved evaluation results to reports/phase13/phase13_1_video_eval_results.json")

if __name__ == "__main__":
    run_evaluation()
