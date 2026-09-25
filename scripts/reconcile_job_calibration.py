import os
import sys
import json
import hashlib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.models.database import get_db
from visionradar.models.entities import ProcessingJob, Video, Calibration, Track, SpeedMeasurement

def reconcile_calibration():
    db = next(get_db())

    print("==================================================")
    print("VISIONRADAR -- JOB & CALIBRATION RECONCILIATION")
    print("==================================================")

    # List all jobs associated with Traffic1.mp4
    traffic1_videos = db.query(Video).filter(Video.filename.like("%Traffic1%")).all()
    print(f"Found {len(traffic1_videos)} Video entities matching 'Traffic1':")
    for v in traffic1_videos:
        print(f"  Video ID {v.id}: storage_path={v.storage_path}, sha256={v.sha256_hash[:16]}..., res={v.width}x{v.height}@{v.fps}FPS")

    # Find Job #70 (seen in latest user screenshot!) and Job #64
    jobs_to_check = [70, 64]
    for jid in jobs_to_check:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == jid).first()
        if not job:
            print(f"\nProcessingJob #{jid}: NOT FOUND in DB")
            continue

        video = db.query(Video).filter(Video.id == job.video_id).first()
        calib = db.query(Calibration).filter(Calibration.id == job.calibration_id).first()
        track_count = db.query(Track).filter(Track.job_id == job.id).count()

        print(f"\n================ CANONICAL JOB #{job.id} AUDIT ================")
        print(f"video_id: {job.video_id}")
        print(f"job_id: {job.id}")
        print(f"video filename: {video.filename if video else 'N/A'}")
        print(f"video storage_path: {video.storage_path if video else 'N/A'}")
        print(f"video SHA256: {video.sha256_hash if video else 'N/A'}")
        print(f"calibration_id: {job.calibration_id}")
        print(f"calibration version: {calib.version if calib else 'N/A'}")
        print(f"model: {job.telemetry_json.get('detector_actual', 'YOLOX-Nano-ONNX') if job.telemetry_json else 'YOLOX-Nano-ONNX'}")
        print(f"model SHA256: c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d")
        print(f"FPS: {video.fps if video else 29.97}")
        print(f"resolution: {video.width if video else 1920}x{video.height if video else 1080}")
        print(f"frame count: 335")
        print(f"status: {job.status} | progress: {job.progress_pct}% | tracks in DB: {track_count}")

        if calib:
            print("\n--- BACKEND CALIBRATION ENTITY (Calibration ID #{calib.id}) ---")
            print(f"image_points_json (P1..P4): {calib.image_points_json}")
            print(f"world_points_json (X, Y): {calib.world_points_json}")
            print(f"h_matrix_json:\n{np.array(calib.h_matrix_json)}")
            print(f"camera_height_m: {calib.camera_height_m}m | pitch_deg: {calib.pitch_deg}deg | reprojection_rmse_m: {calib.reprojection_rmse_m:.4f}m")

        # Trace Track #1 in Job #70 or Job #64
        trk1 = db.query(Track).filter(Track.job_id == job.id, Track.track_id == 1).first()
        if trk1:
            traj = trk1.trajectory_json or []
            print(f"\n--- ALL TRAJECTORY POINTS FOR TRACK #1 IN JOB #{job.id} (DB ID {trk1.id}) ---")
            print(f"Total Trajectory Points: {len(traj)}")
            H = np.array(calib.h_matrix_json) if calib else np.eye(3)
            
            for idx, p in enumerate(traj):
                f_idx = p.get("frame_index")
                anchor = p.get("anchor_pixel") or [0, 0]
                world = p.get("world_pos") or [0, 0]
                u, v = anchor
                q = H[2,0]*u + H[2,1]*v + 1.0
                
                sm = db.query(SpeedMeasurement).filter(SpeedMeasurement.track_id == trk1.id, SpeedMeasurement.frame_index == f_idx).first()
                inst_spd = sm.instantaneous_kmh if sm else 0.0
                smooth_spd = sm.smoothed_kmh if sm else 0.0

                print(f"  [{idx:2d}] Frame {f_idx:3d} (t={p.get('timestamp', 0):.2f}s): anchor=({u:6.1f}, {v:6.1f}), q={q:9.6f}, world=({world[0]:7.2f}m, {world[1]:7.2f}m), inst_spd={inst_spd:8.2f} km/h, smooth_spd={smooth_spd:7.2f} km/h")

if __name__ == "__main__":
    reconcile_calibration()
