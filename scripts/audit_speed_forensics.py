import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.models.database import get_db
from visionradar.models.entities import ProcessingJob, Video, Calibration, Track, SpeedMeasurement
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.speed import MonocularSpeedEstimator

def audit_runtime():
    db = next(get_db())
    
    # 1. IDENTIFY EXACT CURRENT JOB & VIDEO
    job = db.query(ProcessingJob).filter(ProcessingJob.status == "SUCCEEDED").order_by(ProcessingJob.id.desc()).first()
    if not job:
        print("ERROR: No succeeded job found")
        return

    video = db.query(Video).filter(Video.id == job.video_id).first()
    calib = db.query(Calibration).filter(Calibration.id == job.calibration_id).first()

    print("==================================================")
    print("VISIONRADAR — SCREENSHOT FORENSIC RUNTIME AUDIT")
    print("==================================================")
    print(f"Job ID: {job.id}")
    print(f"Video ID: {video.id} | Filename: {video.filename}")
    print(f"Resolution: {video.width}x{video.height} @ {video.fps:.2f} FPS")
    print(f"Duration: {video.duration_sec:.2f}s | Codec: {video.codec}")
    print(f"Calibration ID: {calib.id if calib else 'None'}")

    # 2. CALIBRATION MATRIX & CONTROL POINTS
    if calib:
        H = np.array(calib.h_matrix_json)
        print("\n--- 2. CALIBRATION MATRIX & CONTROL POINTS ---")
        print(f"Image Points (P1..P4): {calib.image_points_json}")
        print(f"World Points (X, Y): {calib.world_points_json}")
        print(f"H-Matrix:\n{H}")
        print(f"Reprojection RMSE: {calib.reprojection_rmse_m:.4f} meters")
        print(f"Camera Height: {calib.camera_height_m}m | Pitch: {calib.pitch_deg}deg")
    else:
        H = None

    # 3. INSPECT EXACT TRACK DATA (#1, #12, #14, #15)
    target_track_ids = [1, 12, 14, 15]
    print("\n--- 3. EXACT TRACK DATA INVESTIGATION ---")
    
    for tid in target_track_ids:
        trk = db.query(Track).filter(Track.job_id == job.id, Track.track_id == tid).first()
        if not trk:
            # Fallback search any track with track_id == tid
            trk = db.query(Track).filter(Track.track_id == tid).order_by(Track.id.desc()).first()

        if not trk:
            print(f"\nTrack #{tid}: NOT FOUND in DB")
            continue

        traj = trk.trajectory_json or []
        speeds = db.query(SpeedMeasurement).filter(SpeedMeasurement.track_id == trk.id).order_by(SpeedMeasurement.frame_index.asc()).all()

        print(f"\n================ TRACK #{trk.track_id} (DB ID: {trk.id}) ================")
        print(f"Class: {trk.vehicle_class} | Confidence: {trk.confidence:.4f}")
        print(f"First Frame: {trk.first_frame} | Last Frame: {trk.last_frame} | Total Points: {len(traj)}")

        if traj:
            p_first = traj[0]
            p_last = traj[-1]
            print(f"First Point (f={p_first.get('frame_index')}, t={p_first.get('timestamp')}s): pixel={p_first.get('anchor_pixel')}, world={p_first.get('world_pos')}, bbox={p_first.get('bbox')}")
            print(f"Last Point  (f={p_last.get('frame_index')}, t={p_last.get('timestamp')}s): pixel={p_last.get('anchor_pixel')}, world={p_last.get('world_pos')}, bbox={p_last.get('bbox')}")

            # Compute pixel displacement and world displacement
            u1, v1 = p_first.get('anchor_pixel') or [0, 0]
            u2, v2 = p_last.get('anchor_pixel') or [0, 0]
            pix_dist = np.hypot(u2 - u1, v2 - v1)

            w1 = p_first.get('world_pos') or [0, 0]
            w2 = p_last.get('world_pos') or [0, 0]
            world_dist = np.hypot(w2[0] - w1[0], w2[1] - w1[1])
            dt = p_last.get('timestamp', 0) - p_first.get('timestamp', 0)
            raw_speed_mps = world_dist / dt if dt > 0 else 0
            raw_speed_kmh = raw_speed_mps * 3.6

            print(f"Total Trajectory Pixel Displacement: {pix_dist:.2f} px")
            print(f"Total Trajectory World Displacement: {world_dist:.2f} m over {dt:.3f} s")
            print(f"Unsmoothed Path Average Speed: {raw_speed_mps:.2f} m/s = {raw_speed_kmh:.2f} km/h")

        if speeds:
            print("Speed Measurements Sample (First 3 & Last 3):")
            for s in speeds[:3]:
                print(f"  Frame {s.frame_index} (t={s.timestamp:.2f}s): inst={s.instantaneous_kmh:.2f} km/h, smooth={s.smoothed_kmh:.2f} km/h, uncert=±{s.uncertainty_kmh:.2f} km/h, err_comp={s.error_components_json}")
            for s in list(speeds)[-3:]:
                print(f"  Frame {s.frame_index} (t={s.timestamp:.2f}s): inst={s.instantaneous_kmh:.2f} km/h, smooth={s.smoothed_kmh:.2f} km/h, uncert=±{s.uncertainty_kmh:.2f} km/h, err_comp={s.error_components_json}")

    # 4. TEST ±6 FRAME LOGIC & DISPLACEMENT ANALYSIS
    print("\n--- 4. TEST ±6 FRAME TEMPORAL MATCHING ANALYSIS ---")
    current_vid_frame = 150  # Mid-video evaluation frame
    print(f"Evaluation Video Frame: {current_vid_frame} (t = {current_vid_frame / video.fps:.2f}s)")
    
    all_tracks = db.query(Track).filter(Track.job_id == job.id).all()
    for trk in all_tracks:
        traj = trk.trajectory_json or []
        pts_in_range = [p for p in traj if abs(p.get("frame_index", 0) - current_vid_frame) <= 6]
        if pts_in_range:
            matched_pt = min(pts_in_range, key=lambda p: abs(p.get("frame_index", 0) - current_vid_frame))
            f_diff = matched_pt.get("frame_index") - current_vid_frame
            t_diff = f_diff / video.fps
            bbox = matched_pt.get("bbox") or [0, 0, 0, 0]
            
            # Find speed
            sm = db.query(SpeedMeasurement).filter(SpeedMeasurement.track_id == trk.id, SpeedMeasurement.frame_index == matched_pt.get("frame_index")).first()
            spd = sm.smoothed_kmh if sm else 60.0
            spd_ms = spd / 3.6
            dist_drift = spd_ms * abs(t_diff)

            print(f"Track #{trk.track_id:2d}: Matched Frame {matched_pt.get('frame_index')} (Diff: {f_diff:+d} frames = {t_diff:+.3f}s) | Speed: {spd:.1f} km/h | Dist Drift during diff: {dist_drift:.2f} m | BBox: {bbox}")

if __name__ == "__main__":
    audit_runtime()
