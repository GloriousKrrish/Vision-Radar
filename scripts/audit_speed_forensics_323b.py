import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.models.database import get_db
from visionradar.models.entities import ProcessingJob, Video, Calibration, Track, SpeedMeasurement
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.speed import MonocularSpeedEstimator
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint

def audit_tracks():
    db = next(get_db())
    job = db.query(ProcessingJob).filter(ProcessingJob.id == 70).first()
    if not job:
        print("Job #70 not found")
        return

    calib = db.query(Calibration).filter(Calibration.id == (job.calibration_id or 1)).first()
    calibrator = HomographyCalibrator(image_points=calib.image_points_json, world_points=calib.world_points_json)
    estimator = MonocularSpeedEstimator(calibrator)

    tracks = db.query(Track).filter(Track.job_id == 70).all()
    print(f"Total Tracks in Job #70: {len(tracks)}")

    target_ids = [1, 12, 14, 15]

    for trk in tracks:
        if trk.track_id in target_ids:
            traj = VehicleTrajectory(trk.track_id, trk.vehicle_class)
            for p in (trk.trajectory_json or []):
                traj.add_point(TrajectoryPoint(p['frame_index'], p['timestamp'], bbox=p['bbox'], anchor_pixel=p['anchor_pixel']))

            estimator.project_trajectory(traj, 1920, 1080)

            valid_count = sum(1 for p in traj.points if p.roi_status == "VALID")
            invalid_count = len(traj.points) - valid_count

            print(f"\n================ TRACK #{trk.track_id} ({trk.vehicle_class}) ================")
            print(f"Total Frames: {len(traj.points)} | Valid ROI: {valid_count} | Invalid/OUT_OF_ROI: {invalid_count}")

            speeds = db.query(SpeedMeasurement).filter(SpeedMeasurement.track_id == trk.id).all()
            print(f"Database Speed Measurements Count: {len(speeds)}")

            # Print frame trace for specific frames
            print(f"\nDetailed Frame Samples for Track #{trk.track_id}:")
            for p in traj.points:
                if trk.track_id == 1 and p.frame_index not in [0, 55, 66, 105, 200, 297, 298, 299] and p.frame_index % 10 != 0:
                    continue
                if trk.track_id != 1 and p.frame_index % 20 != 0:
                    continue

                u, v = p.anchor_pixel
                in_roi = calibrator.is_point_in_quadrilateral(u, v, 1920, 1080)
                is_stable, reason = calibrator.is_homography_stable(u, v, 1920, 1080)

                pt = np.array([u * (800/1920), v * (450/1080), 1.0], dtype=np.float64)
                res = calibrator.H @ pt
                denom = float(res[2])

                wx, wy = p.world_pos if p.world_pos else (0.0, 0.0)

                est = estimator.estimate_speed_at_frame(traj, p.frame_index)
                sp_kmh = est.smoothed_kmh if (est and est.smoothed_kmh > 0) else 0.0
                sp_val = est.validity if est else "N/A"

                print(f"  Frame {p.frame_index:3d} | Anchor=({u:6.1f}, {v:6.1f}) | ROI={str(in_roi):<5} | Denom={denom:7.4f} | World=({wx:6.2f}, {wy:6.2f}) | Status={p.roi_status:<16} | Speed={sp_kmh:6.1f} km/h ({sp_val})")

if __name__ == "__main__":
    audit_tracks()
