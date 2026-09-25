import os
import pytest
from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detector import LightweightDetector
from visionradar.cv.tracker import ByteTrackTracker
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.speed import MonocularSpeedEstimator
from visionradar.cv.analytics import TrafficAnalyticsEngine
from visionradar.cv.violations import ViolationRuleEngine
from visionradar.cv.evidence import EvidenceWriter

def test_full_cv_pipeline():
    video_path = "data/videos/synthetic_highway.mp4"
    assert os.path.exists(video_path), "Synthetic video fixture must exist."

    # 1. Decode video metadata
    decoder = VideoDecoder(video_path)
    meta = decoder.get_metadata()
    assert meta.width == 800
    assert meta.height == 450
    assert meta.total_frames == 300

    # 2. Calibration (4 corners of synthetic highway road)
    # Projections of (lx, d): proj(-1, 150), proj(13, 150), proj(13, 0.5), proj(-1, 0.5)
    img_pts = [(330.0, 160.0), (470.0, 160.0), (748.0, 435.0), (51.0, 435.0)]
    world_pts = [(-1.0, 150.0), (13.0, 150.0), (13.0, 0.5), (-1.0, 0.5)]

    calibrator = HomographyCalibrator(image_points=img_pts, world_points=world_pts)
    estimator = MonocularSpeedEstimator(calibrator=calibrator, window_size_frames=15, fps=meta.fps)

    detector = LightweightDetector()
    tracker = ByteTrackTracker(iou_threshold=0.05)
    rule_engine = ViolationRuleEngine(speed_limit_kmh=75.0)
    evidence_writer = EvidenceWriter(output_dir="data/evidence/test_run")

    active_tracks = {}
    speed_records = []
    violations = []

    # Process first 120 frames
    for frame_idx, timestamp, frame in decoder.decode_frames():
        if frame_idx >= 120:
            break

        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame_idx, timestamp)

        for trk in tracks:
            active_tracks[trk.track_id] = trk
            estimator.project_trajectory(trk.trajectory)
            est = estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=frame_idx)
            if est is not None:
                speed_records.append({
                    "track_id": trk.track_id,
                    "smoothed_kmh": est.smoothed_kmh,
                    "vehicle_class": trk.vehicle_class
                })
                viol = rule_engine.evaluate_track(
                    track_id=trk.track_id,
                    vehicle_class=trk.vehicle_class,
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    estimate=est
                )
                if viol is not None:
                    violations.append(viol)
                    evidence_writer.write_evidence_package(
                        violation_id=viol.violation_id,
                        frame_img=frame,
                        bbox=trk.bbox,
                        metadata=viol.to_dict()
                    )

    # Verify Analytics
    summary = TrafficAnalyticsEngine.compute_summary(speed_records)
    assert summary["total_vehicles"] > 0
    assert summary["mean_speed_kmh"] > 0.0

    print("Pipeline test executed successfully with summary:", summary)
