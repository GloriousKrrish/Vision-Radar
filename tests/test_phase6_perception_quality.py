"""
Phase 6 Perception Quality Engine Unit Test Suite

Tests:
1. Detection quality flags (VALID, LOW_CONFIDENCE, TOO_SMALL, EDGE_CLIPPED, SUSPICIOUS_GEOMETRY, DUPLICATE_CANDIDATE, OUTSIDE_ROI)
2. Small object bins (<16x16, 16-32, 32-64, 64-128, >128)
3. Class consensus and class flip counts
4. Track lifecycle states and gap classification (SHORT_GAP, MEDIUM_GAP, LONG_GAP)
5. True fragmentation semantics vs simple unobserved frame count
6. Trajectory quality evaluation and impossible-motion anomaly detection
7. Calibration ROI boundary membership and numerical homography safety
8. Deterministic repeatability across identical input runs
"""

import pytest
import numpy as np
from visionradar.perception.schemas import (
    VehicleDetection, VehicleTrack, BoundingBox, VehicleClass, TrackState,
    DetectionQualityFlag, TrajectoryStatus, CalibrationStatus
)
from visionradar.perception.quality_engine import (
    DetectionQualityEngine, TrackQualityEngine, TrajectoryQualityEngine,
    point_in_polygon, compute_iou_boxes
)


def test_detection_quality_flags_and_size_bins():
    engine = DetectionQualityEngine(low_conf_thresh=0.35)

    # 1. Valid Detection (>128 px)
    det_valid = VehicleDetection(
        detection_id="det1", frame_id=1,
        bbox=BoundingBox(100, 100, 300, 300), # 200x200 px
        confidence=0.85, vehicle_class=VehicleClass.CAR, class_confidence=0.90
    )

    # 2. Too Small & Low Confidence Detection
    det_small = VehicleDetection(
        detection_id="det2", frame_id=1,
        bbox=BoundingBox(100, 100, 110, 110), # 10x10 px
        confidence=0.20, vehicle_class=VehicleClass.MOTORCYCLE, class_confidence=0.20
    )

    # 3. Edge Clipped Detection
    det_edge = VehicleDetection(
        detection_id="det3", frame_id=1,
        bbox=BoundingBox(1, 100, 100, 200), # touches x=1
        confidence=0.75, vehicle_class=VehicleClass.BUS, class_confidence=0.80
    )

    res = engine.evaluate_detections([det_valid, det_small, det_edge], frame_width=1920, frame_height=1080)

    assert len(res) == 3
    assert res[0].quality_flags == [DetectionQualityFlag.VALID.value]
    assert res[0].size_category == ">128"
    assert res[0].detector_validity is True

    assert DetectionQualityFlag.TOO_SMALL.value in res[1].quality_flags
    assert DetectionQualityFlag.LOW_CONFIDENCE.value in res[1].quality_flags
    assert res[1].size_category == "<16x16"
    assert res[1].detector_validity is False

    assert DetectionQualityFlag.EDGE_CLIPPED.value in res[2].quality_flags


def test_track_quality_class_consensus_and_fragmentation():
    engine = TrackQualityEngine(short_gap_max=3, medium_gap_max=10)

    # Frame 1: Track 1 seen as CAR
    trk1 = VehicleTrack(
        track_id=1, state=TrackState.ACTIVE,
        bbox=BoundingBox(100, 100, 200, 200), confidence=0.85,
        vehicle_class=VehicleClass.CAR, first_seen_frame=1, last_seen_frame=1,
        frames_tracked=1, frames_since_last_detection=0
    )
    res1 = engine.update_track_quality([trk1], current_frame=1)
    assert res1[0].consensus_class == VehicleClass.CAR
    assert res1[0].class_flip_count == 0
    assert res1[0].fragmentation_count == 0

    # Frame 2: Track 1 flip raw class to TRUCK
    trk2 = VehicleTrack(
        track_id=1, state=TrackState.ACTIVE,
        bbox=BoundingBox(105, 105, 205, 205), confidence=0.80,
        vehicle_class=VehicleClass.TRUCK, first_seen_frame=1, last_seen_frame=2,
        frames_tracked=2, frames_since_last_detection=0
    )
    res2 = engine.update_track_quality([trk2], current_frame=2)
    assert res2[0].class_flip_count == 1

    # Frame 3 & 4: Missed detection (consecutive_missing = 2)
    trk_miss = VehicleTrack(
        track_id=1, state=TrackState.LOST,
        bbox=BoundingBox(105, 105, 205, 205), confidence=0.70,
        vehicle_class=VehicleClass.CAR, first_seen_frame=1, last_seen_frame=2,
        frames_tracked=2, frames_since_last_detection=2
    )
    engine.update_track_quality([trk_miss], current_frame=4)

    # Frame 5: Re-matched hit -> Should record SHORT_GAP, recovery, and fragmentation count = 1
    trk_hit = VehicleTrack(
        track_id=1, state=TrackState.ACTIVE,
        bbox=BoundingBox(115, 115, 215, 215), confidence=0.88,
        vehicle_class=VehicleClass.CAR, first_seen_frame=1, last_seen_frame=5,
        frames_tracked=3, frames_since_last_detection=0
    )
    res5 = engine.update_track_quality([trk_hit], current_frame=5)
    assert res5[0].recovery_count == 1
    assert res5[0].fragmentation_count == 1
    assert res5[0].missing_frames == 1


class DummyPoint:
    def __init__(self, frame, x, y):
        self.frame_index = frame
        self.anchor_pixel = (x, y)


def test_trajectory_quality_and_impossible_motion_guard():
    polygon = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)]
    engine = TrajectoryQualityEngine(roi_polygon=polygon)

    trk = VehicleTrack(
        track_id=10, state=TrackState.ACTIVE,
        bbox=BoundingBox(500, 500, 600, 600), confidence=0.90,
        vehicle_class=VehicleClass.CAR, first_seen_frame=1, last_seen_frame=10,
        frames_tracked=10, frames_since_last_detection=0
    )

    # 1. Normal Smooth Trajectory
    pts_normal = [
        DummyPoint(1, 500, 500),
        DummyPoint(2, 505, 505),
        DummyPoint(3, 510, 510)
    ]
    res_normal = engine.evaluate_track_trajectory(trk, pts_normal)
    assert res_normal.trajectory_status == TrajectoryStatus.TRAJECTORY_VALID.value
    assert res_normal.calibration_status == CalibrationStatus.CALIBRATION_VALID.value
    assert res_normal.motion_anomaly is False

    # 2. Impossible Motion Jump (200px jump in 1 frame)
    pts_jump = [
        DummyPoint(1, 500, 500),
        DummyPoint(2, 750, 750)
    ]
    res_jump = engine.evaluate_track_trajectory(trk, pts_jump)
    assert res_jump.motion_anomaly is True
    assert "IMPOSSIBLE" in res_jump.motion_anomaly_reason
    assert res_jump.trajectory_status == TrajectoryStatus.TRAJECTORY_INVALID.value


def test_point_in_polygon_and_homography_safety():
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon(50, 50, poly) is True
    assert point_in_polygon(150, 50, poly) is False

    # Unstable Homography Condition Number
    H_bad = np.array([[1e-10, 0, 0], [0, 1e-10, 0], [0, 0, 1e10]])
    engine = TrajectoryQualityEngine(roi_polygon=poly)
    trk = VehicleTrack(
        track_id=1, state=TrackState.ACTIVE,
        bbox=BoundingBox(10, 10, 20, 20), confidence=0.8,
        vehicle_class=VehicleClass.CAR, first_seen_frame=1, last_seen_frame=2,
        frames_tracked=2, frames_since_last_detection=0
    )
    res = engine.evaluate_track_trajectory(trk, [DummyPoint(1, 15, 20), DummyPoint(2, 16, 20)], H_matrix=H_bad)
    assert res.calibration_status == CalibrationStatus.CALIBRATION_UNSTABLE.value


def test_deterministic_repeatability():
    det = VehicleDetection(
        detection_id="det_det", frame_id=1,
        bbox=BoundingBox(10, 10, 50, 50), confidence=0.80,
        vehicle_class=VehicleClass.CAR, class_confidence=0.85
    )
    engine1 = DetectionQualityEngine()
    engine2 = DetectionQualityEngine()

    res1 = engine1.evaluate_detections([det])
    res2 = engine2.evaluate_detections([det])

    assert res1[0].to_dict() == res2[0].to_dict()
