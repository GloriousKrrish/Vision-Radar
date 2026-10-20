import pytest
from visionradar.cv.detector import Detection
from visionradar.cv.tracker import ByteTrackTracker
from visionradar.cv.violations import ViolationRuleEngine
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint
from visionradar.examination.tracker import PersonTracker, ExaminationObjectTracker

def test_bytetrack_predict_step():
    tracker = ByteTrackTracker()
    det = Detection(bbox=(100.0, 100.0, 150.0, 160.0), confidence=0.90, class_id=2, class_name="Car")
    tracks = tracker.update([det], frame_index=0, timestamp=0.0)
    assert len(tracks) == 1
    t1 = tracks[0]
    orig_box = t1.bbox

    # Call predict_step on skipped frame (no detector)
    pred_tracks = tracker.predict_step(frame_index=1, timestamp=0.033)
    assert len(pred_tracks) == 1
    assert pred_tracks[0].track_id == t1.track_id
    assert pred_tracks[0].last_frame == 1
    assert len(pred_tracks[0].trajectory.points) == 2

def test_person_and_object_tracker_predict_step():
    p_tracker = PersonTracker()
    raw_p = [{"bbox": [50.0, 50.0, 120.0, 200.0], "confidence": 0.88}]
    p_tracks = p_tracker.update(raw_p, frame_index=0)
    assert len(p_tracks) == 1

    # Predict step on skipped frame
    pred_p = p_tracker.predict_step(frame_index=1)
    assert len(pred_p) == 1
    assert pred_p[0].track_id == p_tracks[0].track_id

    obj_tracker = ExaminationObjectTracker()
    # update with dummy object
    from visionradar.examination.schemas import ExaminationObjectDetection
    raw_obj = ExaminationObjectDetection(
        object_id=1,
        object_class="cell phone",
        bbox=[60.0, 70.0, 80.0, 100.0],
        confidence=0.75
    )
    o_tracks = obj_tracker.update([raw_obj], frame_index=0)
    assert len(o_tracks) == 1

    pred_o = obj_tracker.predict_step(frame_index=1)
    assert len(pred_o) == 1
    assert pred_o[0].object_class == "cell phone"

def test_wrong_way_driver_violation():
    rule_engine = ViolationRuleEngine(speed_limit_kmh=60.0)

    # Normal southward trajectory: increasing Y
    traj_normal = VehicleTrajectory(track_id=1, vehicle_class="Car")
    for f in range(20):
        y = 100.0 + f * 5.0  # moves southward from 100 to 195
        traj_normal.add_point(TrajectoryPoint(frame_index=f, timestamp=f * 0.033, bbox=(200, y, 250, y + 40)))

    viol_normal = rule_engine.evaluate_wrong_way(
        track_id=1,
        vehicle_class="Car",
        frame_index=19,
        timestamp=0.63,
        trajectory=traj_normal,
        expected_direction="SOUTHWARD"
    )
    assert viol_normal is None

    # Wrong-way trajectory: moving northward (decreasing Y) in a SOUTHWARD lane
    traj_wrong = VehicleTrajectory(track_id=2, vehicle_class="Car")
    for f in range(20):
        y = 300.0 - f * 5.0  # moves northward from 300 to 205 (delta = -95)
        traj_wrong.add_point(TrajectoryPoint(frame_index=f, timestamp=f * 0.033, bbox=(200, y, 250, y + 40)))

    viol_wrong = rule_engine.evaluate_wrong_way(
        track_id=2,
        vehicle_class="Car",
        frame_index=19,
        timestamp=0.63,
        trajectory=traj_wrong,
        expected_direction="SOUTHWARD"
    )
    assert viol_wrong is not None
    assert viol_wrong.violation_type == "WRONG_WAY_DRIVER"
    assert "WRONG-WAY" in viol_wrong.location_label
    assert viol_wrong.track_id == 2
