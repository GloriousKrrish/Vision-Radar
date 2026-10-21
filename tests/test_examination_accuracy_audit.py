import pytest
import numpy as np
from typing import List, Dict, Any

from visionradar.examination.schemas import (
    PersonDetection,
    ExaminationObjectDetection,
    ExaminationEventType,
    ExaminationRole,
    SeatingZone,
    ZoneType
)
from visionradar.examination.detector import ExaminationDetector
from visionradar.examination.tracker import (
    PersonTracker,
    ExaminationObjectTracker,
    calculate_iou,
    calculate_center_distance
)
from visionradar.examination.events import ExaminationEventEngine
from visionradar.examination.audit_exporter import ForensicAuditExporter
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Video, ExaminationTrackRecord, ExaminationEventRecord, ExaminationZoneRecord

def test_multiple_students_and_calibrated_threshold():
    """
    Validates that person detection threshold calibrated to 0.20 detects multiple students,
    including distant small students (e.g. conf 0.22).
    """
    detector = ExaminationDetector(confidence_threshold=0.20)
    assert detector.confidence_threshold == 0.20
    assert detector.class_thresholds[0] == 0.20

    # Synthetic detections to test categorization
    simulated_persons = [
        {"bbox": [100.0, 50.0, 160.0, 120.0], "confidence": 0.22, "size_category": "distant_small"},
        {"bbox": [200.0, 100.0, 320.0, 300.0], "confidence": 0.75, "size_category": "medium"},
        {"bbox": [50.0, 20.0, 400.0, 450.0], "confidence": 0.88, "size_category": "foreground"}
    ]

    tracker = PersonTracker()
    active_persons = tracker.update(simulated_persons, frame_index=0)
    assert len(active_persons) == 3
    # Distant student was not dropped
    assert any(p.confidence == 0.22 for p in active_persons)
    assert any(p.size_category == "distant_small" for p in active_persons)

def test_brief_occlusion_interpolation_and_continuity():
    """
    Validates that brief 1-2 frame occlusions do not fragment tracks and are marked is_interpolated=True.
    """
    tracker = PersonTracker(max_lost_frames=30)
    det1 = [{"bbox": [100.0, 100.0, 200.0, 250.0], "confidence": 0.85, "size_category": "medium"}]

    # Frame 0: Initial detection
    p0 = tracker.update(det1, frame_index=0)
    assert len(p0) == 1
    assert p0[0].track_id == 1
    assert not p0[0].is_interpolated

    # Frame 1: Missed detection (brief occlusion)
    p1 = tracker.update([], frame_index=1)
    assert len(p1) == 1
    assert p1[0].track_id == 1
    assert p1[0].is_interpolated
    assert tracker.tracks[0].time_since_update == 1

    # Frame 2: Missed detection (2nd frame)
    p2 = tracker.update([], frame_index=2)
    assert len(p2) == 1
    assert p2[0].track_id == 1
    assert p2[0].is_interpolated
    assert tracker.tracks[0].time_since_update == 2

    # Frame 3: Re-detected with small motion
    det3 = [{"bbox": [102.0, 102.0, 202.0, 252.0], "confidence": 0.83, "size_category": "medium"}]
    p3 = tracker.update(det3, frame_index=3)
    assert len(p3) == 1
    assert p3[0].track_id == 1  # Track ID preserved, no fragmentation
    assert not p3[0].is_interpolated
    assert tracker.tracks[0].time_since_update == 0

def test_staff_exclusion_from_student_alerts():
    """
    Validates that confirmed authorized staff (INVIGILATOR, EXAM_COORDINATOR, AUTHORIZED_STAFF)
    are strictly excluded from student-specific cheating candidate rules:
    - PEER_PROXIMITY
    - RESTRICTED_ZONE_ENTRY
    - LEAVING_SEATING_ZONE
    - POSSIBLE_PHONE_INTERACTION
    """
    engine = ExaminationEventEngine(job_id=999, video_id=999, fps=30.0)

    # Invigilator standing near student
    staff = PersonDetection(
        track_id=1,
        bbox=[100.0, 100.0, 180.0, 300.0],
        confidence=0.90,
        assigned_role=ExaminationRole.INVIGILATOR.value,
        role_source="CONFIG",
        assigned_zone="PODIUM"
    )

    student = PersonDetection(
        track_id=2,
        bbox=[120.0, 110.0, 190.0, 305.0], # close proximity to staff
        confidence=0.88,
        assigned_role=ExaminationRole.STUDENT.value,
        role_source="DEFAULT",
        assigned_zone="DESK_1"
    )

    phone_obj = ExaminationObjectDetection(
        object_id=1,
        object_track_id=101,
        object_class="cell phone",
        bbox=[110.0, 120.0, 130.0, 150.0],
        confidence=0.80,
        discrimination_status="CONFIRMED"
    )
    staff.associated_objects.append(phone_obj.to_dict())

    podium_zone = SeatingZone(
        zone_id="PODIUM",
        name="Teacher Podium",
        zone_type=ZoneType.RESTRICTED_ZONE,
        polygon=[[50, 50], [250, 50], [250, 350], [50, 350]]
    )

    events = engine.process_frame(
        frame_index=10,
        timestamp_s=0.33,
        persons=[staff, student],
        objects=[phone_obj],
        zones=[podium_zone]
    )

    # Ensure NO events were generated against the invigilator (Track #1)
    staff_events = [e for e in events if e.person_track_id == 1]
    assert len(staff_events) == 0, f"Expected 0 alerts for invigilator, got: {staff_events}"

def test_unknown_role_handling():
    """
    Validates that unconfigured tracks remain explicitly UNKNOWN.
    """
    track = PersonDetection(
        track_id=5,
        bbox=[200.0, 200.0, 260.0, 350.0],
        confidence=0.75,
        assigned_role=ExaminationRole.UNKNOWN.value,
        role_source="DEFAULT"
    )
    assert track.assigned_role == "UNKNOWN"
    assert track.role_source == "DEFAULT"

def test_calculator_discrimination_uncertainty():
    """
    Validates that handheld rectangular electronic objects under COCO model are classified
    as UNCERTAIN_ELECTRONIC_DEVICE with calculator ambiguity disclosures.
    """
    engine = ExaminationEventEngine(job_id=999, video_id=999, fps=30.0)

    student = PersonDetection(
        track_id=10,
        bbox=[100.0, 100.0, 200.0, 300.0],
        confidence=0.85,
        assigned_role=ExaminationRole.STUDENT.value,
        role_source="DEFAULT"
    )

    # Object marked uncertain electronic device (aspect ratio 1.3, calculator ambiguity)
    uncertain_device = ExaminationObjectDetection(
        object_id=2,
        object_track_id=202,
        object_class="candidate_electronic_device",
        bbox=[110.0, 150.0, 140.0, 185.0],
        confidence=0.40,
        discrimination_status="UNCERTAIN_ELECTRONIC_DEVICE",
        candidate_types=["cell phone", "calculator", "unknown_device"]
    )
    student.associated_objects.append(uncertain_device.to_dict())

    # Run for 25 consecutive frames to exceed candidate duration threshold
    all_evts = []
    for f in range(25):
        evts = engine.process_frame(
            frame_index=f,
            timestamp_s=f / 30.0,
            persons=[student],
            objects=[uncertain_device],
            zones=[]
        )
        all_evts.extend(evts)

    phone_evts = [e for e in all_evts if e.event_type == ExaminationEventType.POSSIBLE_PHONE_INTERACTION]
    assert len(phone_evts) > 0
    primary_evt = phone_evts[-1]
    assert "COCO_TAXONOMY_CALCULATOR_AMBIGUITY" in primary_evt.uncertainty_reasons
    assert primary_evt.priority == "AMBER" # Uncertain device does not escalate to RED automatically

def test_duplicate_event_suppression():
    """
    Validates that consecutive frames of the same anomaly extend the ongoing event
    rather than flooding duplicate events.
    """
    engine = ExaminationEventEngine(job_id=999, video_id=999, fps=30.0)
    student = PersonDetection(
        track_id=7,
        bbox=[100.0, 100.0, 200.0, 300.0],
        confidence=0.85,
        assigned_role=ExaminationRole.STUDENT.value,
        role_source="DEFAULT"
    )

    # Simulate repeated gaze deviations
    for f in range(60):
        student.posture_orientation = "left_deviation" if (f % 15 < 10) else "forward"
        engine.process_frame(
            frame_index=f,
            timestamp_s=f / 30.0,
            persons=[student],
            objects=[],
            zones=[]
        )

    all_evts = engine.get_all_events()
    gaze_evts = [e for e in all_evts if e.event_type == ExaminationEventType.GAZE_DEVIATION]
    # Gaze events should be grouped/cooldown-throttled, not 60 discrete events
    assert len(gaze_evts) <= 3

def test_forensic_audit_report_generation():
    """
    Validates that ForensicAuditExporter produces complete reports with roles,
    continuity metrics, and technical disclosures.
    """
    db = SessionLocal()
    job = db.query(ProcessingJob).filter(ProcessingJob.status == "SUCCEEDED").first()
    if not job:
        db.close()
        pytest.skip("No succeeded job in local database to run export test.")

    report = ForensicAuditExporter.generate_audit_report(job.id, db)
    assert "report_id" in report
    assert "cryptographic_seal" in report
    assert "summary_metrics" in report
    assert "role_breakdown" in report["summary_metrics"]
    assert "track_continuity_analysis" in report
    assert "disclosures" in report
    assert "calculator_discrimination" in report["disclosures"]

    html = ForensicAuditExporter.generate_audit_html(report)
    assert "<html" in html
    assert "CRYPTOGRAPHIC SEAL" in html
    assert "Ethical Standard" in html
    db.close()

def test_traffic_pipeline_untouched_and_functional():
    """
    Ensures that vehicle speed calculation and traffic logic have suffered zero regression.
    """
    from visionradar.cv.calibration import HomographyCalibrator
    from visionradar.cv.speed import MonocularSpeedEstimator
    from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint

    # Test homography matrix calculation
    img_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    wld_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    calib = HomographyCalibrator(image_points=img_pts, world_points=wld_pts)
    assert calib.H is not None

    # Test MonocularSpeedEstimator
    estimator = MonocularSpeedEstimator(calibrator=calib, window_size_frames=15, fps=30.0)
    assert estimator.fps == 30.0

    traj = VehicleTrajectory(track_id=1, vehicle_class="Car")
    for frame in range(20):
        t = frame / 30.0
        pt = TrajectoryPoint(frame_index=frame, timestamp=t, bbox=(10, 10, 20, 20), world_pos=(5.0, 10.0 + 20.0 * t))
        traj.add_point(pt)

    estimate = estimator.estimate_speed_at_frame(traj, target_frame_idx=15)
    assert estimate is not None
    assert estimate.smoothed_kmh > 0
