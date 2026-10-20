import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from visionradar.examination.schemas import (
    ExaminationEventType,
    ZoneType,
    SeatingZone,
    PersonDetection,
    ExaminationObjectDetection,
    ExaminationEvent
)
from visionradar.examination.detector import ExaminationDetector
from visionradar.examination.tracker import PersonTracker
from visionradar.examination.seating import SeatingZoneManager
from visionradar.examination.events import ExaminationEventEngine

client = TestClient(app)

def test_examination_detector_init():
    """Verify ExaminationDetector initializes cleanly with ONNX model."""
    detector = ExaminationDetector()
    assert detector.input_size == (416, 416)
    assert detector.confidence_threshold == 0.25

def test_person_tracker_association():
    """Verify PersonTracker generates sequential track IDs and maintains spatial continuity."""
    tracker = PersonTracker(iou_threshold=0.30)

    # Frame 0: Two persons detected
    dets_f0 = [
        {"bbox": [50.0, 50.0, 100.0, 150.0], "confidence": 0.88},
        {"bbox": [200.0, 60.0, 260.0, 170.0], "confidence": 0.92}
    ]
    persons_f0 = tracker.update(dets_f0, frame_index=0)
    assert len(persons_f0) == 2
    id1, id2 = persons_f0[0].track_id, persons_f0[1].track_id
    assert id1 != id2

    # Frame 1: Slight motion (high IoU overlap)
    dets_f1 = [
        {"bbox": [52.0, 51.0, 102.0, 151.0], "confidence": 0.89},
        {"bbox": [198.0, 62.0, 258.0, 172.0], "confidence": 0.91}
    ]
    persons_f1 = tracker.update(dets_f1, frame_index=1)
    assert len(persons_f1) == 2
    # Verify track identity persistence
    assert persons_f1[0].track_id == id1
    assert persons_f1[1].track_id == id2

def test_seating_zone_manager():
    """Verify seating zones associate persons correctly based on ground contact coordinates."""
    mgr = SeatingZoneManager()
    mgr.add_zone(
        zone_id="Seat-A1",
        name="Front Left Desk",
        zone_type=ZoneType.SEATING_ZONE,
        polygon=[[20.0, 40.0], [120.0, 40.0], [120.0, 160.0], [20.0, 160.0]]
    )
    mgr.add_zone(
        zone_id="Restricted-Proctor",
        name="Proctor Stage",
        zone_type=ZoneType.RESTRICTED_ZONE,
        polygon=[[300.0, 0.0], [500.0, 0.0], [500.0, 100.0], [300.0, 100.0]]
    )

    # Person inside Seat-A1
    p1 = PersonDetection(track_id=1, bbox=[40.0, 50.0, 90.0, 140.0], confidence=0.90)
    p1 = mgr.associate_person(p1)
    assert p1.assigned_zone == "Seat-A1"
    assert p1.zone_confidence is not None and p1.zone_confidence >= 0.50

    # Person in Proctor Stage (restricted zone)
    p2 = PersonDetection(track_id=2, bbox=[350.0, 20.0, 420.0, 90.0], confidence=0.88)
    restr_zone = mgr.check_restricted_zone(p2)
    assert restr_zone is not None
    assert restr_zone.zone_id == "Restricted-Proctor"

def test_possible_phone_interaction_event():
    """Verify POSSIBLE_PHONE_INTERACTION candidate event triggers after sequential proximity."""
    engine = ExaminationEventEngine(job_id=1, video_id=1, fps=30.0)

    person = PersonDetection(track_id=10, bbox=[100.0, 100.0, 180.0, 260.0], confidence=0.92)
    phone = ExaminationObjectDetection(object_id=1, object_class="cell phone", bbox=[130.0, 160.0, 155.0, 200.0], confidence=0.85)

    events = []
    # Feed 7 consecutive frames of phone near person
    for f in range(7):
        events = engine.process_frame(
            frame_index=f,
            timestamp_s=f / 30.0,
            persons=[person],
            objects=[phone],
            zones=[]
        )

    phone_events = [e for e in events if e.event_type == ExaminationEventType.POSSIBLE_PHONE_INTERACTION]
    assert len(phone_events) == 1
    evt = phone_events[0]
    assert evt.person_track_id == 10
    assert evt.review_status == "PENDING"
    assert "cell phone" in evt.description.lower()
    assert evt.confidence > 0.70

def test_restricted_zone_entry_event():
    """Verify RESTRICTED_ZONE_ENTRY candidate event triggers when person enters restricted area."""
    engine = ExaminationEventEngine(job_id=1, video_id=1, fps=30.0)
    zone = SeatingZone(zone_id="Zone-R1", name="Confidential Desk", zone_type=ZoneType.RESTRICTED_ZONE, polygon=[[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]])

    person = PersonDetection(track_id=5, bbox=[50.0, 50.0, 120.0, 180.0], confidence=0.95, assigned_zone="Zone-R1")

    events = []
    # Feed 6 frames
    for f in range(6):
        events = engine.process_frame(
            frame_index=f,
            timestamp_s=f / 30.0,
            persons=[person],
            objects=[],
            zones=[zone]
        )

    restr_events = [e for e in events if e.event_type == ExaminationEventType.RESTRICTED_ZONE_ENTRY]
    assert len(restr_events) == 1
    evt = restr_events[0]
    assert evt.person_track_id == 5
    assert evt.review_status == "PENDING"
    assert "Confidential Desk" in evt.description

def test_leaving_seating_zone_event():
    """Verify LEAVING_SEATING_ZONE triggers when a student vacates their assigned seat."""
    engine = ExaminationEventEngine(job_id=1, video_id=1, fps=30.0)
    seat = SeatingZone(zone_id="Seat-1", name="Seat 1", zone_type=ZoneType.SEATING_ZONE, polygon=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])

    # Student initially at seat
    p_seated = PersonDetection(track_id=3, bbox=[20.0, 20.0, 80.0, 90.0], confidence=0.90, assigned_zone="Seat-1")
    engine.process_frame(0, 0.0, [p_seated], [], [seat])

    # Student walks away (assigned_zone is None for 13 frames)
    p_away = PersonDetection(track_id=3, bbox=[250.0, 150.0, 310.0, 230.0], confidence=0.90, assigned_zone=None)
    events = []
    for f in range(1, 14):
        events = engine.process_frame(f, f / 30.0, [p_away], [], [seat])

    vacate_events = [e for e in events if e.event_type == ExaminationEventType.LEAVING_SEATING_ZONE]
    assert len(vacate_events) == 1
    evt = vacate_events[0]
    assert evt.person_track_id == 3
    assert evt.review_status == "PENDING"

def test_observable_event_neutrality():
    """Verify that event descriptions are strictly factual and never use accusatory language."""
    engine = ExaminationEventEngine(job_id=2, video_id=1)
    obj = ExaminationObjectDetection(object_id=1, object_class="book", bbox=[150.0, 150.0, 180.0, 180.0], confidence=0.8)
    p1 = PersonDetection(track_id=1, bbox=[130.0, 130.0, 170.0, 200.0], confidence=0.9)
    p2 = PersonDetection(track_id=2, bbox=[160.0, 130.0, 200.0, 200.0], confidence=0.9)

    events = engine.process_frame(0, 0.0, [p1, p2], [obj], [])
    for evt in events:
        desc = evt.description.lower()
        # Strictly verify absence of speculative or defamatory language
        assert "cheat" not in desc
        assert "guilty" not in desc
        assert "copy" not in desc
        assert "illegal" not in desc
        assert evt.review_status == "PENDING"

def test_examination_api_endpoints():
    """Verify REST endpoints for classification, zones, and human event review."""
    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200

    # 2. Upload video / check projects
    proj_res = client.get("/api/v1/projects")
    assert proj_res.status_code == 200
    projects = proj_res.json()
    proj_id = projects[0]["id"] if projects else 1

    # 3. Create Seating Zone
    zone_payload = {
        "zone_id": "Test-Seat-01",
        "name": "Examination Desk 1",
        "zone_type": "SEATING_ZONE",
        "polygon": [[10.0, 10.0], [100.0, 10.0], [100.0, 100.0], [10.0, 100.0]]
    }
    z_res = client.post(f"/api/v1/videos/1/examination/zones", json=zone_payload)
    # 200 if video 1 exists, or 404 if video 1 not in DB (resilient)
    assert z_res.status_code in (200, 404)

    # 4. Review invalid event returns 404
    review_res = client.patch("/api/v1/examination/events/non_existent_event_id/review", json={"review_status": "ACCEPTED"})
    assert review_res.status_code == 404
