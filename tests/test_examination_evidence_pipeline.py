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
from visionradar.examination.detector import ExaminationDetector, DEFAULT_CLASS_THRESHOLDS
from visionradar.examination.tracker import PersonTracker, ExaminationObjectTracker, associate_persons_and_objects
from visionradar.examination.events import ExaminationEventEngine

client = TestClient(app)

def test_detector_class_thresholds_and_metadata():
    """Verify class-specific thresholds and model metadata provenance."""
    detector = ExaminationDetector()
    meta = detector.get_model_metadata()

    assert meta["model_name"] == "YOLOX-Nano ONNX"
    assert meta["runtime"] == "ONNXRuntime CPUExecutionProvider (Thread-isolated)"
    assert meta["input_dimensions"] == [416, 416]
    assert "cell phone" in meta["supported_classes"].values()
    assert "book" in meta["supported_classes"].values()
    assert meta["class_thresholds"]["cell phone"] <= 0.20
    assert meta["class_thresholds"]["person"] >= 0.25
    assert len(meta["known_limitations"]) >= 3

def test_person_object_association_and_uncertainty():
    """Verify evidence-based spatial association between person and phone."""
    person = PersonDetection(track_id=1, bbox=[100.0, 100.0, 300.0, 500.0], confidence=0.92)
    # Phone on desk in front of lower body
    phone = ExaminationObjectDetection(
        object_id=1,
        object_class="cell phone",
        bbox=[160.0, 360.0, 220.0, 440.0],
        confidence=0.45
    )

    associate_persons_and_objects([person], [phone])

    assert phone.association_status == "INTERACTING"
    assert phone.near_person_track_id == 1
    assert phone.association_confidence >= 0.70
    assert len(person.associated_objects) == 1
    assert person.associated_objects[0]["object_class"] == "cell phone"

def test_phone_interaction_temporal_windowing():
    """Verify phone interaction triggers with intermittent hand occlusion (non-consecutive frames)."""
    engine = ExaminationEventEngine(job_id=10, video_id=1, fps=30.0)
    person = PersonDetection(track_id=1, bbox=[100.0, 100.0, 300.0, 500.0], confidence=0.92)
    phone = ExaminationObjectDetection(
        object_id=1,
        object_class="cell phone",
        bbox=[160.0, 360.0, 220.0, 440.0],
        confidence=0.42,
        near_person_track_id=1,
        association_status="INTERACTING",
        association_confidence=0.88
    )

    # Frame 0: phone detected
    engine.process_frame(0, 0.0, [person], [phone], [])
    # Frame 1: hand occludes phone (missed detection)
    engine.process_frame(1, 0.033, [person], [], [])
    # Frame 2: phone detected
    engine.process_frame(2, 0.066, [person], [phone], [])
    # Frame 3: hand occludes phone again
    engine.process_frame(3, 0.100, [person], [], [])
    # Frame 4: phone detected (3rd observation in window)
    events = engine.process_frame(4, 0.133, [person], [phone], [])

    assert len(events) >= 1
    phone_evt = next((e for e in events if e.event_type == ExaminationEventType.POSSIBLE_PHONE_INTERACTION), None)
    assert phone_evt is not None
    assert phone_evt.person_track_id == 1
    assert phone_evt.priority == "RED"
    assert phone_evt.evidence_strength in ("HIGH", "MEDIUM")
    assert "INTERMITTENT_HAND_OCCLUSION" in phone_evt.uncertainty_reasons
    assert len(phone_evt.supporting_frames) == 3

def test_zero_seating_zone_false_alarms():
    """Verify that when no seating zones exist, LEAVING_SEATING_ZONE never fires."""
    engine = ExaminationEventEngine(job_id=11, video_id=2, fps=30.0)
    person = PersonDetection(track_id=1, bbox=[100.0, 100.0, 200.0, 300.0], confidence=0.90, assigned_zone=None)

    # Stream 30 frames with empty zones
    for f in range(30):
        evts = engine.process_frame(f, f / 30.0, [person], [], [])
        vacate_evts = [e for e in evts if e.event_type == ExaminationEventType.LEAVING_SEATING_ZONE]
        assert len(vacate_evts) == 0

    all_evts = engine.get_all_events()
    assert len([e for e in all_evts if e.event_type == ExaminationEventType.LEAVING_SEATING_ZONE]) == 0

def test_primary_finding_ranking():
    """Verify that RED priority with HIGH evidence is selected as primary candidate finding."""
    engine = ExaminationEventEngine(job_id=12, video_id=3, fps=30.0)
    person = PersonDetection(track_id=1, bbox=[100.0, 100.0, 300.0, 500.0], confidence=0.92)
    phone = ExaminationObjectDetection(
        object_id=1,
        object_class="cell phone",
        bbox=[160.0, 360.0, 220.0, 440.0],
        confidence=0.55,
        near_person_track_id=1,
        association_status="INTERACTING",
        association_confidence=0.92
    )

    # Feed 5 frames with phone
    for f in range(5):
        engine.process_frame(f, f / 30.0, [person], [phone], [])

    all_evts = engine.get_all_events()
    assert len(all_evts) >= 1
    primary = next((e for e in all_evts if e.primary_finding), None)
    assert primary is not None
    assert primary.event_type == ExaminationEventType.POSSIBLE_PHONE_INTERACTION
    assert primary.priority == "RED"
    assert primary.evidence_strength == "HIGH"

def test_proctor_report_generation():
    """Verify GET /api/v1/jobs/{job_id}/examination/report returns complete compliant schema."""
    # Run against verification job created earlier
    res = client.get("/api/v1/jobs/663/examination/report")
    assert res.status_code == 200
    report = res.json()

    assert "report_id" in report
    assert "video_metadata" in report
    assert "analysis_configuration" in report
    assert "primary_candidate_finding" in report
    assert "ethical_and_privacy_disclosures" in report
    assert report["primary_candidate_finding"]["event_type"] == "POSSIBLE_PHONE_INTERACTION"
    assert report["ethical_and_privacy_disclosures"]["proctor_in_the_loop"] == "MANDATORY. Automated events represent candidate visual patterns for human review only."
    assert "biometric_facial_recognition" in report["ethical_and_privacy_disclosures"]
