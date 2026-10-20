import pytest
from visionradar.examination.schemas import PersonDetection, ExaminationEventType
from visionradar.examination.events import ExaminationEventEngine

def test_gaze_deviation_event():
    engine = ExaminationEventEngine(job_id=1, video_id=1, fps=30.0)

    # Simulate 22 frames of lateral head turn / deviation
    events = []
    for f in range(25):
        p = PersonDetection(
            track_id=1,
            bbox=[100.0, 100.0, 180.0, 220.0],
            confidence=0.90,
            posture_orientation="lateral_right"
        )
        events = engine.process_frame(
            frame_index=f,
            timestamp_s=f / 30.0,
            persons=[p],
            objects=[],
            zones=[]
        )

    gaze_events = [e for e in engine.get_all_events() if e.event_type == ExaminationEventType.GAZE_DEVIATION]
    assert len(gaze_events) == 1
    assert gaze_events[0].person_track_id == 1
    assert gaze_events[0].priority == "AMBER"
    assert "gaze deviation" in gaze_events[0].description.lower()

def test_peer_proximity_event():
    engine = ExaminationEventEngine(job_id=1, video_id=1, fps=30.0)

    # Simulate two students sitting in close proximity (< 100px) across 20 frames
    for f in range(20):
        p1 = PersonDetection(track_id=1, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.88)
        p2 = PersonDetection(track_id=2, bbox=[140.0, 100.0, 200.0, 220.0], confidence=0.85)
        engine.process_frame(
            frame_index=f,
            timestamp_s=f / 30.0,
            persons=[p1, p2],
            objects=[],
            zones=[]
        )

    prox_events = [e for e in engine.get_all_events() if e.event_type == ExaminationEventType.PEER_PROXIMITY]
    assert len(prox_events) == 1
    assert prox_events[0].priority == "AMBER"
    assert "proximity" in prox_events[0].description.lower()
