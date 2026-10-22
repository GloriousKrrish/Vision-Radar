"""
VisionRadar Phase 5.6 Automated Test Suite:
Perspective-Aware Proximity Events, Stable Baseline Estimation, Motion Gating, and False-Positive Reduction.
"""

import pytest
import numpy as np
from visionradar.examination.events import ExaminationEventEngine
from visionradar.examination.schemas import PersonDetection, ExaminationEventType, SeatingZone, ZoneType, ExaminationRole

def test_stationary_pairs_at_stable_distances_generate_zero_events():
    """
    Validates that two stationary students sitting in adjacent desks or rows
    at a stable distance (even if < 110px in 2D image coordinates) generate ZERO
    false-positive proximity events across a sustained duration.
    """
    engine = ExaminationEventEngine(job_id=1, video_id=1, fps=30.0)

    # Simulate 120 frames (4.0s) of two students sitting stationary with minor detector noise (1px)
    np.random.seed(42)
    for f in range(120):
        noise1 = np.random.uniform(-0.8, 0.8, 4)
        noise2 = np.random.uniform(-0.8, 0.8, 4)
        # Bounding box centers: (130, 200) and (200, 200) -> 70px separation (under old 110px threshold!)
        p1 = PersonDetection(
            track_id=1,
            bbox=[100.0 + noise1[0], 140.0 + noise1[1], 160.0 + noise1[2], 260.0 + noise1[3]],
            confidence=0.88,
            assigned_role="STUDENT"
        )
        p2 = PersonDetection(
            track_id=2,
            bbox=[170.0 + noise2[0], 140.0 + noise2[1], 230.0 + noise2[2], 260.0 + noise2[3]],
            confidence=0.88,
            assigned_role="STUDENT"
        )
        engine.process_frame(
            frame_index=f,
            timestamp_s=f / 30.0,
            persons=[p1, p2],
            objects=[],
            zones=[]
        )

    prox_events = [e for e in engine.get_all_events() if e.event_type == ExaminationEventType.PEER_PROXIMITY]
    assert len(prox_events) == 0, f"Expected 0 proximity events for stationary students, got {len(prox_events)}"


def test_dynamic_distance_reduction_triggers_event():
    """
    Validates that when a student actively converges toward another student's desk
    (distance reduced by >=35% with active displacement), a PEER_PROXIMITY event is legitimately triggered.
    """
    engine = ExaminationEventEngine(job_id=2, video_id=2, fps=30.0)

    # Phase 1: Establish resting baseline at 120px separation (frames 0 to 30)
    for f in range(30):
        p1 = PersonDetection(track_id=10, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.90)
        p2 = PersonDetection(track_id=20, bbox=[220.0, 100.0, 280.0, 220.0], confidence=0.90)
        engine.process_frame(f, f / 30.0, [p1, p2], [], [])

    assert (10, 20) in engine._pairwise_baselines
    baseline_d0 = engine._pairwise_baselines[(10, 20)]["d0_px"]
    assert 115.0 <= baseline_d0 <= 125.0

    # Phase 2: Track 20 actively moves toward Track 10, closing distance to 50px (>55% reduction)
    for f in range(30, 60):
        # Move Track 20 from x=220 to x=150
        x_shift = max(150.0, 220.0 - (f - 30) * 5.0)
        p1 = PersonDetection(track_id=10, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.90)
        p2 = PersonDetection(track_id=20, bbox=[x_shift, 100.0, x_shift + 60.0, 220.0], confidence=0.90)
        engine.process_frame(f, f / 30.0, [p1, p2], [], [])

    prox_events = [e for e in engine.get_all_events() if e.event_type == ExaminationEventType.PEER_PROXIMITY]
    assert len(prox_events) == 1
    evt = prox_events[0]
    assert evt.person_track_id == 10
    assert evt.priority == "AMBER"
    assert "desk convergence" in evt.description.lower()
    assert "reduction" in evt.description.lower()


def test_one_moving_person_and_one_stationary_person():
    """
    Validates that proximity correctly detects an examinee or candidate walking up
    to a seated examinee's desk.
    """
    engine = ExaminationEventEngine(job_id=3, video_id=3, fps=30.0)

    # Candidate 5 is seated at (100, 100)
    # Candidate 8 walks from (350, 100) over 40 frames into Candidate 5's desk (140, 100)
    for f in range(50):
        x8 = max(140.0, 350.0 - f * 6.0)
        p5 = PersonDetection(track_id=5, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.88)
        p8 = PersonDetection(track_id=8, bbox=[x8, 100.0, x8 + 60.0, 220.0], confidence=0.88)
        engine.process_frame(f, f / 30.0, [p5, p8], [], [])

    prox_events = [e for e in engine.get_all_events() if e.event_type == ExaminationEventType.PEER_PROXIMITY]
    assert len(prox_events) >= 1
    assert any(e.person_track_id == 5 for e in prox_events)


def test_seating_zone_suppresses_inter_zone_stationary_proximity():
    """
    Validates that when candidates are established within configured distinct seating zones,
    inter-desk stationary proximity is suppressed by classroom geometry immunity.
    """
    engine = ExaminationEventEngine(job_id=4, video_id=4, fps=30.0)

    z1 = SeatingZone(zone_id="Desk-01", name="Desk 01", zone_type=ZoneType.SEATING_ZONE, polygon=[[80, 80], [170, 80], [170, 250], [80, 250]])
    z2 = SeatingZone(zone_id="Desk-02", name="Desk 02", zone_type=ZoneType.SEATING_ZONE, polygon=[[175, 80], [260, 80], [260, 250], [175, 250]])

    for f in range(60):
        p1 = PersonDetection(track_id=1, bbox=[100.0, 100.0, 150.0, 220.0], confidence=0.90, assigned_zone="Desk-01")
        p2 = PersonDetection(track_id=2, bbox=[185.0, 100.0, 235.0, 220.0], confidence=0.90, assigned_zone="Desk-02")
        engine.process_frame(f, f / 30.0, [p1, p2], [], [z1, z2])

    prox_events = [e for e in engine.get_all_events() if e.event_type == ExaminationEventType.PEER_PROXIMITY]
    assert len(prox_events) == 0


def test_temporal_gap_tolerance_prevents_duplicate_fragmentation():
    """
    Validates that a temporary tracking dropout or brief jitter (10 frames = 0.33s)
    does not fragment an ongoing proximity interaction into multiple duplicate events.
    """
    engine = ExaminationEventEngine(job_id=5, video_id=5, fps=30.0)

    # Establish baseline
    for f in range(25):
        p1 = PersonDetection(track_id=1, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.90)
        p2 = PersonDetection(track_id=2, bbox=[220.0, 100.0, 280.0, 220.0], confidence=0.90)
        engine.process_frame(f, f / 30.0, [p1, p2], [], [])

    # Converge and trigger event (frames 25 to 45)
    for f in range(25, 45):
        p1 = PersonDetection(track_id=1, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.90)
        p2 = PersonDetection(track_id=2, bbox=[140.0, 100.0, 200.0, 220.0], confidence=0.90) # close
        engine.process_frame(f, f / 30.0, [p1, p2], [], [])

    # Brief 10-frame detector occlusion / dropout (gap)
    for f in range(45, 55):
        # Track 2 is missing for 10 frames
        p1 = PersonDetection(track_id=1, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.90)
        engine.process_frame(f, f / 30.0, [p1], [], [])

    # Track 2 reappears in close proximity (frames 55 to 70)
    for f in range(55, 70):
        p1 = PersonDetection(track_id=1, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.90)
        p2 = PersonDetection(track_id=2, bbox=[140.0, 100.0, 200.0, 220.0], confidence=0.90)
        engine.process_frame(f, f / 30.0, [p1, p2], [], [])

    all_evts = engine.get_all_events()
    prox_events = [e for e in all_evts if e.event_type == ExaminationEventType.PEER_PROXIMITY]
    # Gap tolerance should merge or maintain this as ONE continuous occurrence rather than 2 fragmented events
    assert len(prox_events) == 1, f"Expected 1 unified event, got {len(prox_events)}"


def test_missing_baseline_fallback():
    """
    Validates that if a sudden heavy physical overlap occurs before a 20-frame baseline
    can be learned, the system safely falls back to direct overlap criteria with an uncertainty flag.
    """
    engine = ExaminationEventEngine(job_id=6, video_id=6, fps=30.0)

    # Immediate collision / overlap (< 40px) from frame 0
    for f in range(20):
        p1 = PersonDetection(track_id=1, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.85)
        p2 = PersonDetection(track_id=2, bbox=[120.0, 100.0, 180.0, 220.0], confidence=0.85) # overlapping
        engine.process_frame(f, f / 30.0, [p1, p2], [], [])

    prox_events = [e for e in engine.get_all_events() if e.event_type == ExaminationEventType.PEER_PROXIMITY]
    assert len(prox_events) == 1
    assert "UNESTABLISHED_PAIR_BASELINE" in prox_events[0].uncertainty_reasons
    assert prox_events[0].confidence == 0.65


def test_engine_isolation():
    """
    Validates that two independent engine instances do not share or leak baseline state.
    """
    engine_a = ExaminationEventEngine(job_id=101, video_id=101, fps=30.0)
    engine_b = ExaminationEventEngine(job_id=102, video_id=102, fps=30.0)

    p1 = PersonDetection(track_id=1, bbox=[100.0, 100.0, 160.0, 220.0], confidence=0.90)
    p2 = PersonDetection(track_id=2, bbox=[200.0, 100.0, 260.0, 220.0], confidence=0.90)

    for f in range(25):
        engine_a.process_frame(f, f / 30.0, [p1, p2], [], [])

    assert (1, 2) in engine_a._pairwise_baselines
    assert (1, 2) not in engine_b._pairwise_baselines
