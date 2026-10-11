import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

from visionradar.intelligence.anomaly.features import TrajectoryFeatureEngine
from visionradar.intelligence.anomaly.engine import BehavioralAnomalyEngine, TrackAnomalyStateMachine
from visionradar.intelligence.anomaly.schemas import (
    AnomalyType, AnomalySeverity, AnomalyMethod, AnomalyState, TrajectoryFeatures, TrajectoryAnomaly
)

SYNTHETIC_LABEL = "SYNTHETIC TEST DATA"

def test_feature_extraction_and_validity_metadata():
    """1. Test feature extraction and structured FeatureValue metadata (value, unit, source, validity, quality)."""
    engine = TrajectoryFeatureEngine(fps=30.0)
    traj = [
        {"anchor_pixel": [100.0, 200.0], "timestamp": 0.0, "confidence": 0.90},
        {"anchor_pixel": [110.0, 205.0], "timestamp": 0.033, "confidence": 0.92},
        {"anchor_pixel": [120.0, 210.0], "timestamp": 0.066, "confidence": 0.88},
        {"anchor_pixel": [130.0, 215.0], "timestamp": 0.100, "confidence": 0.91}
    ]
    features = engine.extract_features(1, "Car", traj)
    assert features is not None
    assert features.track_id == 1
    assert features.vehicle_class == "Car"
    
    # Check structured feature dictionary matching Part 2 requirement
    assert "structured_features" in features.to_dict()
    sf = features.structured_features
    required_keys = ["track_duration_sec", "total_trajectory_length", "displacement", "lateral_displacement", "trajectory_smoothness"]
    for k in required_keys:
        assert k in sf, f"Missing structured feature metadata key: {k}"
        assert "value" in sf[k]
        assert "unit" in sf[k]
        assert "source" in sf[k]
        assert "validity" in sf[k]
        assert "quality" in sf[k]
        assert sf[k]["validity"] in ["VALID", "INVALID", "LOW_CONFIDENCE"]

def test_robust_baseline_calculation():
    """2. Test robust median and MAD baseline calculation."""
    engine = BehavioralAnomalyEngine()
    feat_engine = TrajectoryFeatureEngine()
    
    population = []
    for i in range(1, 10):
        traj = [{"anchor_pixel": [100.0 + f * 5.0, 200.0], "timestamp": f * 0.033, "confidence": 0.85} for f in range(1, 10)]
        tf = feat_engine.extract_features(i, "Car", traj)
        if tf:
            population.append(tf)
            
    med, mad = engine.calculate_robust_baseline(population, "trajectory_smoothness")
    assert med > 0.0
    assert mad > 0.0
    
    z_score = engine.compute_modified_zscore(1.0, med, mad)
    assert isinstance(z_score, float)

def test_vehicle_level_anomaly_detection():
    """3. Test vehicle-level anomaly detection (Trajectory deviation, lateral movement, sudden decel)."""
    engine = BehavioralAnomalyEngine()
    feat_engine = TrajectoryFeatureEngine()
    
    # Create erratic trajectory (high lateral movement)
    traj_erratic = []
    for f in range(1, 20):
        traj_erratic.append({
            "anchor_pixel": [100.0 + f * 5.0, 200.0 + (f * 8.0 if f > 10 else 0.0)],
            "timestamp": f * 0.033,
            "confidence": 0.88
        })
    tf = feat_engine.extract_features(42, "Car", traj_erratic)
    assert tf is not None
    
    anomalies = engine.detect_anomalies(tf, population_features=[tf], job_id=1, current_frame=20, timestamp=0.66)
    assert len(anomalies) > 0, f"Expected anomaly detection for erratic trajectory ({SYNTHETIC_LABEL})."
    assert anomalies[0].track_id == 42
    assert "explanation" in anomalies[0].to_dict()
    assert "feature_contributions" in anomalies[0].to_dict()

def test_context_aware_filtering():
    """4. Test that slow/stopped vehicle in CONGESTED traffic is NOT flagged as ABNORMAL_STOP_DURATION."""
    engine = BehavioralAnomalyEngine()
    feat_engine = TrajectoryFeatureEngine()
    
    # Stopped vehicle trajectory
    traj_stopped = [{"anchor_pixel": [100.0, 200.0], "timestamp": f * 0.1, "confidence": 0.90} for f in range(1, 50)]
    tf = feat_engine.extract_features(10, "Car", traj_stopped)
    
    # In FREE_FLOW context: Should trigger ABNORMAL_STOP_DURATION
    anoms_free = engine.detect_anomalies(tf, population_features=[tf], context_traffic_state="FREE_FLOW")
    stop_anoms_free = [a for a in anoms_free if a.anomaly_type == AnomalyType.ABNORMAL_STOP_DURATION]
    assert len(stop_anoms_free) > 0
    
    # In CONGESTED context: Context-aware filtering prevents false stop alarm
    anoms_cong = engine.detect_anomalies(tf, population_features=[tf], context_traffic_state="CONGESTED")
    stop_anoms_cong = [a for a in anoms_cong if a.anomaly_type == AnomalyType.ABNORMAL_STOP_DURATION]
    assert len(stop_anoms_cong) == 0, "Stop duration in congestion must be filtered out by context awareness."

def test_anomaly_state_machine_temporal_persistence():
    """5. Test state machine lifecycle: NORMAL -> OBSERVED -> SUSPICIOUS -> CONFIRMED_ANOMALY -> COOLDOWN -> NORMAL."""
    sm = TrackAnomalyStateMachine(track_id=7, anomaly_type=AnomalyType.ABNORMAL_LATERAL_MOVEMENT, min_observed_frames=3, min_cooldown_frames=5)
    
    # Frame 1: Trigger -> OBSERVED
    res1 = sm.update(has_trigger=True, score=0.75, frame_index=1, timestamp=0.033, explanation="Lateral shift", contribs={"lat": 65.0})
    assert sm.state == AnomalyState.OBSERVED
    assert res1 is None
    
    # Frame 2: Trigger -> OBSERVED (counter=2)
    res2 = sm.update(has_trigger=True, score=0.80, frame_index=2, timestamp=0.066, explanation="Lateral shift", contribs={"lat": 68.0})
    assert sm.state == AnomalyState.OBSERVED
    assert res2 is None

    # Frame 3: Trigger -> CONFIRMED_ANOMALY (counter=3)
    res3 = sm.update(has_trigger=True, score=0.85, frame_index=3, timestamp=0.100, explanation="Lateral shift", contribs={"lat": 72.0})
    assert sm.state == AnomalyState.CONFIRMED_ANOMALY
    assert res3 is None
    
    # Frame 4: No trigger -> Finalized payload returned & enters COOLDOWN
    res4 = sm.update(has_trigger=False, score=0.0, frame_index=4, timestamp=0.133, explanation="", contribs={})
    assert sm.state == AnomalyState.COOLDOWN
    assert res4 is not None
    assert res4["peak_score"] == 0.85
    assert res4["start_frame"] == 1
    assert res4["end_frame"] == 4

def test_traffic_level_anomalies():
    """6. Test traffic-level flow drop and occupancy spike anomaly detection."""
    engine = BehavioralAnomalyEngine()
    
    flow_hist = [50.0, 48.0, 52.0, 49.0, 5.0] # Abrupt flow drop
    occ_hist = [10.0, 12.0, 11.0, 10.0, 45.0] # Abrupt occupancy spike
    
    traffic_anoms = engine.detect_traffic_level_anomalies(
        job_id=1,
        frame_index=100,
        timestamp=3.33,
        flow_history=flow_hist,
        occupancy_history=occ_hist,
        lane_occupancies={"Lane 1": 35, "Lane 2": 2},
        congestion_state_change=("FREE_FLOW", "CONGESTED")
    )
    
    types_detected = [a.anomaly_type for a in traffic_anoms]
    assert AnomalyType.UNUSUAL_FLOW_DROP in types_detected
    assert AnomalyType.UNUSUAL_OCCUPANCY_SPIKE in types_detected
    assert AnomalyType.ABRUPT_CONGESTION_ONSET in types_detected
    assert AnomalyType.UNUSUAL_LANE_IMBALANCE in types_detected

def test_multi_vehicle_interaction():
    """7. Test multi-vehicle proximity interaction anomaly detection."""
    engine = BehavioralAnomalyEngine()
    tracks = [
        {"track_id": 1, "bbox": [100, 100, 150, 150], "speed_kmh": 45.0},
        {"track_id": 2, "bbox": [110, 110, 160, 160], "speed_kmh": 0.0} # Close distance < 45px with speed delta
    ]
    mvi_anoms = engine.detect_multi_vehicle_interactions(job_id=1, frame_index=50, timestamp=1.66, active_tracks=tracks)
    assert len(mvi_anoms) > 0
    assert mvi_anoms[0].anomaly_type == AnomalyType.MULTI_VEHICLE_INTERACTION_ANOMALY
    assert mvi_anoms[0].involved_track_ids == [1, 2]

def test_multi_signal_event_correlation():
    """8. Test multi-signal event correlation (without unproven causal inference)."""
    engine = BehavioralAnomalyEngine()
    
    class DummyEvent:
        def __init__(self, tid, etype):
            self.track_id = tid
            self.event_type = etype
            
    events = [DummyEvent(15, "stopped_vehicle")]
    anom = TrajectoryAnomaly(
        anomaly_id="a1", track_id=15, job_id=1,
        anomaly_type=AnomalyType.SUDDEN_DECELERATION, severity=AnomalySeverity.HIGH,
        score=0.85, confidence=0.9, start_frame=1, end_frame=1, timestamp=0.1,
        duration_sec=1.0, method=AnomalyMethod.RULE_BASED, explanation="Decel",
        evidence_frame_indices=[1], feature_contributions={}
    )
    
    correlated = engine.correlate_events(job_id=1, frame_index=10, timestamp=0.33, events=events, anomalies=[anom])
    assert len(correlated) > 0
    assert correlated[0].anomaly_type == AnomalyType.MULTI_SIGNAL_TRAFFIC_ANOMALY
    assert "multiple traffic signals temporally co-occurred" in correlated[0].explanation.lower()

def test_deterministic_behavior():
    """9. Test deterministic repeatability on synthetic feature input."""
    engine = BehavioralAnomalyEngine()
    feat_engine = TrajectoryFeatureEngine()
    
    traj = [{"anchor_pixel": [100.0 + f * 5.0, 200.0 + f * 4.0], "timestamp": f * 0.033, "confidence": 0.90} for f in range(1, 20)]
    tf = feat_engine.extract_features(88, "Car", traj)
    
    res1 = engine.detect_anomalies(tf, population_features=[tf])
    res2 = engine.detect_anomalies(tf, population_features=[tf])
    
    assert len(res1) == len(res2)
    if res1:
        assert res1[0].score == res2[0].score
        assert res1[0].explanation == res2[0].explanation

def test_no_fabricated_ground_truth_or_accuracy():
    """10. Test that validation_status is GT-UNAVAILABLE / ENGINEERING-VALIDATED and never fabricates accuracy."""
    anom = TrajectoryAnomaly(
        anomaly_id="a_test", track_id=1, job_id=1,
        anomaly_type=AnomalyType.TRAJECTORY_DEVIATION, severity=AnomalySeverity.MEDIUM,
        score=0.75, confidence=0.85, start_frame=1, end_frame=1, timestamp=0.1,
        duration_sec=1.0, method=AnomalyMethod.STATISTICAL_ROBUST_MAD, explanation="Test",
        evidence_frame_indices=[1], feature_contributions={}, validation_status="ENGINEERING-VALIDATED"
    )
    d = anom.to_dict()
    assert d["validation_status"] in ["ENGINEERING-VALIDATED", "GT-VALIDATED", "QUALITATIVE", "NOT-EVALUABLE"]
    assert "accuracy" not in d, "Accuracy metric must NOT be fabricated for anomaly detection without GT."
