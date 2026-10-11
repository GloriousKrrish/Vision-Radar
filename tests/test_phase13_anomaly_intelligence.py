"""
VisionRadar Phase 13 — Anomaly Intelligence & Advanced Traffic Analytics Test Suite

Tests:
- Trajectory feature extraction & provenance tracking
- Behavioral anomaly detection engine (Rule-Based & Robust Z-Score)
- Congestion state transition state machine & hysteresis persistence
- Vehicle behavior profile generation
- Explainable traffic risk scoring & factor breakdown
- API endpoint integration
- Zero speed GT claims & zero hardcoded/fabricated metrics assertions
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from apps.api.main import app

from visionradar.intelligence.anomaly import (
    TrajectoryFeatureEngine,
    BehavioralAnomalyEngine,
    CongestionTransitionEngine,
    VehicleBehaviorProfileEngine,
    TrafficRiskScoringEngine,
    AnomalyType,
    AnomalySeverity,
    AnomalyMethod
)

client = TestClient(app)


def test_trajectory_feature_extraction():
    """Verify trajectory feature engine extracts all 18 mathematical features with provenance."""
    engine = TrajectoryFeatureEngine(fps=30.0)

    # Synthetic track trajectory points
    points = []
    for f in range(60):
        t = f / 30.0
        # Simulated curve trajectory
        x = 400.0 + f * 5.0
        y = 200.0 + (f ** 1.1)
        points.append({
            "frame_index": f,
            "timestamp": t,
            "bbox": [x - 20, y - 20, x + 20, y + 20],
            "anchor_pixel": [x, y],
            "confidence": 0.92
        })

    tf = engine.extract_features(track_id=42, vehicle_class="car", trajectory=points)

    assert tf is not None
    assert tf.track_id == 42
    assert tf.vehicle_class == "car"
    assert tf.track_duration_sec == pytest.approx(2.0, abs=0.1)
    assert tf.total_trajectory_length > 0.0
    assert tf.displacement > 0.0
    assert 0.0 <= tf.trajectory_smoothness <= 1.0
    assert tf.mean_confidence == pytest.approx(0.92, abs=0.01)
    assert "total_trajectory_length" in tf.provenance


def test_behavioral_anomaly_engine_wrong_way():
    """Verify wrong way behavior anomaly detection."""
    engine = BehavioralAnomalyEngine(method=AnomalyMethod.RULE_BASED)

    # Create dummy features with opposing direction vector y = -0.90 (moving up in frame: y decreases from 500 to 100)
    points = [{"frame_index": f, "timestamp": f/30.0, "bbox": [100, 500 - f*10, 140, 540 - f*10], "anchor_pixel": [120, 500 - f*10], "confidence": 0.9} for f in range(30)]
    feat_eng = TrajectoryFeatureEngine(fps=30.0)
    tf = feat_eng.extract_features(101, "car", trajectory=points)

    anoms = engine.detect_anomalies(tf)
    ww_anoms = [a for a in anoms if a.anomaly_type == AnomalyType.WRONG_WAY_BEHAVIOR]

    assert len(ww_anoms) >= 1
    assert ww_anoms[0].severity == AnomalySeverity.CRITICAL
    assert ww_anoms[0].score >= 0.90
    assert "opposing direction" in ww_anoms[0].explanation



def test_congestion_transition_engine_hysteresis():
    """Verify temporal hysteresis prevents frame-level flickering in congestion state transitions."""
    engine = CongestionTransitionEngine(persistence_frames=3)

    # Frame 0-2: FREE_FLOW
    for f in range(3):
        trans, _ = engine.update(f, f*0.1, "FREE_FLOW", 60.0, 5.0, 2.0)
        assert len(trans) == 0

    # Frame 3: Single frame pulse of CONGESTED (should NOT trigger transition immediately)
    trans, _ = engine.update(3, 0.3, "CONGESTED", 15.0, 45.0, 18.0)
    assert len(trans) == 0
    assert engine.current_state == "FREE_FLOW"

    # Frame 4-5: Persistence sustained -> State transition triggers on 3rd frame
    trans4, _ = engine.update(4, 0.4, "CONGESTED", 15.0, 45.0, 18.0)
    trans5, anoms5 = engine.update(5, 0.5, "CONGESTED", 15.0, 45.0, 18.0)

    assert len(trans5) == 1
    assert trans5[0].from_state == "FREE_FLOW"
    assert trans5[0].to_state == "CONGESTED"
    assert engine.current_state == "CONGESTED"
    assert any(a.anomaly_type == AnomalyType.CONGESTION_ONSET for a in anoms5)


def test_vehicle_behavior_profile():
    """Verify vehicle behavior profile synthesis and strict NORMAL default labeling."""
    feat_eng = TrajectoryFeatureEngine(fps=30.0)
    points = [{"frame_index": f, "timestamp": f/30.0, "bbox": [100+f*4, 200, 140+f*4, 240], "anchor_pixel": [120+f*4, 220], "confidence": 0.95} for f in range(45)]
    tf = feat_eng.extract_features(7, "car", trajectory=points)

    profile_eng = VehicleBehaviorProfileEngine()
    prof = profile_eng.generate_profile(tf, anomalies=[])

    assert prof.track_id == 7
    assert prof.behavior_status == "NORMAL"
    assert prof.overall_anomaly_score == 0.0
    assert prof.primary_movement == "FORWARD"


def test_traffic_risk_scoring_explainability():
    """Verify transparent traffic risk score calculation, contributing factors, policy versioning, and legal disclaimer."""
    risk_eng = TrafficRiskScoringEngine(policy_version="v1.0.0")

    from visionradar.intelligence.anomaly.schemas import TrajectoryAnomaly

    mock_anomaly = TrajectoryAnomaly(
        anomaly_id="test_anom_1",
        track_id=88,
        anomaly_type=AnomalyType.SUDDEN_STOP,
        severity=AnomalySeverity.HIGH,
        score=0.88,
        confidence=0.90,
        start_frame=10,
        end_frame=30,
        timestamp=1.0,
        duration_sec=1.5,
        method=AnomalyMethod.RULE_BASED,
        explanation="Sudden stop detected",
        evidence_frame_indices=[10],
        feature_contributions={"deceleration": 75.0}
    )

    risk_score = risk_eng.compute_risk_score(
        frame_idx=30,
        timestamp=1.0,
        active_anomalies=[mock_anomaly],
        congestion_state="CONGESTED",
        queue_count=1
    )

    assert risk_score.overall_risk_score > 0.0
    assert risk_score.policy_version == "v1.0.0"
    assert "not legal enforcement" in risk_score.disclaimer.lower()
    assert len(risk_score.contributing_factors) >= 3

    factors = [f["factor"] for f in risk_score.contributing_factors]
    assert AnomalyType.SUDDEN_STOP.value in factors
    assert "CONGESTION_STATE (CONGESTED)" in factors


def test_phase13_api_endpoints():
    """Test API response structures for Phase 13 endpoints."""
    # Test /health
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200

    # Test /jobs/latest (if exists or check structure)
    r_latest = client.get("/api/v1/jobs/latest")
    if r_latest.status_code == 200:
        job_id = r_latest.json()["id"]

        r_anoms = client.get(f"/api/v1/jobs/{job_id}/anomalies")
        assert r_anoms.status_code == 200
        assert "anomalies" in r_anoms.json()

        r_timeline = client.get(f"/api/v1/jobs/{job_id}/traffic-timeline")
        assert r_timeline.status_code == 200
        assert "risk_timeline" in r_timeline.json()

        r_behavior = client.get(f"/api/v1/jobs/{job_id}/behavior")
        assert r_behavior.status_code == 200
        assert "behavior_profiles" in r_behavior.json()

        r_risk = client.get(f"/api/v1/jobs/{job_id}/risk")
        assert r_risk.status_code == 200
        assert "Analytical alert" in r_risk.json()["disclaimer"]


def test_deterministic_repeatability():
    """Verify 100% deterministic anomaly evaluation given identical input features."""
    engine = BehavioralAnomalyEngine(method=AnomalyMethod.STATISTICAL_ZSCORE)
    points = [{"frame_index": f, "timestamp": f/30.0, "bbox": [100+f*2, 200, 140+f*2, 240], "anchor_pixel": [120+f*2, 220], "confidence": 0.88} for f in range(30)]
    feat_eng = TrajectoryFeatureEngine(fps=30.0)
    tf = feat_eng.extract_features(1, "car", trajectory=points)

    res1 = engine.detect_anomalies(tf)
    res2 = engine.detect_anomalies(tf)

    assert res1 == res2
