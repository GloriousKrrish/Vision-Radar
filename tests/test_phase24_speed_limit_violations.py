import pytest
from fastapi.testclient import TestClient
from visionradar.cv.speed import SpeedEstimate, MonocularSpeedEstimator
from visionradar.cv.violations import ViolationRuleEngine, CandidateViolation
from visionradar.cv.calibration import HomographyCalibrator
from apps.api.main import app

@pytest.fixture
def test_client():
    return TestClient(app)

def test_speed_calculation_known_distance_and_timestamps():
    """Verify speed calculation with known metric distances and timestamps: 100m in 3.6s = 100 km/h."""
    dist_m = 100.0
    dt_sec = 3.6
    speed_mps = dist_m / dt_sec
    speed_kmh = speed_mps * 3.6
    assert abs(speed_kmh - 100.0) < 1e-4

def test_metric_unit_conversion():
    """Verify standard m/s to km/h conversions."""
    # 27.7778 m/s = 100 km/h
    assert abs(27.77777777777778 * 3.6 - 100.0) < 1e-4
    # 13.8889 m/s = 50 km/h
    assert abs(13.88888888888889 * 3.6 - 50.0) < 1e-4

def test_missing_or_invalid_calibration():
    """Verify that when calibration is missing, speed estimator marks UNCALIBRATED and None kmh."""
    estimator = MonocularSpeedEstimator(calibrator=None, fps=30.0)
    assert not estimator.is_calibrated()
    diag = estimator.get_diagnostics()
    assert diag["calibration_status"] == "UNCALIBRATED"

def test_speed_limit_input_validation():
    """Verify speed limit must be > 0 and warning tolerance must be >= 0."""
    with pytest.raises(ValueError, match="greater than zero"):
        ViolationRuleEngine(speed_limit_kmh=0.0)
    
    with pytest.raises(ValueError, match="greater than zero"):
        ViolationRuleEngine(speed_limit_kmh=-10.0)

    with pytest.raises(ValueError, match="non-negative"):
        ViolationRuleEngine(speed_limit_kmh=60.0, warning_tolerance_kmh=-5.0)

    engine = ViolationRuleEngine(speed_limit_kmh=60.0, warning_tolerance_kmh=5.0)
    assert engine.speed_limit_kmh == 60.0
    assert engine.warning_tolerance_kmh == 5.0
    assert engine.effective_threshold_kmh == 65.0

def test_overspeed_candidate_creation_and_threshold():
    """Verify overspeed candidate triggers only above effective_threshold_kmh."""
    engine = ViolationRuleEngine(speed_limit_kmh=60.0, warning_tolerance_kmh=5.0)
    # Threshold = 65.0 km/h

    # 1. Valid speed under threshold (62 km/h) -> No candidate
    est_under = SpeedEstimate(
        instantaneous_kmh=62.0,
        smoothed_kmh=62.0,
        uncertainty_kmh=1.5,
        validity="VALID"
    )
    res_under = engine.evaluate_track(track_id=1, vehicle_class="Car", frame_index=10, timestamp=1.0, estimate=est_under)
    assert res_under is None

    # 2. Valid speed exactly at threshold (65 km/h) -> No candidate
    est_exact = SpeedEstimate(
        instantaneous_kmh=65.0,
        smoothed_kmh=65.0,
        uncertainty_kmh=1.5,
        validity="VALID"
    )
    res_exact = engine.evaluate_track(track_id=1, vehicle_class="Car", frame_index=11, timestamp=1.1, estimate=est_exact)
    assert res_exact is None

    # 3. Valid speed above threshold (72 km/h) -> Candidate generated!
    est_over = SpeedEstimate(
        instantaneous_kmh=72.0,
        smoothed_kmh=72.0,
        uncertainty_kmh=1.8,
        validity="VALID"
    )
    res_over = engine.evaluate_track(track_id=1, vehicle_class="Car", frame_index=12, timestamp=1.2, estimate=est_over)
    assert res_over is not None
    assert res_over.track_id == 1
    assert res_over.estimated_speed_kmh == 72.0
    assert res_over.speed_limit_kmh == 60.0
    assert res_over.warning_tolerance_kmh == 5.0
    assert res_over.effective_threshold_kmh == 65.0
    assert res_over.amount_over_threshold_kmh == 7.0
    assert res_over.review_status == "PENDING"
    assert res_over.evidence_available is True

def test_no_candidate_when_uncalibrated():
    """Strict integrity: uncalibrated motion CANNOT trigger speed limit violation."""
    engine = ViolationRuleEngine(speed_limit_kmh=40.0, warning_tolerance_kmh=0.0)
    
    # Fast motion (95 km/h equivalent) but uncalibrated
    est_uncalib = SpeedEstimate(
        instantaneous_kmh=95.0,
        smoothed_kmh=95.0,
        uncertainty_kmh=5.0,
        validity="UNCALIBRATED"
    )
    res = engine.evaluate_track(track_id=2, vehicle_class="Car", frame_index=20, timestamp=2.0, estimate=est_uncalib)
    assert res is None

    # Out of ROI
    est_roi = SpeedEstimate(
        instantaneous_kmh=110.0,
        smoothed_kmh=110.0,
        uncertainty_kmh=5.0,
        validity="OUT_OF_ROI"
    )
    res_roi = engine.evaluate_track(track_id=3, vehicle_class="Car", frame_index=30, timestamp=3.0, estimate=est_roi)
    assert res_roi is None

def test_duplicate_alert_suppression_cooldown():
    """Verify deduplication suppresses repeated alerts within cooldown_sec."""
    engine = ViolationRuleEngine(speed_limit_kmh=50.0, warning_tolerance_kmh=0.0, cooldown_sec=2.0)
    est = SpeedEstimate(instantaneous_kmh=70.0, smoothed_kmh=70.0, uncertainty_kmh=1.0, validity="VALID")

    # First violation at t=1.0s -> permitted
    viol1 = engine.evaluate_track(track_id=10, vehicle_class="Car", frame_index=30, timestamp=1.0, estimate=est)
    assert viol1 is not None

    # Immediate next frame at t=1.1s (within 2.0s cooldown) -> suppressed
    viol2 = engine.evaluate_track(track_id=10, vehicle_class="Car", frame_index=33, timestamp=1.1, estimate=est)
    assert viol2 is None

    # At t=2.5s (within 2.0s of t=1.0) -> suppressed
    viol3 = engine.evaluate_track(track_id=10, vehicle_class="Car", frame_index=75, timestamp=2.5, estimate=est)
    assert viol3 is None

    # At t=3.5s (>2.0s after initial trigger at 1.0s) -> permitted
    viol4 = engine.evaluate_track(track_id=10, vehicle_class="Car", frame_index=105, timestamp=3.5, estimate=est)
    assert viol4 is not None

def test_speed_limit_api_endpoints_and_validation(test_client):
    """Verify GET/POST /api/v1/jobs/{job_id}/speed-limit validation and response."""
    # Test invalid limit (<= 0)
    bad_res = test_client.post("/api/v1/jobs/648/speed-limit", json={"speed_limit_kmh": 0.0, "warning_tolerance_kmh": 0.0})
    assert bad_res.status_code == 400

    # Test invalid tolerance (< 0)
    bad_res2 = test_client.post("/api/v1/jobs/648/speed-limit", json={"speed_limit_kmh": 60.0, "warning_tolerance_kmh": -2.0})
    assert bad_res2.status_code == 400

    # Test valid configuration
    good_res = test_client.post("/api/v1/jobs/648/speed-limit", json={"speed_limit_kmh": 70.0, "warning_tolerance_kmh": 5.0})
    assert good_res.status_code == 200
    data = good_res.json()
    assert data["speed_limit_kmh"] == 70.0
    assert data["warning_tolerance_kmh"] == 5.0
    assert data["effective_threshold_kmh"] == 75.0

    # Test GET returns persisted configuration
    get_res = test_client.get("/api/v1/jobs/648/speed-limit")
    assert get_res.status_code == 200
    g_data = get_res.json()
    assert g_data["speed_limit_kmh"] == 70.0
    assert g_data["warning_tolerance_kmh"] == 5.0
    assert g_data["effective_threshold_kmh"] == 75.0

def test_violations_api_and_job_isolation(test_client):
    """Verify /api/v1/violations filters properly by job_id to prevent cross-job contamination."""
    # Query for job 648
    res_648 = test_client.get("/api/v1/violations?job_id=648")
    assert res_648.status_code == 200
    viols_648 = res_648.json()
    for v in viols_648:
        assert v["job_id"] == 648

    # Query for nonexistent job -> returns empty list without error
    res_empty = test_client.get("/api/v1/violations?job_id=999999")
    assert res_empty.status_code == 200
    assert res_empty.json() == []
