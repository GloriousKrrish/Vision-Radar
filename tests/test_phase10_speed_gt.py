import pytest
import numpy as np
from visionradar.cv.speed_gt import GroundTruthSpeedRecord, VehicleMatchResult, SpeedGTEvaluator
from visionradar.cv.speed import SpeedProvenance

def test_ground_truth_speed_record_schema():
    gt = GroundTruthSpeedRecord(
        dataset="BrnoCompSpeed",
        sequence_id="session0_left",
        vehicle_id="veh_001",
        timestamp=1.5,
        frame_index=45,
        speed_kmh=54.2,
        source="OPTO_SENSOR",
        measurement_type="ENTRY_EXIT_AVERAGE",
        confidence=1.0,
        camera_id="cam_left",
        license_metadata="VUTBR Research License"
    )
    d = gt.to_dict()
    assert d["dataset"] == "BrnoCompSpeed"
    assert d["vehicle_id"] == "veh_001"
    assert d["speed_kmh"] == 54.2
    assert d["source"] == "OPTO_SENSOR"

def test_vehicle_matching():
    est_provenance = [
        SpeedProvenance(
            job_id="job1", video_id="vid1", track_id=1, frame_index=10, timestamp=0.33,
            speed_method="KALMAN_SMOOTHED", speed_kmh=52.0, speed_mps=14.44, distance_m=5.0,
            elapsed_time_s=0.33, sample_count=10, trajectory_length_m=5.0, uncertainty_kmh=2.5,
            calibration_id="cal1", calibration_status="VALID", trajectory_quality=1.0, validity="VALID",
            invalid_reason=None
        ),
        SpeedProvenance(
            job_id="job1", video_id="vid1", track_id=1, frame_index=20, timestamp=0.66,
            speed_method="KALMAN_SMOOTHED", speed_kmh=52.5, speed_mps=14.58, distance_m=10.0,
            elapsed_time_s=0.66, sample_count=20, trajectory_length_m=10.0, uncertainty_kmh=2.5,
            calibration_id="cal1", calibration_status="VALID", trajectory_quality=1.0, validity="VALID",
            invalid_reason=None
        ),
        SpeedProvenance(
            job_id="job1", video_id="vid1", track_id=1, frame_index=30, timestamp=1.00,
            speed_method="KALMAN_SMOOTHED", speed_kmh=53.0, speed_mps=14.72, distance_m=15.0,
            elapsed_time_s=1.00, sample_count=30, trajectory_length_m=15.0, uncertainty_kmh=2.5,
            calibration_id="cal1", calibration_status="VALID", trajectory_quality=1.0, validity="VALID",
            invalid_reason=None
        )
    ]

    gt_records = [
        GroundTruthSpeedRecord("BrnoCompSpeed", "seq1", "gt_veh_A", 0.33, 10, 52.1, "OPTO_SENSOR", "INSTANTANEOUS", 1.0, "cam1", "MIT"),
        GroundTruthSpeedRecord("BrnoCompSpeed", "seq1", "gt_veh_A", 0.66, 20, 52.6, "OPTO_SENSOR", "INSTANTANEOUS", 1.0, "cam1", "MIT"),
        GroundTruthSpeedRecord("BrnoCompSpeed", "seq1", "gt_veh_A", 1.00, 30, 53.1, "OPTO_SENSOR", "INSTANTANEOUS", 1.0, "cam1", "MIT"),
    ]

    matches = SpeedGTEvaluator.match_tracks_to_gt(est_provenance, gt_records)
    assert len(matches) == 1
    assert matches[0].estimated_track_id == 1
    assert matches[0].ground_truth_vehicle_id == "gt_veh_A"
    assert matches[0].matching_method == "SPATIO_TEMPORAL"

def test_accuracy_metrics_computation():
    y_true = [50.0, 60.0, 70.0, 80.0, 90.0]
    y_pred = [51.0, 59.0, 71.0, 78.0, 92.0]

    metrics = SpeedGTEvaluator.compute_accuracy_metrics(y_true, y_pred)
    assert metrics["sample_count"] == 5
    assert metrics["mae"] == pytest.approx(1.4, abs=0.1)
    assert metrics["rmse"] == pytest.approx(1.483, abs=0.1)
    assert metrics["median_ae"] == pytest.approx(1.0, abs=0.1)
    assert "r2" in metrics

def test_uncertainty_coverage_evaluation():
    y_true = [50.0, 60.0, 70.0]
    y_pred = [51.0, 62.0, 68.0] # errors: 1.0, 2.0, 2.0
    uncertainties = [1.5, 1.5, 2.5] # 1.0 <= 1.5 (True), 2.0 <= 1.5 (False), 2.0 <= 2.5 (True)

    cov = SpeedGTEvaluator.evaluate_uncertainty_coverage(y_true, y_pred, uncertainties)
    assert cov["coverage_1sigma"] == pytest.approx(2/3, abs=0.01)
    assert cov["coverage_2sigma"] == 1.0
    assert "heuristic_note" in cov

def test_calibration_sensitivity_evaluation():
    y_true = [50.0, 60.0, 70.0]
    y_pred = [50.0, 60.0, 70.0]

    sens = SpeedGTEvaluator.evaluate_calibration_sensitivity(y_true, y_pred, perturbations=[-0.05, 0.0, 0.05])
    assert "0%" in sens
    assert sens["0%"]["mae"] == 0.0
    assert sens["+5%"]["mae"] > 0.0
