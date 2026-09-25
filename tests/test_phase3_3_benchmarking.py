import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.benchmarks.schema import GroundTruthObservation, SpeedObservation, SpeedComparisonRecord, BenchmarkMetrics
from visionradar.benchmarks.provider import SyntheticGroundTruthProvider
from visionradar.benchmarks.matcher import VehicleMatcher
from visionradar.benchmarks.engine import SpeedBenchmarkEngine

def test_ground_truth_provider():
    provider = SyntheticGroundTruthProvider(fps=29.97, num_frames=100)
    obs = provider.load_ground_truth()
    assert len(obs) > 0
    first = obs[0]
    assert isinstance(first, GroundTruthObservation)
    assert first.speed_kmh in [60.0, 80.0, 105.0, 45.0, 90.0]

def test_vehicle_matcher():
    provider = SyntheticGroundTruthProvider(fps=29.97, num_frames=50)
    gt_obs = provider.load_ground_truth()
    matcher = VehicleMatcher()

    fake_estimated_tracks = [
        {
            "track_id": 1,
            "speed_kmh": 60.5,
            "trajectory": [
                {"timestamp": 0.0, "world_pos": (2.0, 140.0)},
                {"timestamp": 0.1, "world_pos": (2.0, 138.3)}
            ]
        },
        {
            "track_id": 2,
            "speed_kmh": 79.8,
            "trajectory": [
                {"timestamp": 0.0, "world_pos": (6.0, 145.0)},
                {"timestamp": 0.1, "world_pos": (6.0, 142.8)}
            ]
        }
    ]

    matches = matcher.match_tracks(fake_estimated_tracks, gt_obs)
    assert len(matches) == 2
    assert matches[0].estimated_track_id == 1
    assert matches[1].estimated_track_id == 2

def test_metric_calculations_and_mape_edge_case():
    engine = SpeedBenchmarkEngine()
    comparisons = [
        SpeedComparisonRecord("b1", "r1", 1, "gt_1", 0.0, 60.5, 60.0, 0.5, 0.5, 0.0083, 2.0, 1.0),
        SpeedComparisonRecord("b1", "r1", 2, "gt_2", 0.0, 79.0, 80.0, -1.0, 1.0, 0.0125, 2.0, 1.0),
        SpeedComparisonRecord("b1", "r1", 3, "gt_3", 0.0, 104.0, 105.0, -1.0, 1.0, 0.0095, 2.5, 1.0)
    ]

    metrics = engine._calculate_aggregate_metrics(comparisons, rejected_count=0, vehicle_count=3)
    assert metrics.total_valid_samples == 3
    assert abs(metrics.mae_kmh - 0.83) < 0.05
    assert abs(metrics.bias_kmh - (-0.5)) < 0.05
    assert metrics.r2_score > 0.99

def test_empty_match_handling():
    engine = SpeedBenchmarkEngine()
    metrics = engine._calculate_aggregate_metrics([], rejected_count=5, vehicle_count=0)
    assert metrics.total_valid_samples == 0
    assert metrics.total_rejected_samples == 5
    assert metrics.mae_kmh == 0.0
    assert metrics.r2_score == 0.0

def test_uncertainty_coverage_evaluation():
    engine = SpeedBenchmarkEngine()
    comparisons = [
        SpeedComparisonRecord("b1", "r1", 1, "gt_1", 0.0, 60.5, 60.0, 0.5, 0.5, 0.0083, 2.0, 1.0),
        SpeedComparisonRecord("b1", "r1", 2, "gt_2", 0.0, 79.0, 80.0, -1.0, 1.0, 0.0125, 2.0, 1.0),
        SpeedComparisonRecord("b1", "r1", 3, "gt_3", 0.0, 104.0, 105.0, -1.0, 1.0, 0.0095, 2.5, 1.0),
    ]

    cov = engine.evaluate_uncertainty_coverage(comparisons)
    assert cov.total_evaluated == 3
    assert cov.coverage_1sigma_pct == 100.0
    assert cov.coverage_2sigma_pct == 100.0

def test_calibration_sensitivity_experiment():
    engine = SpeedBenchmarkEngine()
    provider = SyntheticGroundTruthProvider(fps=29.97, num_frames=30)
    fake_tracks = [
        {"track_id": 1, "speed_kmh": 60.0, "trajectory": [{"timestamp": 0.0, "world_pos": (2.0, 140.0)}]},
        {"track_id": 2, "speed_kmh": 80.0, "trajectory": [{"timestamp": 0.0, "world_pos": (6.0, 145.0)}]}
    ]

    sens = engine.run_calibration_sensitivity_experiment(fake_tracks, provider)
    assert len(sens) > 0
    for s in sens:
        assert hasattr(s, "parameter_name")
        assert hasattr(s, "perturbation_pct")

def test_smoothing_ablation_experiment():
    engine = SpeedBenchmarkEngine()
    provider = SyntheticGroundTruthProvider(fps=29.97, num_frames=30)
    fake_tracks = [
        {"track_id": 1, "speed_kmh": 60.0, "trajectory": [{"timestamp": 0.0, "world_pos": (2.0, 140.0)}]},
        {"track_id": 2, "speed_kmh": 80.0, "trajectory": [{"timestamp": 0.0, "world_pos": (6.0, 145.0)}]}
    ]

    ablation = engine.run_smoothing_ablation_experiment(fake_tracks, provider)
    assert len(ablation) == 4
    methods = [a.method_name for a in ablation]
    assert "Raw Instantaneous Velocity" in methods
    assert "Kalman Smoothed Velocity" in methods

def test_observation_window_analysis():
    engine = SpeedBenchmarkEngine()
    provider = SyntheticGroundTruthProvider(fps=29.97, num_frames=30)
    fake_tracks = [
        {"track_id": 1, "speed_kmh": 60.0, "trajectory": [{"timestamp": 0.0, "world_pos": (2.0, 140.0)}, {"timestamp": 1.0, "world_pos": (2.0, 123.3)}]}
    ]

    windows = engine.run_observation_window_analysis(fake_tracks, provider, windows_sec=[0.5, 1.0, 2.0])
    assert len(windows) == 3
    assert windows[0].window_length_s == 0.5
