from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class GroundTruthObservation:
    ground_truth_id: str
    timestamp: float
    vehicle_id: str
    speed_mps: float
    speed_kmh: float
    position_x: float
    position_y: float
    source: str = "SYNTHETIC_GROUND_TRUTH"
    confidence: float = 1.0
    annotation_timestamp: float = 0.0

@dataclass
class SpeedObservation:
    run_id: str
    video_id: str
    track_id: int
    timestamp: float
    speed_method: str  # REGRESSION, PATH_AVERAGE, INSTANTANEOUS
    speed_mps: float
    speed_kmh: float
    uncertainty_kmh: float
    sample_count: int
    distance_m: float
    elapsed_time_s: float
    validity: bool = True

@dataclass
class TrackMatch:
    estimated_track_id: int
    ground_truth_id: str
    matching_method: str  # SPATIAL_TEMPORAL_IOU, TRAJECTORY_SIMILARITY
    matching_score: float
    match_confidence: float

@dataclass
class SpeedComparisonRecord:
    benchmark_id: str
    run_id: str
    track_id: int
    ground_truth_id: str
    timestamp: float
    estimated_speed_kmh: float
    ground_truth_speed_kmh: float
    signed_error_kmh: float
    absolute_error_kmh: float
    relative_error: float
    uncertainty_kmh: float
    matching_confidence: float
    distance_from_camera_m: float = 50.0
    lane_id: str = "Lane 1"
    speed_method: str = "PATH_AVERAGE"
    validity: bool = True

@dataclass
class BenchmarkMetrics:
    total_vehicles_evaluated: int
    total_valid_samples: int
    total_rejected_samples: int
    mae_kmh: float
    rmse_kmh: float
    median_ae_kmh: float
    p95_ae_kmh: float
    mape_percent: float
    bias_kmh: float
    r2_score: float
    std_dev_error_kmh: float
    min_error_kmh: float
    max_error_kmh: float
    error_by_speed_range: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_by_distance_zone: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_by_lane: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class UncertaintyCoverageResult:
    coverage_1sigma_pct: float
    coverage_2sigma_pct: float
    coverage_3sigma_pct: float
    calibration_error: float
    total_evaluated: int

@dataclass
class SensitivityResult:
    parameter_name: str
    perturbation_pct: float
    perturbed_value: float
    mae_kmh: float
    rmse_kmh: float
    bias_kmh: float

@dataclass
class AblationResult:
    method_name: str
    mae_kmh: float
    rmse_kmh: float
    p95_ae_kmh: float
    bias_kmh: float
    valid_vehicles: int

@dataclass
class ObservationWindowResult:
    window_length_s: float
    mae_kmh: float
    rmse_kmh: float
    uncertainty_kmh: float
    sample_count: int

@dataclass
class FailureCase:
    track_id: int
    ground_truth_id: str
    ground_truth_speed_kmh: float
    estimated_speed_kmh: float
    absolute_error_kmh: float
    uncertainty_kmh: float
    trajectory_length_m: float
    lane: str
    distance_m: float
    calibration_region: str
    detection_count: int
    tracking_continuity: float
    possible_cause: str
