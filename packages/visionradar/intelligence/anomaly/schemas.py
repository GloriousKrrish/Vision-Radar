"""
VisionRadar Anomaly Intelligence Schemas (Phase 14)

Defines dataclasses, enums, feature metadata wrappers, and serialization helpers
for trajectory features, behavioral baseline, vehicle-level anomalies, traffic-level anomalies,
multi-vehicle interaction, state machine transitions, and evidence metadata.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional


class AnomalyType(str, Enum):
    # Vehicle-level anomalies
    TRAJECTORY_DEVIATION = "TRAJECTORY_DEVIATION"
    SUDDEN_STOP = "SUDDEN_STOP"
    ABNORMAL_STOP_DURATION = "ABNORMAL_STOP_DURATION"
    UNUSUAL_LANE_CHANGE = "UNUSUAL_LANE_CHANGE"
    WRONG_WAY_BEHAVIOR = "WRONG_WAY_BEHAVIOR"
    UNUSUAL_LATERAL_MOVEMENT = "UNUSUAL_LATERAL_MOVEMENT"
    ABNORMAL_LATERAL_MOVEMENT = "ABNORMAL_LATERAL_MOVEMENT"
    UNUSUAL_DIRECTION_CHANGE = "UNUSUAL_DIRECTION_CHANGE"
    UNUSUAL_MOVEMENT_PATTERN = "UNUSUAL_MOVEMENT_PATTERN"
    SUDDEN_DECELERATION = "SUDDEN_DECELERATION"
    
    # Traffic-level anomalies
    DENSITY_ANOMALY = "DENSITY_ANOMALY"
    FLOW_ANOMALY = "FLOW_ANOMALY"
    CONGESTION_TRANSITION_ANOMALY = "CONGESTION_TRANSITION_ANOMALY"
    QUEUE_FORMATION_ANOMALY = "QUEUE_FORMATION_ANOMALY"
    QUEUE_BREAKDOWN_ANOMALY = "QUEUE_BREAKDOWN_ANOMALY"
    CONGESTION_ONSET = "CONGESTION_ONSET"
    CONGESTION_DISSIPATION = "CONGESTION_DISSIPATION"
    QUEUE_FORMATION = "QUEUE_FORMATION"
    QUEUE_BREAKDOWN = "QUEUE_BREAKDOWN"
    UNUSUAL_FLOW_DROP = "UNUSUAL_FLOW_DROP"
    UNUSUAL_OCCUPANCY_SPIKE = "UNUSUAL_OCCUPANCY_SPIKE"
    UNUSUAL_DENSITY_SPIKE = "UNUSUAL_DENSITY_SPIKE"
    ABRUPT_CONGESTION_ONSET = "ABRUPT_CONGESTION_ONSET"
    ABRUPT_CONGESTION_DISSIPATION = "ABRUPT_CONGESTION_DISSIPATION"
    UNUSUAL_LANE_IMBALANCE = "UNUSUAL_LANE_IMBALANCE"

    # Multi-vehicle & Multi-signal anomalies
    MULTI_VEHICLE_INTERACTION_ANOMALY = "MULTI_VEHICLE_INTERACTION_ANOMALY"
    MULTI_SIGNAL_TRAFFIC_ANOMALY = "MULTI_SIGNAL_TRAFFIC_ANOMALY"
    UNUSUAL_VEHICLE_BEHAVIOR = "UNUSUAL_VEHICLE_BEHAVIOR"


class AnomalySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnomalyMethod(str, Enum):
    STATISTICAL_ROBUST_MAD = "STATISTICAL_ROBUST_MAD"
    LOCAL_CONTEXT_ZSCORE = "LOCAL_CONTEXT_ZSCORE"
    RULE_BASED = "RULE_BASED"
    STATISTICAL_ZSCORE = "STATISTICAL_ZSCORE"
    ISOLATION_FOREST = "ISOLATION_FOREST"
    MULTI_SIGNAL_FUSION = "MULTI_SIGNAL_FUSION"


class AnomalyState(str, Enum):
    NORMAL = "NORMAL"
    OBSERVED = "OBSERVED"
    SUSPICIOUS = "SUSPICIOUS"
    CONFIRMED_ANOMALY = "CONFIRMED_ANOMALY"
    COOLDOWN = "COOLDOWN"


@dataclass
class FeatureValue:
    """Every extracted trajectory feature must contain value, unit, source, validity, quality."""
    value: Any
    unit: str
    source: str                          # e.g., "TRAJECTORY_POINTS", "BYTE_TRACK", "LANE_ENGINE"
    validity: str                        # "VALID", "INVALID", "LOW_CONFIDENCE"
    quality: float                       # 0.0 - 1.0 quality metric

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrajectoryFeatures:
    track_id: int
    vehicle_class: str
    track_duration_sec: float
    total_trajectory_length: float
    displacement: float
    trajectory_smoothness: float
    direction_vector: List[float]
    direction_changes: int
    path_curvature: float
    lateral_movement: float
    longitudinal_movement: float
    lane_changes: int
    dwell_duration_sec: float
    stop_duration_sec: float
    movement_ratio: float
    acceleration_proxy: float
    deceleration_proxy: float
    mean_confidence: float
    track_quality_score: float
    neighboring_vehicle_count: int
    local_occupancy: float
    local_density: float
    local_traffic_state: str
    
    # Structured feature metadata dictionary mapping name -> FeatureValue
    structured_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrajectoryAnomaly:
    anomaly_id: str
    track_id: int
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float                        # 0.0 - 1.0 anomaly score
    confidence: float                   # 0.0 - 1.0 confidence
    start_frame: int
    end_frame: int
    timestamp: float
    duration_sec: float
    method: AnomalyMethod
    explanation: str
    evidence_frame_indices: List[int]
    feature_contributions: Dict[str, float]
    job_id: int = 1
    policy_version: str = "v1.0.0"
    validation_status: str = "ENGINEERING-VALIDATED" # "GT-VALIDATED", "ENGINEERING-VALIDATED", "QUALITATIVE", "NOT-EVALUABLE"
    context: Dict[str, Any] = field(default_factory=dict)
    involved_track_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["anomaly_type"] = self.anomaly_type.value
        d["severity"] = self.severity.value
        d["method"] = self.method.value
        return d


@dataclass
class CongestionTransition:
    transition_id: str
    from_state: str
    to_state: str
    timestamp: float
    frame_index: int
    trigger_reason: str
    avg_speed_kmh: float
    density_veh_km: float
    occupancy: float
    active_queue_count: int
    policy_version: str = "v1.0.0"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VehicleBehaviorProfile:
    track_id: int
    vehicle_class: str
    track_duration_sec: float
    assigned_lane: str
    primary_movement: str               # "FORWARD", "STOPPED", "TURNING", "WRONG_WAY"
    stops_count: int
    stop_duration_sec: float
    lane_changes_count: int
    trajectory_smoothness: float
    path_curvature: float
    overall_anomaly_score: float        # 0.0 - 1.0
    behavior_status: str                # "NORMAL" or "ABNORMAL"
    anomalies_flagged: List[str] = field(default_factory=list)
    risk_contribution: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrafficRiskScore:
    overall_risk_score: float           # 0 - 100
    risk_level: str                     # "LOW", "MODERATE", "HIGH", "SEVERE"
    timestamp: float
    frame_index: int
    contributing_factors: List[Dict[str, Any]]
    policy_version: str = "v1.0.0"
    disclaimer: str = "Analytical alert — not legal enforcement score"

    def to_dict(self) -> dict:
        return asdict(self)
