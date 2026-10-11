"""
VisionRadar Phase 15 — Research Workbench Schemas

Defines dataclasses, enums, metric provenance wrappers, configuration snapshots,
ablation descriptors, failure cases, anomaly explainability payloads, and report structures.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional


class GroundTruthClassification(str, Enum):
    GT_VALIDATED = "GT-VALIDATED"
    ENGINEERING_VALIDATED = "ENGINEERING-VALIDATED"
    QUALITATIVE = "QUALITATIVE"
    NOT_EVALUABLE = "NOT-EVALUABLE"


class ExperimentStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FailureCategory(str, Enum):
    SMALL_DISTANT_OBJECT = "small distant object"
    OCCLUSION = "occlusion"
    ID_FRAGMENTATION = "ID fragmentation"
    CALIBRATION_INSTABILITY = "calibration instability"
    PERSPECTIVE_DISTORTION = "perspective distortion"
    DETECTOR_MISS = "detector miss"
    FALSE_DETECTION = "false detection"
    ABNORMAL_TRAJECTORY = "abnormal trajectory"
    ANOMALY_FALSE_CANDIDATE = "anomaly false candidate"
    ACCIDENT_CLASSIFIER_AMBIGUITY = "accident-classifier ambiguity"
    INSUFFICIENT_TRAJECTORY_HISTORY = "insufficient trajectory history"


@dataclass
class MetricValue:
    """Every displayed metric wrapped with full provenance."""
    metric_name: str
    value: Any
    unit: str
    source: str
    experiment_id: str
    calculation_method: str
    ground_truth_status: GroundTruthClassification
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    aggregation_period: Optional[str] = "PER_VIDEO"
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ground_truth_status"] = self.ground_truth_status.value if isinstance(self.ground_truth_status, GroundTruthClassification) else str(self.ground_truth_status)
        return d


@dataclass
class DatasetProvenance:
    dataset_id: str
    dataset_name: str
    video_filename: str
    video_hash: str
    resolution: str
    fps: float
    duration_sec: float
    frame_count: int
    annotation_availability: str        # e.g., "NO_GT", "SPEED_GT_ONLY", "FULL_ANNOTATED"
    ground_truth_type: str
    ground_truth_coverage: str
    source: str
    license: str
    sampling_policy: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConfigurationSnapshot:
    config_hash: str
    detector: Dict[str, Any]
    tracker: Dict[str, Any]
    speed: Dict[str, Any]
    traffic_intelligence: Dict[str, Any]
    anomaly_intelligence: Dict[str, Any]
    accident_intelligence: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FailureRecord:
    failure_id: str
    experiment_id: str
    video_filename: str
    frame_index: int
    timestamp: float
    category: FailureCategory
    description: str
    severity: str                         # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    source: str
    validation_status: GroundTruthClassification
    track_id: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value if isinstance(self.category, FailureCategory) else str(self.category)
        d["validation_status"] = self.validation_status.value if isinstance(self.validation_status, GroundTruthClassification) else str(self.validation_status)
        return d


@dataclass
class AnomalyExplanation:
    anomaly_id: str
    experiment_id: str
    what: str                             # Anomaly type
    when_frame_interval: List[int]       # [start_frame, end_frame]
    when_timestamp_sec: float
    where_bbox: List[float]               # [x1, y1, x2, y2]
    where_lane: str
    which_track_ids: List[int]
    why_rule_contributions: Dict[str, float]
    score: float
    confidence: float
    context: Dict[str, Any]               # nearby vehicles, traffic state, density, occupancy
    baseline: Dict[str, float]            # median, MAD, modified Z-score
    persistence_frames: int
    policy_version: str
    validation_status: GroundTruthClassification
    evidence_frames: Dict[str, str]       # before, event, after image paths

    def to_dict(self) -> dict:
        d = asdict(self)
        d["validation_status"] = self.validation_status.value if isinstance(self.validation_status, GroundTruthClassification) else str(self.validation_status)
        return d


@dataclass
class AblationRecord:
    ablation_id: str
    experiment_id: str
    baseline_experiment_id: str
    parameter_name: str
    baseline_value: Any
    variant_value: Any
    metrics_diff: Dict[str, Any]
    runtime_diff_ms: float
    interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArtifactManifest:
    experiment_id: str
    manifest_path: str
    files: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)
