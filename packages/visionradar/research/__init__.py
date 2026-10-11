"""
VisionRadar Phase 15 — Research Workbench & Experiment Explainability Package
"""

from visionradar.research.schemas import (
    GroundTruthClassification, ExperimentStatus, FailureCategory,
    MetricValue, DatasetProvenance, ConfigurationSnapshot,
    FailureRecord, AnomalyExplanation, AblationRecord, ArtifactManifest
)
from visionradar.research.config_snapshot import ConfigurationSnapshotEngine
from visionradar.research.provenance import ProvenanceEngine
from visionradar.research.comparison import ExperimentComparisonEngine
from visionradar.research.ablation import AblationEngine
from visionradar.research.failure_cases import FailureCaseEngine
from visionradar.research.explainability import ExplainabilityEngine
from visionradar.research.report_generator import ResearchReportGenerator
from visionradar.research.registry import ResearchRegistry

__all__ = [
    "GroundTruthClassification",
    "ExperimentStatus",
    "FailureCategory",
    "MetricValue",
    "DatasetProvenance",
    "ConfigurationSnapshot",
    "FailureRecord",
    "AnomalyExplanation",
    "AblationRecord",
    "ArtifactManifest",
    "ConfigurationSnapshotEngine",
    "ProvenanceEngine",
    "ExperimentComparisonEngine",
    "AblationEngine",
    "FailureCaseEngine",
    "ExplainabilityEngine",
    "ResearchReportGenerator",
    "ResearchRegistry"
]
