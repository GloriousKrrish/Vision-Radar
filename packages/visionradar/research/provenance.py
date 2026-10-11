"""
VisionRadar Phase 15 — Metric & Dataset Provenance Generator

Wraps metrics and dataset references with full provenance, ground-truth classification,
and calculation tracing.
"""

from typing import Dict, Any, Optional, List
from visionradar.research.schemas import (
    MetricValue, DatasetProvenance, GroundTruthClassification
)


class ProvenanceEngine:
    """
    Builds structured MetricValue and DatasetProvenance objects with complete traceability.
    """

    @staticmethod
    def wrap_metric(
        name: str,
        value: Any,
        unit: str,
        source: str,
        experiment_id: str,
        calculation_method: str,
        gt_status: GroundTruthClassification,
        confidence: Optional[float] = None,
        uncertainty: Optional[float] = None,
        aggregation_period: str = "PER_VIDEO",
        notes: Optional[str] = None
    ) -> MetricValue:
        return MetricValue(
            metric_name=name,
            value=value,
            unit=unit,
            source=source,
            experiment_id=experiment_id,
            calculation_method=calculation_method,
            ground_truth_status=gt_status,
            confidence=confidence,
            uncertainty=uncertainty,
            aggregation_period=aggregation_period,
            notes=notes
        )

    @staticmethod
    def build_dataset_provenance(
        dataset_id: str,
        video_filename: str,
        video_hash: str = "a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890",
        resolution: str = "1920x1080",
        fps: float = 30.0,
        duration_sec: float = 10.0,
        frame_count: int = 300,
        annotation_availability: str = "NO_GT",
        ground_truth_type: str = "ENGINEERING_RULES",
        ground_truth_coverage: str = "0%",
        source: str = "VisionRadar Benchmark Suite",
        license_str: str = "Research License",
        sampling_policy: str = "Sequential Decode"
    ) -> DatasetProvenance:

        return DatasetProvenance(
            dataset_id=dataset_id,
            dataset_name=f"Dataset for {video_filename}",
            video_filename=video_filename,
            video_hash=video_hash,
            resolution=resolution,
            fps=fps,
            duration_sec=duration_sec,
            frame_count=frame_count,
            annotation_availability=annotation_availability,
            ground_truth_type=ground_truth_type,
            ground_truth_coverage=ground_truth_coverage,
            source=source,
            license=license_str,
            sampling_policy=sampling_policy
        )
