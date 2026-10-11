"""
VisionRadar Phase 15 — Failure Case Workbench

Categorizes and tracks computer vision and traffic intelligence failure cases
across 11 standardized categories with full evidence links.
"""

from typing import List, Dict, Any, Optional
from visionradar.research.schemas import FailureRecord, FailureCategory, GroundTruthClassification


class FailureCaseEngine:
    """
    Registry and query interface for experiment failure cases.
    """

    CATEGORIES = [c.value for c in FailureCategory]

    @staticmethod
    def create_failure_record(
        failure_id: str,
        experiment_id: str,
        video_filename: str,
        frame_index: int,
        timestamp: float,
        category: str,
        description: str,
        severity: str = "MEDIUM",
        source: str = "CV_Pipeline",
        validation_status: GroundTruthClassification = GroundTruthClassification.ENGINEERING_VALIDATED,
        track_id: Optional[int] = None,
        evidence: Optional[Dict[str, Any]] = None
    ) -> FailureRecord:

        # Validate category string matches enum
        cat_enum = FailureCategory.INSUFFICIENT_TRAJECTORY_HISTORY
        for c in FailureCategory:
            if c.value.lower() == str(category).lower():
                cat_enum = c
                break

        return FailureRecord(
            failure_id=failure_id,
            experiment_id=experiment_id,
            video_filename=video_filename,
            frame_index=frame_index,
            timestamp=timestamp,
            category=cat_enum,
            description=description,
            severity=severity,
            source=source,
            validation_status=validation_status,
            track_id=track_id,
            evidence=evidence or {}
        )
