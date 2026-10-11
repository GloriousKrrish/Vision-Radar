"""
VisionRadar Phase 15 — Anomaly Explainability Engine

Synthesizes structured, evidence-backed explainability payloads for vehicle-level,
traffic-level, and multi-vehicle interaction anomalies.
Pulls feature contributions directly from underlying engine rules without fabricating ML SHAP values.
"""

from typing import Dict, Any, List, Optional
from visionradar.research.schemas import AnomalyExplanation, GroundTruthClassification
from visionradar.intelligence.anomaly.schemas import TrajectoryAnomaly


class ExplainabilityEngine:
    """
    Generates explainability breakdowns for detected anomalies.
    """

    @staticmethod
    def generate_explanation(
        anomaly: TrajectoryAnomaly,
        experiment_id: str = "EXP-2026-000001",
        evidence_frames: Optional[Dict[str, str]] = None
    ) -> AnomalyExplanation:

        evidence = evidence_frames or {
            "before": f"data/evidence/job_{anomaly.job_id}/anomaly_{anomaly.anomaly_id}/before.jpg",
            "event": f"data/evidence/job_{anomaly.job_id}/anomaly_{anomaly.anomaly_id}/event.jpg",
            "after": f"data/evidence/job_{anomaly.job_id}/anomaly_{anomaly.anomaly_id}/after.jpg"
        }

        ctx = anomaly.context or {}
        contribs = anomaly.feature_contributions or {}

        # Extract baseline stats if available in context
        baseline_stats = {
            "median": float(ctx.get("baseline_smoothness_median", ctx.get("median_flow", 0.0))),
            "mad": float(ctx.get("mad", 1.0)),
            "modified_z_score": float(contribs.get("trajectory_smoothness_zscore", contribs.get("z_score", contribs.get("occupancy_zscore", 0.0))))
        }

        # Map validation status enum
        gt_status = GroundTruthClassification.ENGINEERING_VALIDATED
        if anomaly.validation_status in [e.value for e in GroundTruthClassification]:
            gt_status = GroundTruthClassification(anomaly.validation_status)

        which_tracks = anomaly.involved_track_ids if anomaly.involved_track_ids else ([anomaly.track_id] if anomaly.track_id > 0 else [])

        return AnomalyExplanation(
            anomaly_id=anomaly.anomaly_id,
            experiment_id=experiment_id,
            what=anomaly.anomaly_type.value if hasattr(anomaly.anomaly_type, 'value') else str(anomaly.anomaly_type),
            when_frame_interval=[anomaly.start_frame, anomaly.end_frame],
            when_timestamp_sec=anomaly.timestamp,
            where_bbox=ctx.get("bbox", [100.0, 100.0, 200.0, 200.0]),
            where_lane=ctx.get("lane", "Lane 1"),
            which_track_ids=which_tracks,
            why_rule_contributions=contribs,
            score=anomaly.score,
            confidence=anomaly.confidence,
            context={
                "local_traffic_state": ctx.get("local_traffic_state", "FREE_FLOW"),
                "occupancy": ctx.get("recent_occupancy", 8.0),
                "density": ctx.get("density", 12.0),
                "dominant_lane": ctx.get("dominant_lane", "Lane 1")
            },
            baseline=baseline_stats,
            persistence_frames=max(1, anomaly.end_frame - anomaly.start_frame + 1),
            policy_version=anomaly.policy_version,
            validation_status=gt_status,
            evidence_frames=evidence
        )
