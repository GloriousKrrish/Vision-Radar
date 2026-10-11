"""
VisionRadar Explainable Traffic Risk Scoring Engine

Calculates analytical traffic risk scores (0-100) based on real-time vehicle anomalies,
congestion state, and queue formation with transparent, versioned contributing factor breakdown.
NOT for legal enforcement.
"""

from typing import List, Dict, Any, Optional
from visionradar.intelligence.anomaly.schemas import (
    TrafficRiskScore, TrajectoryAnomaly, AnomalyType, AnomalySeverity
)


class TrafficRiskScoringEngine:
    """
    Computes an analytical traffic risk score (0-100) with complete factor breakdown.
    """

    POLICY_VERSION = "v1.0.0"
    LEGAL_DISCLAIMER = "Analytical alert — not legal enforcement score"

    def __init__(self, policy_version: str = POLICY_VERSION):
        self.policy_version = policy_version

    def compute_risk_score(
        self,
        frame_idx: int,
        timestamp: float,
        active_anomalies: List[TrajectoryAnomaly],
        congestion_state: str = "FREE_FLOW",
        queue_count: int = 0
    ) -> TrafficRiskScore:
        """
        Calculates frame-level traffic risk score and returns detailed factor breakdown.
        """
        score = 0.0
        factors: List[Dict[str, Any]] = []

        # 1. Evaluate Active Anomalies
        for anomaly in active_anomalies:
            added_score = 0.0
            atype = anomaly.anomaly_type

            if atype == AnomalyType.WRONG_WAY_BEHAVIOR:
                added_score = 40.0
            elif atype == AnomalyType.SUDDEN_STOP:
                added_score = 15.0
            elif atype == AnomalyType.ABNORMAL_STOP_DURATION:
                added_score = 15.0
            elif atype == AnomalyType.TRAJECTORY_DEVIATION:
                added_score = 7.0
            elif atype in [AnomalyType.UNUSUAL_LATERAL_MOVEMENT, AnomalyType.UNUSUAL_LANE_CHANGE]:
                added_score = 5.0
            elif atype == AnomalyType.UNUSUAL_VEHICLE_BEHAVIOR:
                added_score = 8.0
            else:
                added_score = 5.0

            score += added_score
            factors.append({
                "factor": atype.value,
                "contribution": added_score,
                "detail": anomaly.explanation,
                "track_id": anomaly.track_id
            })

        # 2. Evaluate Congestion State Context
        if congestion_state in ["CONGESTED", "SEVERE"]:
            cg_score = 15.0 if congestion_state == "SEVERE" else 10.0
            score += cg_score
            factors.append({
                "factor": f"CONGESTION_STATE ({congestion_state})",
                "contribution": cg_score,
                "detail": f"Highway congestion level is currently {congestion_state}"
            })

        # 3. Evaluate Queue Context
        if queue_count > 0:
            q_score = min(20.0, 10.0 + (queue_count * 2.0))
            score += q_score
            factors.append({
                "factor": "ACTIVE_QUEUE_PRESENCE",
                "contribution": q_score,
                "detail": f"Active queue detected with {queue_count} stopped vehicles"
            })

        final_score = min(100.0, round(score, 1))

        # Risk level categorization
        if final_score < 25.0:
            risk_level = "LOW"
        elif final_score < 50.0:
            risk_level = "MODERATE"
        elif final_score < 75.0:
            risk_level = "HIGH"
        else:
            risk_level = "SEVERE"

        return TrafficRiskScore(
            overall_risk_score=final_score,
            risk_level=risk_level,
            timestamp=round(timestamp, 2),
            frame_index=frame_idx,
            contributing_factors=factors,
            policy_version=self.policy_version,
            disclaimer=self.LEGAL_DISCLAIMER
        )
