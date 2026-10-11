"""
VisionRadar Vehicle Behavior Profile Engine

Synthesizes comprehensive vehicle behavior profiles for individual track histories,
combining trajectory features, movement classifications, stop metrics, lane usage,
and anomaly flagging with transparent evidence rules.
"""

from typing import List, Dict, Any, Optional
from visionradar.intelligence.anomaly.schemas import (
    TrajectoryFeatures, TrajectoryAnomaly, VehicleBehaviorProfile
)


class VehicleBehaviorProfileEngine:
    """
    Generates structured behavior profiles for track histories.
    """

    @staticmethod
    def generate_profile(
        features: TrajectoryFeatures,
        anomalies: Optional[List[TrajectoryAnomaly]] = None,
        assigned_lane: str = "Unknown"
    ) -> VehicleBehaviorProfile:
        """
        Builds a VehicleBehaviorProfile from extracted features and detected track anomalies.
        """
        track_anomalies = [a for a in (anomalies or []) if a.track_id == features.track_id]

        # Determine primary movement
        if any(a.anomaly_type == "WRONG_WAY_BEHAVIOR" for a in track_anomalies):
            primary_mov = "WRONG_WAY"
        elif features.stop_duration_sec > 2.0 or features.deceleration_proxy > 60.0:
            primary_mov = "STOPPED" if features.stop_duration_sec > 2.0 else "DECELERATING"
        elif features.path_curvature > 0.60 or features.lane_changes > 1:
            primary_mov = "MANEUVERING"
        else:
            primary_mov = "FORWARD"

        # Compute max anomaly score for this track
        max_score = max([a.score for a in track_anomalies], default=0.0)

        # Label policy: strictly NORMAL unless anomalies flagged or score > 0.65
        is_abnormal = len(track_anomalies) > 0 or max_score >= 0.65
        behavior_status = "ABNORMAL" if is_abnormal else "NORMAL"

        flagged_names = [a.anomaly_type.value for a in track_anomalies]

        stops_count = 1 if features.stop_duration_sec > 0.5 else 0

        return VehicleBehaviorProfile(
            track_id=features.track_id,
            vehicle_class=features.vehicle_class,
            track_duration_sec=round(features.track_duration_sec, 2),
            assigned_lane=assigned_lane,
            primary_movement=primary_mov,
            stops_count=stops_count,
            stop_duration_sec=round(features.stop_duration_sec, 2),
            lane_changes_count=features.lane_changes,
            trajectory_smoothness=round(features.trajectory_smoothness, 3),
            path_curvature=round(features.path_curvature, 3),
            overall_anomaly_score=round(max_score, 2),
            behavior_status=behavior_status,
            anomalies_flagged=flagged_names,
            risk_contribution=round(max_score * 30.0, 1) if is_abnormal else 0.0
        )
