"""
VisionRadar Congestion State Transition Intelligence

Tracks temporal transitions between traffic congestion states (FREE_FLOW, MODERATE, HEAVY, CONGESTED, SEVERE)
with temporal hysteresis to prevent frame-level spamming, and emits corresponding congestion anomalies.
"""

from typing import List, Dict, Any, Optional
from visionradar.intelligence.anomaly.schemas import (
    CongestionTransition, TrajectoryAnomaly, AnomalyType, AnomalySeverity, AnomalyMethod
)


class CongestionTransitionEngine:
    """
    State machine for tracking multi-state congestion transitions with hysteresis.
    """

    STATE_HIERARCHY = {
        "FREE_FLOW": 0,
        "MODERATE": 1,
        "HEAVY": 2,
        "CONGESTED": 3,
        "SEVERE": 4
    }

    def __init__(self, persistence_frames: int = 5, policy_version: str = "v1.0.0"):
        self.persistence_frames = persistence_frames
        self.policy_version = policy_version
        self.current_state: str = "FREE_FLOW"
        self.pending_state: Optional[str] = None
        self.pending_count: int = 0
        self.transitions: List[CongestionTransition] = []
        self.active_queue_count: int = 0
        self.previous_queue_count: int = 0

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        congestion_level: str,
        avg_speed_kmh: float,
        density_veh_km: float,
        occupancy: float,
        queue_count: int = 0
    ) -> tuple[List[CongestionTransition], List[TrajectoryAnomaly]]:
        """
        Process a new frame's congestion metrics and emit transitions & anomalies when state changes safely.
        """
        new_transitions: List[CongestionTransition] = []
        anomalies: List[TrajectoryAnomaly] = []

        target_state = congestion_level if congestion_level in self.STATE_HIERARCHY else "FREE_FLOW"

        # Check state transition hysteresis
        if target_state != self.current_state:
            if target_state == self.pending_state:
                self.pending_count += 1
            else:
                self.pending_state = target_state
                self.pending_count = 1

            if self.pending_count >= self.persistence_frames:
                from_st = self.current_state
                to_st = target_state
                self.current_state = target_state
                self.pending_state = None
                self.pending_count = 0

                is_onset = self.STATE_HIERARCHY[to_st] > self.STATE_HIERARCHY[from_st]
                reason = f"Congestion worsened from {from_st} to {to_st}" if is_onset else f"Congestion eased from {from_st} to {to_st}"

                trans = CongestionTransition(
                    transition_id=f"trans_{frame_idx}_{from_st}_to_{to_st}",
                    from_state=from_st,
                    to_state=to_st,
                    timestamp=timestamp,
                    frame_index=frame_idx,
                    trigger_reason=reason,
                    avg_speed_kmh=avg_speed_kmh,
                    density_veh_km=density_veh_km,
                    occupancy=occupancy,
                    active_queue_count=queue_count,
                    policy_version=self.policy_version
                )
                self.transitions.append(trans)
                new_transitions.append(trans)

                # Emit anomaly if transition is significant
                if is_onset and self.STATE_HIERARCHY[to_st] >= self.STATE_HIERARCHY["HEAVY"]:
                    anomalies.append(TrajectoryAnomaly(
                        anomaly_id=f"anom_cg_onset_{frame_idx}",
                        track_id=0,
                        anomaly_type=AnomalyType.CONGESTION_ONSET,
                        severity=AnomalySeverity.HIGH if to_st in ["CONGESTED", "SEVERE"] else AnomalySeverity.MEDIUM,
                        score=0.85,
                        confidence=0.90,
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                        timestamp=timestamp,
                        duration_sec=0.0,
                        method=AnomalyMethod.RULE_BASED,
                        explanation=f"Traffic transition triggered CONGESTION_ONSET: shifted from {from_st} to {to_st}.",
                        evidence_frame_indices=[frame_idx],
                        feature_contributions={"occupancy": occupancy, "density_veh_km": density_veh_km},
                        policy_version=self.policy_version
                    ))
                elif not is_onset and self.STATE_HIERARCHY[from_st] >= self.STATE_HIERARCHY["HEAVY"]:
                    anomalies.append(TrajectoryAnomaly(
                        anomaly_id=f"anom_cg_dissip_{frame_idx}",
                        track_id=0,
                        anomaly_type=AnomalyType.CONGESTION_DISSIPATION,
                        severity=AnomalySeverity.LOW,
                        score=0.60,
                        confidence=0.88,
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                        timestamp=timestamp,
                        duration_sec=0.0,
                        method=AnomalyMethod.RULE_BASED,
                        explanation=f"Traffic transition triggered CONGESTION_DISSIPATION: shifted from {from_st} to {to_st}.",
                        evidence_frame_indices=[frame_idx],
                        feature_contributions={"occupancy": occupancy, "density_veh_km": density_veh_km},
                        policy_version=self.policy_version
                    ))
        else:
            self.pending_state = None
            self.pending_count = 0

        # Queue formation / breakdown monitoring
        if queue_count > 0 and self.previous_queue_count == 0:
            anomalies.append(TrajectoryAnomaly(
                anomaly_id=f"anom_q_form_{frame_idx}",
                track_id=0,
                anomaly_type=AnomalyType.QUEUE_FORMATION,
                severity=AnomalySeverity.MEDIUM,
                score=0.75,
                confidence=0.88,
                start_frame=frame_idx,
                end_frame=frame_idx,
                timestamp=timestamp,
                duration_sec=0.0,
                method=AnomalyMethod.RULE_BASED,
                explanation=f"Queue formation detected with {queue_count} stopped vehicle(s).",
                evidence_frame_indices=[frame_idx],
                feature_contributions={"queue_count": float(queue_count)},
                policy_version=self.policy_version
            ))
        elif queue_count == 0 and self.previous_queue_count > 0:
            anomalies.append(TrajectoryAnomaly(
                anomaly_id=f"anom_q_break_{frame_idx}",
                track_id=0,
                anomaly_type=AnomalyType.QUEUE_BREAKDOWN,
                severity=AnomalySeverity.LOW,
                score=0.55,
                confidence=0.85,
                start_frame=frame_idx,
                end_frame=frame_idx,
                timestamp=timestamp,
                duration_sec=0.0,
                method=AnomalyMethod.RULE_BASED,
                explanation="Queue breakdown detected: queue cleared.",
                evidence_frame_indices=[frame_idx],
                feature_contributions={"queue_count": 0.0},
                policy_version=self.policy_version
            ))

        self.previous_queue_count = queue_count
        return new_transitions, anomalies
