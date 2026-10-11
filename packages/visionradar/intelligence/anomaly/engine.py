"""
VisionRadar Behavioral Anomaly Engine (Phase 14)

Implements explainable statistical baseline (Median, MAD, Modified Z-score),
vehicle-level anomalies, local traffic context filtering, traffic-level anomalies,
multi-vehicle interaction analysis, temporal persistence state machine,
and multi-signal event correlation.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from visionradar.intelligence.anomaly.schemas import (
    TrajectoryFeatures, TrajectoryAnomaly, AnomalyType, AnomalySeverity,
    AnomalyMethod, AnomalyState
)


class TrackAnomalyStateMachine:
    """
    Temporal persistence state machine for anomaly finalization:
    NORMAL -> OBSERVED -> SUSPICIOUS -> CONFIRMED_ANOMALY -> COOLDOWN -> NORMAL
    Prevents single noisy frames from emitting transient alarms.
    """
    def __init__(
        self,
        track_id: int,
        anomaly_type: AnomalyType,
        min_observed_frames: int = 3,
        min_cooldown_frames: int = 15
    ):
        self.track_id = track_id
        self.anomaly_type = anomaly_type
        self.min_observed_frames = min_observed_frames
        self.min_cooldown_frames = min_cooldown_frames

        self.state = AnomalyState.NORMAL
        self.observed_counter = 0
        self.cooldown_counter = 0
        self.start_frame = 0
        self.peak_frame = 0
        self.start_time = 0.0
        self.peak_score = 0.0
        self.accumulated_contribs: Dict[str, float] = {}
        self.latest_explanation = ""

    def update(
        self,
        has_trigger: bool,
        score: float,
        frame_index: int,
        timestamp: float,
        explanation: str,
        contribs: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        finalized_payload = None

        if self.state == AnomalyState.NORMAL:
            if has_trigger:
                self.state = AnomalyState.OBSERVED
                self.observed_counter = 1
                self.start_frame = frame_index
                self.peak_frame = frame_index
                self.start_time = timestamp
                self.peak_score = score
                self.accumulated_contribs = contribs.copy()
                self.latest_explanation = explanation

        elif self.state == AnomalyState.OBSERVED:
            if has_trigger:
                self.observed_counter += 1
                if score > self.peak_score:
                    self.peak_score = score
                    self.peak_frame = frame_index
                if self.observed_counter >= self.min_observed_frames:
                    self.state = AnomalyState.CONFIRMED_ANOMALY
            else:
                self.state = AnomalyState.NORMAL
                self.observed_counter = 0

        elif self.state == AnomalyState.CONFIRMED_ANOMALY:
            if has_trigger:
                if score > self.peak_score:
                    self.peak_score = score
                    self.peak_frame = frame_index
                self.latest_explanation = explanation
            else:
                # Finalize Confirmed Anomaly
                duration = max(0.1, timestamp - self.start_time)
                finalized_payload = {
                    "start_frame": self.start_frame,
                    "end_frame": frame_index,
                    "peak_frame": self.peak_frame,
                    "timestamp": self.start_time,
                    "duration_sec": round(duration, 2),
                    "peak_score": round(self.peak_score, 2),
                    "explanation": self.latest_explanation,
                    "feature_contributions": self.accumulated_contribs
                }
                self.state = AnomalyState.COOLDOWN
                self.cooldown_counter = 0

        elif self.state == AnomalyState.COOLDOWN:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.min_cooldown_frames:
                self.state = AnomalyState.NORMAL
                self.observed_counter = 0
                self.cooldown_counter = 0

        return finalized_payload


class BehavioralAnomalyEngine:
    """
    Advanced Behavioral Anomaly Engine (Phase 14 Compliant).
    """
    def __init__(
        self,
        method: AnomalyMethod = AnomalyMethod.STATISTICAL_ROBUST_MAD,
        sensitivity_threshold: float = 0.65,
        policy_version: str = "v1.0.0"
    ):
        self.method = method
        self.sensitivity_threshold = sensitivity_threshold
        self.policy_version = policy_version
        self.state_machines: Dict[Tuple[int, str], TrackAnomalyStateMachine] = {}

    def calculate_robust_baseline(
        self,
        population: List[TrajectoryFeatures],
        feature_name: str
    ) -> Tuple[float, float]:
        """
        Calculates robust baseline median and Median Absolute Deviation (MAD).
        """
        vals = []
        for feat in population:
            val = getattr(feat, feature_name, None)
            if val is not None and isinstance(val, (int, float)):
                vals.append(float(val))
        
        if not vals or len(vals) < 3:
            return 0.0, 1.0

        vals_arr = np.array(vals, dtype=np.float32)
        med = float(np.median(vals_arr))
        mad = float(np.median(np.abs(vals_arr - med)))
        if mad < 1e-4:
            mad = 1.0  # Prevent division by zero
        return med, mad

    def compute_modified_zscore(self, val: float, median: float, mad: float) -> float:
        """
        Modified Z-score = 0.6745 * (val - median) / MAD
        """
        return 0.6745 * (val - median) / max(1e-4, mad)

    def detect_anomalies(
        self,
        features: TrajectoryFeatures,
        population_features: Optional[List[TrajectoryFeatures]] = None,
        job_id: int = 1,
        current_frame: int = 0,
        timestamp: float = 0.0,
        context_traffic_state: str = "FREE_FLOW"
    ) -> List[TrajectoryAnomaly]:
        anomalies: List[TrajectoryAnomaly] = []
        if not features:
            return anomalies

        pop = population_features or [features]
        tid = features.track_id

        # -------------------------------------------------------------
        # 1. VEHICLE-LEVEL ANOMALIES (Context-Aware)
        # -------------------------------------------------------------

        # Baseline stats across population
        med_smooth, mad_smooth = self.calculate_robust_baseline(pop, "trajectory_smoothness")
        med_lat, mad_lat = self.calculate_robust_baseline(pop, "lateral_movement")
        med_curv, mad_curv = self.calculate_robust_baseline(pop, "path_curvature")
        med_stop, mad_stop = self.calculate_robust_baseline(pop, "stop_duration_sec")

        # A. TRAJECTORY_DEVIATION
        z_smooth = self.compute_modified_zscore(features.trajectory_smoothness, med_smooth, mad_smooth)
        z_curv = self.compute_modified_zscore(features.path_curvature, med_curv, mad_curv)
        if (features.trajectory_smoothness < 0.60 or z_curv > 3.0) and features.total_trajectory_length > 50.0:
            score = min(0.95, round(max(0.65, abs(z_smooth) * 0.15 + z_curv * 0.10), 2))
            contribs = {"trajectory_smoothness_zscore": round(z_smooth, 2), "path_curvature_zscore": round(z_curv, 2)}
            anomalies.append(TrajectoryAnomaly(
                anomaly_id=f"anom_td_job{job_id}_trk{tid}_f{current_frame}",
                track_id=tid,
                job_id=job_id,
                anomaly_type=AnomalyType.TRAJECTORY_DEVIATION,
                severity=AnomalySeverity.MEDIUM if score < 0.80 else AnomalySeverity.HIGH,
                score=score,
                confidence=0.88,
                start_frame=current_frame,
                end_frame=current_frame,
                timestamp=timestamp,
                duration_sec=features.track_duration_sec,
                method=self.method,
                explanation=f"Track #{tid} ({features.vehicle_class}) trajectory smoothness ({features.trajectory_smoothness:.2f}) is {abs(z_smooth):.1f} MADs below baseline in context '{context_traffic_state}'.",
                evidence_frame_indices=[current_frame],
                feature_contributions=contribs,
                policy_version=self.policy_version,
                context={"local_traffic_state": context_traffic_state, "baseline_smoothness_median": med_smooth}
            ))

        # B. ABNORMAL_LATERAL_MOVEMENT
        z_lat = self.compute_modified_zscore(features.lateral_movement, med_lat, mad_lat)
        if features.lateral_movement > 60.0 or z_lat > 3.0:
            score = min(0.95, round(0.60 + z_lat * 0.08, 2))
            contribs = {"lateral_movement_px": features.lateral_movement, "z_score": round(z_lat, 2)}
            anomalies.append(TrajectoryAnomaly(
                anomaly_id=f"anom_alm_job{job_id}_trk{tid}_f{current_frame}",
                track_id=tid,
                job_id=job_id,
                anomaly_type=AnomalyType.ABNORMAL_LATERAL_MOVEMENT,
                severity=AnomalySeverity.MEDIUM if score < 0.80 else AnomalySeverity.HIGH,
                score=score,
                confidence=0.86,
                start_frame=current_frame,
                end_frame=current_frame,
                timestamp=timestamp,
                duration_sec=features.track_duration_sec,
                method=self.method,
                explanation=f"Track #{tid} lateral displacement ({features.lateral_movement:.1f} px) is {z_lat:.1f} MADs above local lane baseline.",
                evidence_frame_indices=[current_frame],
                feature_contributions=contribs,
                policy_version=self.policy_version,
                context={"local_traffic_state": context_traffic_state}
            ))

        # C. UNUSUAL_LANE_CHANGE
        if features.lane_changes >= 2:
            score = min(0.95, round(0.65 + features.lane_changes * 0.10, 2))
            contribs = {"lane_changes": float(features.lane_changes)}
            anomalies.append(TrajectoryAnomaly(
                anomaly_id=f"anom_ulc_job{job_id}_trk{tid}_f{current_frame}",
                track_id=tid,
                job_id=job_id,
                anomaly_type=AnomalyType.UNUSUAL_LANE_CHANGE,
                severity=AnomalySeverity.MEDIUM,
                score=score,
                confidence=0.85,
                start_frame=current_frame,
                end_frame=current_frame,
                timestamp=timestamp,
                duration_sec=features.track_duration_sec,
                method=self.method,
                explanation=f"Track #{tid} executed {features.lane_changes} rapid lane changes across trajectory.",
                evidence_frame_indices=[current_frame],
                feature_contributions=contribs,
                policy_version=self.policy_version,
                context={"local_traffic_state": context_traffic_state}
            ))

        # D. ABNORMAL_STOP_DURATION (Context-Aware: Only flag if NOT in stop-and-go/congested traffic)
        if features.stop_duration_sec >= 3.0 and context_traffic_state not in ["CONGESTED", "STOP_AND_GO"]:
            score = min(0.95, round(0.60 + (features.stop_duration_sec / 10.0), 2))
            contribs = {"stop_duration_sec": features.stop_duration_sec}
            anomalies.append(TrajectoryAnomaly(
                anomaly_id=f"anom_asd_job{job_id}_trk{tid}_f{current_frame}",
                track_id=tid,
                job_id=job_id,
                anomaly_type=AnomalyType.ABNORMAL_STOP_DURATION,
                severity=AnomalySeverity.HIGH if features.stop_duration_sec > 6.0 else AnomalySeverity.MEDIUM,
                score=score,
                confidence=0.90,
                start_frame=current_frame,
                end_frame=current_frame,
                timestamp=timestamp,
                duration_sec=features.stop_duration_sec,
                method=self.method,
                explanation=f"Track #{tid} stationary for {features.stop_duration_sec:.1f}s in free-flow/moderate context '{context_traffic_state}'.",
                evidence_frame_indices=[current_frame],
                feature_contributions=contribs,
                policy_version=self.policy_version,
                context={"local_traffic_state": context_traffic_state}
            ))

        # E. SUDDEN_DECELERATION
        if features.deceleration_proxy > 60.0:
            score = min(0.95, round(0.65 + (features.deceleration_proxy / 200.0), 2))
            contribs = {"deceleration_proxy": features.deceleration_proxy}
            anomalies.append(TrajectoryAnomaly(
                anomaly_id=f"anom_dec_job{job_id}_trk{tid}_f{current_frame}",
                track_id=tid,
                job_id=job_id,
                anomaly_type=AnomalyType.SUDDEN_DECELERATION,
                severity=AnomalySeverity.HIGH,
                score=score,
                confidence=0.89,
                start_frame=current_frame,
                end_frame=current_frame,
                timestamp=timestamp,
                duration_sec=features.track_duration_sec,
                method=self.method,
                explanation=f"Track #{tid} executed sharp deceleration ({features.deceleration_proxy:.1f} px/s).",
                evidence_frame_indices=[current_frame],
                feature_contributions=contribs,
                policy_version=self.policy_version,
                context={"local_traffic_state": context_traffic_state}
            ))

        # F. WRONG_WAY_BEHAVIOR
        if features.direction_vector[1] < -0.80:
            contribs = {"direction_vector_y": abs(features.direction_vector[1])}
            anomalies.append(TrajectoryAnomaly(
                anomaly_id=f"anom_ww_job{job_id}_trk{tid}_f{current_frame}",
                track_id=tid,
                job_id=job_id,
                anomaly_type=AnomalyType.WRONG_WAY_BEHAVIOR,
                severity=AnomalySeverity.CRITICAL,
                score=0.98,
                confidence=0.95,
                start_frame=current_frame,
                end_frame=current_frame,
                timestamp=timestamp,
                duration_sec=features.track_duration_sec,
                method=AnomalyMethod.RULE_BASED,
                explanation=f"Track #{tid} moving opposing direction to flow vector (vy={features.direction_vector[1]:.2f}).",
                evidence_frame_indices=[current_frame],
                feature_contributions=contribs,
                policy_version=self.policy_version,
                context={"local_traffic_state": context_traffic_state}
            ))

        return anomalies

    def detect_traffic_level_anomalies(
        self,
        job_id: int,
        frame_index: int,
        timestamp: float,
        flow_history: List[float],
        occupancy_history: List[float],
        lane_occupancies: Dict[str, int],
        congestion_state_change: Optional[Tuple[str, str]] = None
    ) -> List[TrajectoryAnomaly]:
        """
        Detects traffic-level anomalies (Part 8 & 11 Compliant):
        - UNUSUAL_FLOW_DROP
        - UNUSUAL_OCCUPANCY_SPIKE
        - ABRUPT_CONGESTION_ONSET
        - ABRUPT_CONGESTION_DISSIPATION
        - UNUSUAL_LANE_IMBALANCE
        """
        traffic_anomalies: List[TrajectoryAnomaly] = []

        # 1. UNUSUAL_FLOW_DROP
        if len(flow_history) >= 5:
            recent_flow = flow_history[-1]
            med_flow = float(np.median(flow_history[:-1]))
            if med_flow >= 5.0 and recent_flow <= 0.3 * med_flow:
                drop_pct = round((1.0 - recent_flow / med_flow) * 100.0, 1)
                traffic_anomalies.append(TrajectoryAnomaly(
                    anomaly_id=f"anom_flow_drop_job{job_id}_f{frame_index}",
                    track_id=0,
                    job_id=job_id,
                    anomaly_type=AnomalyType.UNUSUAL_FLOW_DROP,
                    severity=AnomalySeverity.HIGH,
                    score=0.85,
                    confidence=0.88,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    timestamp=timestamp,
                    duration_sec=1.0,
                    method=self.method,
                    explanation=f"Traffic flow dropped abruptly by {drop_pct}% from baseline median ({med_flow:.1f} veh/min -> {recent_flow:.1f} veh/min).",
                    evidence_frame_indices=[frame_index],
                    feature_contributions={"flow_drop_pct": drop_pct, "baseline_median": med_flow},
                    policy_version=self.policy_version,
                    context={"recent_flow": recent_flow, "median_flow": med_flow}
                ))

        # 2. UNUSUAL_OCCUPANCY_SPIKE
        if len(occupancy_history) >= 5:
            occ_arr = np.array(occupancy_history, dtype=np.float32)
            rec_occ = float(occ_arr[-1])
            med_occ = float(np.median(occ_arr[:-1]))
            mad_occ = float(np.median(np.abs(occ_arr[:-1] - med_occ))) or 1.0
            z_occ = (rec_occ - med_occ) / mad_occ
            if z_occ > 3.5 and rec_occ >= 10.0:
                traffic_anomalies.append(TrajectoryAnomaly(
                    anomaly_id=f"anom_occ_spike_job{job_id}_f{frame_index}",
                    track_id=0,
                    job_id=job_id,
                    anomaly_type=AnomalyType.UNUSUAL_OCCUPANCY_SPIKE,
                    severity=AnomalySeverity.MEDIUM,
                    score=min(0.95, round(0.60 + z_occ * 0.08, 2)),
                    confidence=0.87,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    timestamp=timestamp,
                    duration_sec=1.0,
                    method=self.method,
                    explanation=f"Vehicle occupancy surged to {rec_occ:.0f} vehicles ({z_occ:.1f} MADs above baseline).",
                    evidence_frame_indices=[frame_index],
                    feature_contributions={"occupancy_zscore": round(z_occ, 2), "recent_occupancy": rec_occ},
                    policy_version=self.policy_version,
                    context={"recent_occupancy": rec_occ, "median_occupancy": med_occ}
                ))

        # 3. ABRUPT_CONGESTION_ONSET & DISSIPATION
        if congestion_state_change:
            from_st, to_st = congestion_state_change
            if from_st in ["FREE_FLOW", "MODERATE"] and to_st in ["CONGESTED", "STOP_AND_GO"]:
                traffic_anomalies.append(TrajectoryAnomaly(
                    anomaly_id=f"anom_cg_onset_job{job_id}_f{frame_index}",
                    track_id=0,
                    job_id=job_id,
                    anomaly_type=AnomalyType.ABRUPT_CONGESTION_ONSET,
                    severity=AnomalySeverity.HIGH,
                    score=0.88,
                    confidence=0.90,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    timestamp=timestamp,
                    duration_sec=1.0,
                    method=AnomalyMethod.RULE_BASED,
                    explanation=f"Traffic state transitioned abruptly from '{from_st}' to '{to_st}'.",
                    evidence_frame_indices=[frame_index],
                    feature_contributions={"from_state": 1.0, "to_state": 3.0},
                    policy_version=self.policy_version,
                    context={"from_state": from_st, "to_state": to_st}
                ))
            elif from_st in ["CONGESTED", "STOP_AND_GO"] and to_st == "FREE_FLOW":
                traffic_anomalies.append(TrajectoryAnomaly(
                    anomaly_id=f"anom_cg_diss_job{job_id}_f{frame_index}",
                    track_id=0,
                    job_id=job_id,
                    anomaly_type=AnomalyType.ABRUPT_CONGESTION_DISSIPATION,
                    severity=AnomalySeverity.LOW,
                    score=0.70,
                    confidence=0.85,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    timestamp=timestamp,
                    duration_sec=1.0,
                    method=AnomalyMethod.RULE_BASED,
                    explanation=f"Congestion dissipated rapidly from '{from_st}' to '{to_st}'.",
                    evidence_frame_indices=[frame_index],
                    feature_contributions={"from_state": 3.0, "to_state": 1.0},
                    policy_version=self.policy_version,
                    context={"from_state": from_st, "to_state": to_st}
                ))

        # 4. UNUSUAL_LANE_IMBALANCE
        if lane_occupancies and len(lane_occupancies) >= 2:
            total_lane_veh = sum(lane_occupancies.values())
            if total_lane_veh >= 8:
                max_lane, max_cnt = max(lane_occupancies.items(), key=lambda x: x[1])
                imbalance_ratio = max_cnt / max(1, total_lane_veh)
                if imbalance_ratio >= 0.80:
                    traffic_anomalies.append(TrajectoryAnomaly(
                        anomaly_id=f"anom_lane_imb_job{job_id}_f{frame_index}",
                        track_id=0,
                        job_id=job_id,
                        anomaly_type=AnomalyType.UNUSUAL_LANE_IMBALANCE,
                        severity=AnomalySeverity.MEDIUM,
                        score=round(imbalance_ratio, 2),
                        confidence=0.84,
                        start_frame=frame_index,
                        end_frame=frame_index,
                        timestamp=timestamp,
                        duration_sec=1.0,
                        method=self.method,
                        explanation=f"Severe lane utilization imbalance detected: {imbalance_ratio*100:.1f}% of vehicles concentrated in '{max_lane}'.",
                        evidence_frame_indices=[frame_index],
                        feature_contributions={"imbalance_ratio": round(imbalance_ratio, 2)},
                        policy_version=self.policy_version,
                        context={"lane_occupancies": lane_occupancies, "dominant_lane": max_lane}
                    ))

        return traffic_anomalies

    def detect_multi_vehicle_interactions(
        self,
        job_id: int,
        frame_index: int,
        timestamp: float,
        active_tracks: List[Dict[str, Any]]
    ) -> List[TrajectoryAnomaly]:
        """
        Multi-vehicle interaction analysis (Part 9 Compliant):
        Detects sudden trajectory convergence or close proximity interaction among nearby tracks.
        """
        interaction_anomalies: List[TrajectoryAnomaly] = []
        if not active_tracks or len(active_tracks) < 2:
            return interaction_anomalies

        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                t1 = active_tracks[i]
                t2 = active_tracks[j]
                b1 = t1.get("bbox") or [0, 0, 0, 0]
                b2 = t2.get("bbox") or [0, 0, 0, 0]

                c1 = [(b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0]
                c2 = [(b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0]

                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

                # Check close proximity (< 45 px) with high relative velocity / deceleration
                s1 = t1.get("speed_kmh") or 0.0
                s2 = t2.get("speed_kmh") or 0.0

                if dist < 45.0 and (s1 == 0.0 or s2 == 0.0 or abs(s1 - s2) > 30.0):
                    tid1 = t1.get("track_id", 0)
                    tid2 = t2.get("track_id", 0)
                    interaction_anomalies.append(TrajectoryAnomaly(
                        anomaly_id=f"anom_mvi_job{job_id}_trk{tid1}_{tid2}_f{frame_index}",
                        track_id=tid1,
                        job_id=job_id,
                        anomaly_type=AnomalyType.MULTI_VEHICLE_INTERACTION_ANOMALY,
                        severity=AnomalySeverity.HIGH,
                        score=0.86,
                        confidence=0.87,
                        start_frame=frame_index,
                        end_frame=frame_index,
                        timestamp=timestamp,
                        duration_sec=1.0,
                        method=AnomalyMethod.MULTI_SIGNAL_FUSION,
                        explanation=f"Multi-vehicle proximity convergence detected between Track #{tid1} and Track #{tid2} (distance: {dist:.1f} px).",
                        evidence_frame_indices=[frame_index],
                        feature_contributions={"proximity_px": round(dist, 1), "speed_delta_kmh": round(abs(s1 - s2), 1)},
                        policy_version=self.policy_version,
                        context={"interacted_tracks": [tid1, tid2], "proximity_distance_px": round(dist, 1)},
                        involved_track_ids=[tid1, tid2]
                    ))

        return interaction_anomalies

    def correlate_events(
        self,
        job_id: int,
        frame_index: int,
        timestamp: float,
        events: List[Any],
        anomalies: List[TrajectoryAnomaly]
    ) -> List[TrajectoryAnomaly]:
        """
        Event Correlation Engine (Part 11 Compliant):
        Correlates multiple signals temporally co-occurring without inferring unproven causality.
        """
        correlated: List[TrajectoryAnomaly] = []
        if not events and not anomalies:
            return correlated

        # Group by track ID
        track_signals: Dict[int, List[str]] = {}
        for ev in events:
            tid = getattr(ev, "track_id", 0)
            if tid not in track_signals:
                track_signals[tid] = []
            track_signals[tid].append(f"event:{getattr(ev, 'event_type', 'unknown')}")

        for anom in anomalies:
            tid = anom.track_id
            if tid not in track_signals:
                track_signals[tid] = []
            track_signals[tid].append(f"anomaly:{anom.anomaly_type.value}")

        for tid, signals in track_signals.items():
            if len(signals) >= 2 and tid > 0:
                sig_str = ", ".join(signals)
                correlated.append(TrajectoryAnomaly(
                    anomaly_id=f"anom_msta_job{job_id}_trk{tid}_f{frame_index}",
                    track_id=tid,
                    job_id=job_id,
                    anomaly_type=AnomalyType.MULTI_SIGNAL_TRAFFIC_ANOMALY,
                    severity=AnomalySeverity.HIGH,
                    score=0.92,
                    confidence=0.91,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    timestamp=timestamp,
                    duration_sec=1.0,
                    method=AnomalyMethod.MULTI_SIGNAL_FUSION,
                    explanation=f"Multiple traffic signals temporally co-occurred for Track #{tid}: [{sig_str}].",
                    evidence_frame_indices=[frame_index],
                    feature_contributions={"signal_count": float(len(signals))},
                    policy_version=self.policy_version,
                    context={"co_occurring_signals": signals},
                    involved_track_ids=[tid]
                ))

        return correlated
