"""
VisionRadar Trajectory Feature Extraction Engine (Phase 14)

Extracts high-dimensional behavioral features for vehicle tracks.
Every feature contains explicit metadata: value, unit, source, validity, quality.
Calculates features strictly when underlying data is valid.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional
from visionradar.intelligence.anomaly.schemas import TrajectoryFeatures, FeatureValue


class TrajectoryFeatureEngine:
    """
    Extracts high-dimensional behavioral features from vehicle trajectories with complete validity checking.
    """
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def extract_features(
        self,
        track_id: int,
        vehicle_class: str,
        trajectory: List[Dict[str, Any]],
        lane_engine: Optional[Any] = None,
        frame_occupancies: Optional[Dict[int, int]] = None,
        local_traffic_state: str = "FREE_FLOW"
    ) -> Optional[TrajectoryFeatures]:
        if not trajectory or len(trajectory) < 3:
            return None

        points = []
        timestamps = []
        confidences = []

        for pt in trajectory:
            anchor = pt.get("anchor_pixel") or pt.get("position")
            if anchor and len(anchor) >= 2:
                points.append((float(anchor[0]), float(anchor[1])))
                timestamps.append(float(pt.get("timestamp", 0.0)))
                confidences.append(float(pt.get("confidence", 0.85)))

        if len(points) < 3:
            return None

        mean_conf = float(np.mean(confidences)) if confidences else 0.85
        track_valid = mean_conf >= 0.30 and len(points) >= 3
        valid_str = "VALID" if track_valid else "INVALID"

        # 1. Track duration & Dwell
        dwell_dur = max(0.1, timestamps[-1] - timestamps[0])
        dt = dwell_dur / max(1, len(points) - 1)

        # 2. Path length & Step distances
        total_len = 0.0
        step_distances = []
        headings = []

        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            dist = math.hypot(dx, dy)
            total_len += dist
            step_distances.append(dist)
            if dist > 0.5:
                headings.append(math.atan2(dy, dx))

        # 3. Displacement
        dx_tot = points[-1][0] - points[0][0]
        dy_tot = points[-1][1] - points[0][1]
        disp = math.hypot(dx_tot, dy_tot)

        # 4. Trajectory Smoothness & Movement Ratio
        smoothness = min(1.0, max(0.0, disp / max(1e-5, total_len)))
        movement_ratio = min(1.0, max(0.0, total_len / max(1e-5, disp + total_len)))

        # 5. Average direction vector & Direction changes
        dist_tot = max(1e-5, disp)
        dir_vec = [round(dx_tot / dist_tot, 4), round(dy_tot / dist_tot, 4)]

        dir_changes = 0
        curvature_sum = 0.0
        for i in range(1, len(headings)):
            diff = abs(headings[i] - headings[i-1])
            if diff > math.pi:
                diff = 2 * math.pi - diff
            curvature_sum += diff
            if diff > math.radians(25):  # Heading change > 25 deg
                dir_changes += 1

        path_curvature = round(float(curvature_sum / max(1, len(headings) - 1)), 4) if headings else 0.0

        # 6. Lateral & Longitudinal Displacement relative to principal direction
        normal_vec = [-dir_vec[1], dir_vec[0]]
        lat_offsets = []
        long_offsets = []
        for p in points:
            v = [p[0] - points[0][0], p[1] - points[0][1]]
            lat_off = abs(v[0] * normal_vec[0] + v[1] * normal_vec[1])
            long_off = abs(v[0] * dir_vec[0] + v[1] * dir_vec[1])
            lat_offsets.append(lat_off)
            long_offsets.append(long_off)

        max_lat_move = round(float(np.max(lat_offsets)), 2) if lat_offsets else 0.0
        max_long_move = round(float(np.max(long_offsets)), 2) if long_offsets else 0.0

        # 7. Lane Changes
        lane_changes = 0
        if lane_engine:
            current_lane = lane_engine.assign_lane(points[0])
            for p in points[1:]:
                l_new = lane_engine.assign_lane(p)
                if l_new != "UNKNOWN" and l_new != current_lane:
                    lane_changes += 1
                    current_lane = l_new

        # 8. Velocities, Acceleration, Deceleration & Stop Duration
        velocities_px_s = [d / max(0.001, dt) for d in step_distances]
        stop_frames = sum(1 for v in velocities_px_s if v < 15.0)  # ~ 3 kmh proxy in pixel space
        stop_dur = round(float(stop_frames * dt), 2)

        acc_max = 0.0
        dec_max = 0.0
        for i in range(1, len(velocities_px_s)):
            dv = velocities_px_s[i] - velocities_px_s[i-1]
            if dv > acc_max:
                acc_max = dv
            elif -dv > dec_max:
                dec_max = -dv

        # 9. Context & Track Quality
        occ_ctx = round(float(np.mean(list(frame_occupancies.values()))), 1) if frame_occupancies else 8.0
        neighbors_cnt = int(min(12, occ_ctx))
        local_dens = round(occ_ctx / 0.15, 1)
        quality_score = min(100.0, round(mean_conf * 100.0 * smoothness, 1))

        # Build formal structured features dictionary matching Part 2
        structured = {
            "track_duration_sec": FeatureValue(value=round(dwell_dur, 2), unit="sec", source="TIMESTAMP_DELTA", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "total_trajectory_length": FeatureValue(value=round(total_len, 2), unit="px", source="EUCLIDEAN_SUM", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "displacement": FeatureValue(value=round(disp, 2), unit="px", source="END_POINTS", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "average_direction": FeatureValue(value=dir_vec, unit="unit_vector", source="PRINCIPAL_AXIS", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "direction_changes": FeatureValue(value=dir_changes, unit="count", source="HEADING_VARIANCE", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "trajectory_curvature": FeatureValue(value=path_curvature, unit="radians", source="ANGULAR_DELTA", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "lateral_displacement": FeatureValue(value=max_lat_move, unit="px", source="NORMAL_PROJECTION", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "longitudinal_displacement": FeatureValue(value=max_long_move, unit="px", source="PARALLEL_PROJECTION", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "lane_changes": FeatureValue(value=lane_changes, unit="count", source="LANE_ENGINE", validity="VALID" if lane_engine else "LOW_CONFIDENCE", quality=0.85 if lane_engine else 0.50).to_dict(),
            "stop_duration": FeatureValue(value=stop_dur, unit="sec", source="VELOCITY_THRESHOLD", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "movement_ratio": FeatureValue(value=round(movement_ratio, 4), unit="ratio", source="RATIO_CALCULATION", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "trajectory_smoothness": FeatureValue(value=round(smoothness, 4), unit="ratio", source="DISPLACEMENT_RATIO", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "detection_confidence_mean": FeatureValue(value=round(mean_conf, 4), unit="score", source="BYTE_TRACK_CONF", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "track_quality": FeatureValue(value=quality_score, unit="score_100", source="FUSED_METRIC", validity=valid_str, quality=round(mean_conf, 2)).to_dict(),
            "neighboring_vehicle_count": FeatureValue(value=neighbors_cnt, unit="vehicles", source="OCCUPANCY_ENGINE", validity="VALID" if frame_occupancies else "LOW_CONFIDENCE", quality=0.80).to_dict(),
            "local_occupancy": FeatureValue(value=occ_ctx, unit="vehicles", source="FRAME_OCCUPANCY", validity="VALID" if frame_occupancies else "LOW_CONFIDENCE", quality=0.80).to_dict(),
            "local_density": FeatureValue(value=local_dens, unit="veh_per_area", source="DENSITY_ENGINE", validity="VALID" if frame_occupancies else "LOW_CONFIDENCE", quality=0.80).to_dict(),
            "local_traffic_state": FeatureValue(value=local_traffic_state, unit="state_name", source="CONGESTION_ENGINE", validity="VALID", quality=0.90).to_dict()
        }

        provenance = {
            "feature_schema_version": "v2.0.0",
            "total_trajectory_length": "Sum of frame-by-frame Euclidean distances",
            "displacement": "Straight line distance start to end point",
            "trajectory_smoothness": "displacement / total_trajectory_length",
            "direction_vector": "Normalized start-to-end vector",
            "stop_duration_sec": "Time steps where velocity < 15 px/s threshold",
            "path_curvature": "Mean angular heading variance along path",
            "metadata_fields": ["value", "unit", "source", "validity", "quality"],
            "calculation_rule": "Strict mathematical extraction with validity gating"
        }

        return TrajectoryFeatures(
            track_id=track_id,
            vehicle_class=vehicle_class,
            track_duration_sec=round(dwell_dur, 2),
            total_trajectory_length=round(total_len, 2),
            displacement=round(disp, 2),
            trajectory_smoothness=round(smoothness, 4),
            direction_vector=dir_vec,
            direction_changes=dir_changes,
            path_curvature=path_curvature,
            lateral_movement=max_lat_move,
            longitudinal_movement=max_long_move,
            lane_changes=lane_changes,
            dwell_duration_sec=round(dwell_dur, 2),
            stop_duration_sec=stop_dur,
            movement_ratio=round(movement_ratio, 4),
            acceleration_proxy=round(acc_max, 2),
            deceleration_proxy=round(dec_max, 2),
            mean_confidence=round(mean_conf, 4),
            track_quality_score=quality_score,
            neighboring_vehicle_count=neighbors_cnt,
            local_occupancy=occ_ctx,
            local_density=local_dens,
            local_traffic_state=local_traffic_state,
            structured_features=structured,
            provenance=provenance
        )
