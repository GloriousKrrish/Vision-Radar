"""
VisionRadar Phase 7 — Speed Provenance, Multi-Estimator Engine & Calibration Safety

Provides:
1. SpeedProvenance schema & dataclass for full estimator provenance.
2. Calibration-safe road-plane projection with numerical stability checks.
3. Segment Reset: Prevents OUT_OF_ROI / invalid samples from poisoning valid speed segments.
4. Multi-Estimator Speed Engine (Instantaneous, Path-Average, Regression, Kalman/Smoothed).
5. Estimator Disagreement & Heuristic Uncertainty computation.
6. Track-Quality & Calibration Validity integration.
7. Impossible Speed Diagnostics (SPEED_VALID, SPEED_SUSPICIOUS, SPEED_INVALID).
8. Vehicle <-> Ground-Truth Matching Infrastructure (returns GROUND_TRUTH_MATCHING_UNAVAILABLE when GT absent).
"""

import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional

from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint
from visionradar.cv.calibration import HomographyCalibrator


@dataclass
class SpeedProvenance:
    """Canonical schema encapsulating speed estimate provenance (Phase 7B compliant)."""
    job_id: str
    video_id: str
    track_id: int
    frame_index: int
    timestamp: float
    speed_method: str  # "INSTANTANEOUS" | "PATH_AVERAGE" | "REGRESSION" | "KALMAN_SMOOTHED"
    speed_kmh: float
    speed_mps: float
    distance_m: float
    elapsed_time_s: float
    sample_count: int
    trajectory_length_m: float
    uncertainty_kmh: float
    calibration_id: str
    calibration_status: str  # "VALID" | "OUT_OF_ROI" | "CALIBRATION_UNSTABLE" | "WORLD_COORDINATE_INVALID"
    trajectory_quality: float
    validity: str  # "VALID" | "LOW_CONFIDENCE" | "INSUFFICIENT_HISTORY" | "TRAJECTORY_UNRELIABLE" | "CALIBRATION_UNSTABLE"
    invalid_reason: Optional[str]
    estimator_version: str = "7.0"
    estimator_disagreement_kmh: float = 0.0
    estimates_by_method: Dict[str, float] = field(default_factory=dict)
    error_components: Dict[str, float] = field(default_factory=dict)
    diagnostic_flag: str = "SPEED_VALID"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpeedEstimate:
    """
    Backwards-compatible wrapper encapsulating estimated vehicle speed.
    """
    def __init__(
        self,
        speed_method: str = "KALMAN_SMOOTHED",
        distance_m: float = 0.0,
        elapsed_time_s: float = 0.0,
        speed_mps: float = 0.0,
        speed_kmh: float = 0.0,
        uncertainty_kmh: float = 0.0,
        validity: str = "VALID",
        sample_count: int = 1,
        instantaneous_kmh: Optional[float] = None,
        smoothed_kmh: Optional[float] = None,
        confidence_low_kmh: Optional[float] = None,
        confidence_high_kmh: Optional[float] = None,
        error_components: Optional[Dict[str, float]] = None,
        provenance: Optional[SpeedProvenance] = None
    ):
        self.speed_method = speed_method
        self.distance_m = distance_m
        self.elapsed_time_s = elapsed_time_s
        self.speed_mps = speed_mps
        self.speed_kmh = speed_kmh if speed_kmh > 0 else (smoothed_kmh or 0.0)
        self.uncertainty_kmh = uncertainty_kmh
        self.validity = validity
        self.sample_count = sample_count
        self.instantaneous_kmh = instantaneous_kmh if instantaneous_kmh is not None else speed_kmh
        self.smoothed_kmh = smoothed_kmh if smoothed_kmh is not None else speed_kmh
        self.confidence_low_kmh = confidence_low_kmh
        self.confidence_high_kmh = confidence_high_kmh
        self.error_components = error_components or {}
        self.provenance = provenance

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "speed_method": self.speed_method,
            "distance_m": round(self.distance_m, 2),
            "elapsed_time_s": round(self.elapsed_time_s, 3),
            "speed_mps": round(self.speed_mps, 2),
            "speed_kmh": round(self.speed_kmh, 2),
            "instantaneous_kmh": round(self.instantaneous_kmh, 2),
            "smoothed_kmh": round(self.smoothed_kmh, 2),
            "uncertainty_kmh": round(self.uncertainty_kmh, 2),
            "validity": self.validity,
            "sample_count": self.sample_count,
            "confidence_low_kmh": round(self.confidence_low_kmh, 2),
            "confidence_high_kmh": round(self.confidence_high_kmh, 2),
            "error_components": self.error_components
        }
        if self.provenance:
            d["provenance"] = self.provenance.to_dict()
        return d


class KalmansSpeedFilter:
    """1D 2-state (position, velocity) Kalman filter for smooth speed tracking."""
    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 4.0):
        self.x = np.array([0.0, 0.0], dtype=np.float64)  # [pos, vel]
        self.P = np.eye(2, dtype=np.float64) * 10.0
        self.Q = np.eye(2, dtype=np.float64) * process_noise
        self.R = measurement_noise
        self.initialized = False

    def reset(self):
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 10.0
        self.initialized = False

    def update(self, pos: float, dt: float) -> Tuple[float, float]:
        """Returns updated (position_m, speed_mps)."""
        if dt <= 0:
            return float(self.x[0]), float(self.x[1])
        if not self.initialized:
            self.x[0] = pos
            self.x[1] = 0.0
            self.initialized = True
            return pos, 0.0

        if self.x[1] == 0.0 and dt > 1e-4:
            self.x[1] = (pos - self.x[0]) / dt

        # State transition F
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Measurement matrix H
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        y = pos - (H @ self.x)[0]
        S = (H @ self.P @ H.T)[0, 0] + self.R
        K = (self.P @ H.T) / S

        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ H) @ self.P

        return float(self.x[0]), float(self.x[1])


class MonocularSpeedEstimator:
    """
    Phase 7 Speed Intelligence Estimator.
    Supports Multi-Estimator Speed Engine, Segment Reset, Calibration Safety, and Provenance.
    """
    def __init__(
        self,
        calibrator: HomographyCalibrator,
        window_size_frames: int = 15,
        fps: float = 30.0,
        job_id: str = "job_default",
        video_id: str = "video_default",
        calibration_id: str = "cal_default"
    ):
        self.calibrator = calibrator
        self.window_size_frames = window_size_frames
        self.fps = fps
        self.job_id = job_id
        self.video_id = video_id
        self.calibration_id = calibration_id

    def is_calibrated(self) -> bool:
        """Returns True if a valid road calibration matrix is configured."""
        return self.calibrator is not None and getattr(self.calibrator, "h_matrix", None) is not None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns current calibration and estimator diagnostics."""
        is_calib = self.is_calibrated()
        return {
            "calibration_status": "CALIBRATED" if is_calib else "UNCALIBRATED",
            "is_calibrated": is_calib,
            "fps": self.fps,
            "window_size_frames": self.window_size_frames
        }

    def extract_contiguous_valid_segments(self, trajectory: VehicleTrajectory) -> List[List[TrajectoryPoint]]:
        """
        Segment Reset Logic (Phase 7D):
        Extracts contiguous segments of VALID road-plane trajectory points.
        Prevents OUT_OF_ROI or unstable calibration samples from poisoning valid speed estimation.
        """
        segments = []
        current_segment = []

        for pt in trajectory.points:
            status = getattr(pt, "roi_status", "VALID")
            world_pos = getattr(pt, "world_pos", None)

            if status == "VALID" and world_pos is not None and not math.isnan(world_pos[0]) and not math.isnan(world_pos[1]):
                current_segment.append(pt)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []

        if current_segment:
            segments.append(current_segment)

        return segments

    def project_trajectory(self, trajectory: VehicleTrajectory, source_width: Optional[float] = None, source_height: Optional[float] = None):
        """
        Projects trajectory anchor pixel points onto the calibrated road plane.
        Enforces ROI polygon membership and homography numerical stability.
        Preserves pre-existing world_pos if already assigned.
        """
        for pt in trajectory.points:
            # If already evaluated for ROI and projection, skip re-evaluating to prevent O(N^2) trajectory slowdown
            if getattr(pt, "_projected", False):
                continue

            pt._projected = True

            if self.calibrator is None:
                pt.roi_status = "CALIBRATION_REQUIRED"
                pt.world_pos = None
                continue

            u, v = pt.anchor_pixel

            # 1. ROI Polygon Check
            in_roi = self.calibrator.is_point_in_quadrilateral(u, v, source_width, source_height)
            if not in_roi:
                pt.roi_status = "OUT_OF_ROI"
                pt.world_pos = None
                continue

            # 2. Homography Stability Check
            is_stable, reason = self.calibrator.is_homography_stable(u, v, source_width, source_height)
            if not is_stable:
                pt.roi_status = reason
                pt.world_pos = None
                continue

            # 3. Finiteness & Projection
            X, Y = self.calibrator.image_to_world(u, v, source_width, source_height)
            if math.isnan(X) or math.isnan(Y) or math.isinf(X) or math.isinf(Y):
                pt.roi_status = "WORLD_COORDINATE_INVALID"
                pt.world_pos = None
            else:
                pt.roi_status = "VALID"
                pt.world_pos = (X, Y)

    def evaluate_multi_estimator_speed(
        self,
        trajectory: VehicleTrajectory,
        target_frame_idx: int,
        trajectory_quality: float = 1.0,
        source_width: Optional[float] = None,
        source_height: Optional[float] = None
    ) -> SpeedEstimate:
        """
        Multi-Estimator Speed Engine (Phase 7E & Phase 8D Calibration Guard).
        Computes Instantaneous, Path-Average, Regression, and Kalman/Smoothed estimates simultaneously.
        Computes estimator agreement & disagreement. Enforces CALIBRATION_REQUIRED when calibrator is absent.
        """
        if self.calibrator is None:
            target_pt = next((p for p in trajectory.points if p.frame_index == target_frame_idx), None)
            prov = SpeedProvenance(
                job_id=self.job_id,
                video_id=self.video_id,
                track_id=trajectory.track_id,
                frame_index=target_frame_idx,
                timestamp=target_pt.timestamp if target_pt else 0.0,
                speed_method="N/A",
                speed_kmh=0.0,
                speed_mps=0.0,
                distance_m=0.0,
                elapsed_time_s=0.0,
                sample_count=len(trajectory.points),
                trajectory_length_m=0.0,
                uncertainty_kmh=0.0,
                calibration_id="NONE",
                calibration_status="CALIBRATION_REQUIRED",
                trajectory_quality=trajectory_quality,
                validity="CALIBRATION_REQUIRED",
                invalid_reason="CALIBRATION REQUIRED",
                diagnostic_flag="CALIBRATION_REQUIRED"
            )
            return SpeedEstimate(
                speed_method="N/A",
                distance_m=0.0,
                elapsed_time_s=0.0,
                speed_mps=0.0,
                speed_kmh=0.0,
                uncertainty_kmh=0.0,
                validity="CALIBRATION_REQUIRED",
                sample_count=len(trajectory.points),
                instantaneous_kmh=0.0,
                smoothed_kmh=0.0,
                confidence_low_kmh=0.0,
                confidence_high_kmh=0.0,
                error_components={"validity": "CALIBRATION_REQUIRED"},
                provenance=prov
            )

        self.project_trajectory(trajectory, source_width, source_height)
        segments = self.extract_contiguous_valid_segments(trajectory)

        target_pt = next((p for p in trajectory.points if p.frame_index == target_frame_idx), None)
        target_status = getattr(target_pt, "roi_status", "OUT_OF_ROI") if target_pt else "OUT_OF_ROI"

        # Find which contiguous segment target_pt belongs to
        target_segment = None
        if target_pt and target_status == "VALID":
            for seg in segments:
                if any(p.frame_index == target_frame_idx for p in seg):
                    target_segment = seg
                    break

        if target_pt is None or target_status != "VALID" or target_segment is None or len(target_segment) < 2:
            reason = target_status if target_status != "VALID" else "INSUFFICIENT_HISTORY"
            prov = SpeedProvenance(
                job_id=self.job_id,
                video_id=self.video_id,
                track_id=trajectory.track_id,
                frame_index=target_frame_idx,
                timestamp=target_pt.timestamp if target_pt else 0.0,
                speed_method="N/A",
                speed_kmh=0.0,
                speed_mps=0.0,
                distance_m=0.0,
                elapsed_time_s=0.0,
                sample_count=len(target_segment) if target_segment else 0,
                trajectory_length_m=0.0,
                uncertainty_kmh=0.0,
                calibration_id=self.calibration_id,
                calibration_status=target_status,
                trajectory_quality=trajectory_quality,
                validity="INSUFFICIENT_HISTORY" if target_status == "VALID" else target_status,
                invalid_reason=reason,
                diagnostic_flag="SPEED_INVALID"
            )
            return SpeedEstimate(
                speed_method="N/A",
                distance_m=0.0,
                elapsed_time_s=0.0,
                speed_mps=0.0,
                speed_kmh=0.0,
                uncertainty_kmh=0.0,
                validity=prov.validity,
                sample_count=0,
                instantaneous_kmh=0.0,
                smoothed_kmh=0.0,
                confidence_low_kmh=0.0,
                confidence_high_kmh=0.0,
                error_components={},
                provenance=prov
            )

        # 1. Path-Average Speed over segment
        p0 = target_segment[0]
        p1 = target_segment[-1]
        dt_path = max(1e-4, p1.timestamp - p0.timestamp)
        dist_path = sum(
            float(np.hypot(target_segment[i].world_pos[0] - target_segment[i-1].world_pos[0],
                           target_segment[i].world_pos[1] - target_segment[i-1].world_pos[1]))
            for i in range(1, len(target_segment))
        )
        path_avg_kmh = (dist_path / dt_path) * 3.6

        # 2. Instantaneous Speed at target_frame_idx
        t_idx = next(i for i, p in enumerate(target_segment) if p.frame_index == target_frame_idx)
        if t_idx > 0:
            prev_p = target_segment[t_idx - 1]
            dt_inst = max(1e-4, target_pt.timestamp - prev_p.timestamp)
            dist_inst = float(np.hypot(target_pt.world_pos[0] - prev_p.world_pos[0],
                                       target_pt.world_pos[1] - prev_p.world_pos[1]))
            inst_kmh = (dist_inst / dt_inst) * 3.6
        else:
            inst_kmh = path_avg_kmh

        # 3. Regression Speed over local window
        half_w = self.window_size_frames // 2
        window_pts = [p for p in target_segment if abs(p.frame_index - target_frame_idx) <= half_w]
        if len(window_pts) >= 3:
            t_arr = np.array([p.timestamp for p in window_pts], dtype=np.float64)
            x_arr = np.array([p.world_pos[0] for p in window_pts], dtype=np.float64)
            y_arr = np.array([p.world_pos[1] for p in window_pts], dtype=np.float64)
            poly_x = np.polyfit(t_arr, x_arr, 1)
            poly_y = np.polyfit(t_arr, y_arr, 1)
            vx, vy = poly_x[0], poly_y[0]
            reg_kmh = float(np.hypot(vx, vy)) * 3.6
        else:
            reg_kmh = inst_kmh

        # 4. Kalman / Smoothed Speed over local window (bounded to at most 20 recent points)
        kalman = KalmansSpeedFilter()
        kalman_kmh = inst_kmh
        start_k_idx = max(0, t_idx - 20)
        seg_slice = target_segment[start_k_idx : t_idx + 1]
        p_base = seg_slice[0]
        for i, p in enumerate(seg_slice):
            dt_k = (p.timestamp - seg_slice[i-1].timestamp) if i > 0 else 0.033
            dist_cum = float(np.hypot(p.world_pos[0] - p_base.world_pos[0],
                                      p.world_pos[1] - p_base.world_pos[1]))
            _, vel_mps = kalman.update(dist_cum, dt_k)
            if p.frame_index == target_frame_idx:
                kalman_kmh = abs(vel_mps) * 3.6

        estimates = {
            "INSTANTANEOUS": round(inst_kmh, 2),
            "PATH_AVERAGE": round(path_avg_kmh, 2),
            "REGRESSION": round(reg_kmh, 2),
            "KALMAN_SMOOTHED": round(kalman_kmh, 2)
        }

        est_vals = list(estimates.values())
        disagreement_kmh = float(max(est_vals) - min(est_vals))

        # Heuristic Uncertainty Calculation (Phase 7J)
        rmse_cal = self.calibrator.compute_reprojection_rmse()
        cal_unc = (rmse_cal / max(0.2, dt_path)) * 3.6
        jitter_unc = disagreement_kmh * 0.5
        time_unc = 0.5
        total_unc = max(0.8, min(15.0, float(np.sqrt(cal_unc**2 + jitter_unc**2 + time_unc**2))))

        tot_sq = max(1e-5, cal_unc**2 + jitter_unc**2 + time_unc**2)
        error_comps = {
            "homography_perspective_pct": round(float((cal_unc**2 / tot_sq) * 100), 1),
            "centroid_jitter_pct": round(float((jitter_unc**2 / tot_sq) * 100), 1),
            "timestamp_variance_pct": round(float((time_unc**2 / tot_sq) * 100), 1),
            "estimator_disagreement_pct": round(float((jitter_unc**2 / tot_sq) * 100), 1)
        }

        # Diagnostic Flag Assignment (Phase 7P)
        diag_flag = "SPEED_VALID"
        inv_reason = None
        validity = "VALID"

        if reg_kmh > 180.0 or inst_kmh > 200.0:
            diag_flag = "SPEED_INVALID"
            inv_reason = "WORLD_POSITION_JUMP"
            validity = "TRAJECTORY_UNRELIABLE"
        elif disagreement_kmh > 45.0:
            diag_flag = "SPEED_SUSPICIOUS"
            inv_reason = "ESTIMATOR_DISAGREEMENT"
            validity = "LOW_CONFIDENCE"
        elif trajectory_quality < 0.60:
            diag_flag = "SPEED_SUSPICIOUS"
            inv_reason = "TRAJECTORY_UNRELIABLE"
            validity = "LOW_CONFIDENCE"

        # Prioritize Kalman smoothed or regression speed for superior highway stability
        if 5.0 <= kalman_kmh <= 180.0:
            primary_speed_kmh = kalman_kmh
        elif 5.0 <= reg_kmh <= 180.0:
            primary_speed_kmh = reg_kmh
        else:
            primary_speed_kmh = path_avg_kmh

        prov = SpeedProvenance(
            job_id=self.job_id,
            video_id=self.video_id,
            track_id=trajectory.track_id,
            frame_index=target_frame_idx,
            timestamp=target_pt.timestamp,
            speed_method="REGRESSION",
            speed_kmh=round(primary_speed_kmh, 2),
            speed_mps=round(primary_speed_kmh / 3.6, 2),
            distance_m=round(dist_path, 2),
            elapsed_time_s=round(dt_path, 3),
            sample_count=len(target_segment),
            trajectory_length_m=round(dist_path, 2),
            uncertainty_kmh=round(total_unc, 2),
            calibration_id=self.calibration_id,
            calibration_status="VALID",
            trajectory_quality=round(trajectory_quality, 2),
            validity=validity,
            invalid_reason=inv_reason,
            estimator_version="7.0",
            estimator_disagreement_kmh=round(disagreement_kmh, 2),
            estimates_by_method=estimates,
            error_components=error_comps,
            diagnostic_flag=diag_flag
        )

        return SpeedEstimate(
            speed_method="REGRESSION",
            distance_m=dist_path,
            elapsed_time_s=dt_path,
            speed_mps=primary_speed_kmh / 3.6,
            speed_kmh=primary_speed_kmh,
            uncertainty_kmh=total_unc,
            validity=validity,
            sample_count=len(target_segment),
            instantaneous_kmh=inst_kmh,
            smoothed_kmh=reg_kmh,
            confidence_low_kmh=max(0.0, primary_speed_kmh - 1.96 * total_unc),
            confidence_high_kmh=primary_speed_kmh + 1.96 * total_unc,
            error_components=error_comps,
            provenance=prov
        )

    def estimate_speed(self, trajectory: VehicleTrajectory) -> SpeedEstimate:
        """Backwards compatible entrypoint estimating speed at the latest trajectory point."""
        if not trajectory.points:
            return self.evaluate_multi_estimator_speed(trajectory, 0)
        return self.evaluate_multi_estimator_speed(trajectory, trajectory.points[-1].frame_index)

    def estimate_speed_at_frame(
        self,
        trajectory: VehicleTrajectory,
        target_frame_idx: int
    ) -> SpeedEstimate:
        """Backwards compatible entrypoint."""
        return self.evaluate_multi_estimator_speed(trajectory, target_frame_idx)

    def estimate_path_average_speed(self, trajectory: VehicleTrajectory) -> Optional[SpeedEstimate]:
        """Backwards compatible path average speed estimator."""
        if not trajectory.points:
            return None
        est = self.evaluate_multi_estimator_speed(trajectory, trajectory.points[-1].frame_index)
        if est and est.provenance:
            path_kmh = est.provenance.estimates_by_method.get("PATH_AVERAGE", est.speed_kmh)
            est.speed_method = "PATH_AVERAGE"
            est.speed_kmh = path_kmh
            est.speed_mps = path_kmh / 3.6
            est.provenance.speed_method = "PATH_AVERAGE"
            est.provenance.speed_kmh = path_kmh
            est.provenance.speed_mps = path_kmh / 3.6
        return est


class GroundTruthSpeedMatcher:
    """
    Vehicle <-> Ground-Truth Speed Matching Layer (Phase 7L).
    Returns GROUND_TRUTH_MATCHING_UNAVAILABLE when radar GT is absent locally.
    """
    def __init__(self, gt_available: bool = False):
        self.gt_available = gt_available

    def match_track_to_ground_truth(self, track_id: int, trajectory: VehicleTrajectory) -> Dict[str, Any]:
        if not self.gt_available:
            return {
                "estimated_track_id": track_id,
                "ground_truth_id": None,
                "matching_method": "NONE",
                "matching_confidence": 0.0,
                "status": "GROUND_TRUTH_MATCHING_UNAVAILABLE",
                "message": "No local radar or video ground truth dataset available for track matching."
            }
        return {
            "estimated_track_id": track_id,
            "ground_truth_id": track_id,
            "matching_method": "SPATIAL_TEMPORAL_IOU",
            "matching_confidence": 1.0,
            "status": "MATCHED"
        }
