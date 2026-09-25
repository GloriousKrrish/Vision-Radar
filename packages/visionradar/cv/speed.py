import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint
from visionradar.cv.calibration import HomographyCalibrator

class SpeedEstimate:
    """
    Encapsulates estimated vehicle speed, uncertainty bounds, speed method provenance, and error decomposition.
    """
    def __init__(
        self,
        speed_method: str,
        distance_m: float,
        elapsed_time_s: float,
        speed_mps: float,
        speed_kmh: float,
        uncertainty_kmh: float,
        validity: str,
        sample_count: int,
        instantaneous_kmh: float,
        smoothed_kmh: float,
        confidence_low_kmh: float,
        confidence_high_kmh: float,
        error_components: Dict[str, float]
    ):
        self.speed_method = speed_method
        self.distance_m = distance_m
        self.elapsed_time_s = elapsed_time_s
        self.speed_mps = speed_mps
        self.speed_kmh = speed_kmh
        self.uncertainty_kmh = uncertainty_kmh
        self.validity = validity
        self.sample_count = sample_count
        self.instantaneous_kmh = instantaneous_kmh
        self.smoothed_kmh = smoothed_kmh
        self.confidence_low_kmh = confidence_low_kmh
        self.confidence_high_kmh = confidence_high_kmh
        self.error_components = error_components

    def to_dict(self) -> Dict[str, Any]:
        return {
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

class MonocularSpeedEstimator:
    """
    Timestamp-aware monocular vehicle speed estimator supporting REGRESSION, PATH_AVERAGE, and INSTANTANEOUS methods.
    """

    def __init__(
        self,
        calibrator: HomographyCalibrator,
        window_size_frames: int = 15,
        fps: float = 30.0
    ):
        self.calibrator = calibrator
        self.window_size_frames = window_size_frames
        self.fps = fps

    def project_trajectory(self, trajectory: VehicleTrajectory):
        """
        Projects all trajectory anchor pixel points onto the calibrated road plane.
        """
        for pt in trajectory.points:
            u, v = pt.anchor_pixel
            pt.world_pos = self.calibrator.image_to_world(u, v)

    def estimate_path_average_speed(self, trajectory: VehicleTrajectory) -> Optional[SpeedEstimate]:
        """
        Estimates total path-average speed over entire track lifetime.
        """
        valid_pts = [p for p in trajectory.points if p.world_pos is not None]
        if len(valid_pts) < 2:
            return None

        p_start = valid_pts[0]
        p_end = valid_pts[-1]
        dt = p_end.timestamp - p_start.timestamp
        if dt < 1e-4:
            return None

        # Cumulative path distance
        dist = 0.0
        for i in range(1, len(valid_pts)):
            w1 = valid_pts[i - 1].world_pos
            w2 = valid_pts[i].world_pos
            dist += float(np.hypot(w2[0] - w1[0], w2[1] - w1[1]))

        speed_mps = dist / dt
        speed_kmh = speed_mps * 3.6
        unc = max(1.0, min(12.0, (self.calibrator.compute_reprojection_rmse() / dt) * 3.6))

        return SpeedEstimate(
            speed_method="PATH_AVERAGE",
            distance_m=dist,
            elapsed_time_s=dt,
            speed_mps=speed_mps,
            speed_kmh=speed_kmh,
            uncertainty_kmh=unc,
            validity="VALID" if speed_kmh <= 180.0 else "LOW_CONFIDENCE",
            sample_count=len(valid_pts),
            instantaneous_kmh=speed_kmh,
            smoothed_kmh=speed_kmh,
            confidence_low_kmh=max(0.0, speed_kmh - 1.96 * unc),
            confidence_high_kmh=speed_kmh + 1.96 * unc,
            error_components={
                "homography_perspective_pct": 60.0,
                "centroid_jitter_pct": 25.0,
                "timestamp_variance_pct": 15.0
            }
        )

    def estimate_speed_at_frame(
        self,
        trajectory: VehicleTrajectory,
        target_frame_idx: int
    ) -> Optional[SpeedEstimate]:
        """
        Estimates vehicle speed at target_frame_idx using a temporal window around target_frame_idx.
        """
        if not trajectory.points:
            return None

        half_w = self.window_size_frames // 2
        window_pts = [
            p for p in trajectory.points
            if abs(p.frame_index - target_frame_idx) <= half_w and p.world_pos is not None
        ]

        if len(window_pts) < 3:
            return self._calculate_instantaneous_speed(trajectory, target_frame_idx)

        t = np.array([p.timestamp for p in window_pts], dtype=np.float64)
        X = np.array([p.world_pos[0] for p in window_pts], dtype=np.float64)
        Y = np.array([p.world_pos[1] for p in window_pts], dtype=np.float64)

        dt = t[-1] - t[0]
        if dt < 1e-4:
            return None

        poly_x, res_x, _, _, _ = np.polyfit(t, X, 1, full=True)
        poly_y, res_y, _, _, _ = np.polyfit(t, Y, 1, full=True)

        vx = poly_x[0]
        vy = poly_y[0]

        speed_mps = float(np.hypot(vx, vy))
        smoothed_kmh = float(speed_mps * 3.6)
        dist_m = float(np.hypot(X[-1] - X[0], Y[-1] - Y[0]))

        inst_estimate = self._calculate_instantaneous_speed(trajectory, target_frame_idx)
        inst_kmh = inst_estimate.instantaneous_kmh if inst_estimate else smoothed_kmh

        rmse_cal = self.calibrator.compute_reprojection_rmse()
        mse_x = res_x[0] / len(t) if len(res_x) > 0 else 0.05
        mse_y = res_y[0] / len(t) if len(res_y) > 0 else 0.05
        fit_residual_std = float(np.sqrt(mse_x + mse_y))

        cal_uncertainty_kmh = (rmse_cal / max(0.2, dt)) * 3.6
        jitter_uncertainty_kmh = (fit_residual_std / max(0.2, dt)) * 3.6
        time_uncertainty_kmh = 0.5

        total_uncertainty_kmh = float(np.sqrt(
            cal_uncertainty_kmh**2 + jitter_uncertainty_kmh**2 + time_uncertainty_kmh**2
        ))
        total_uncertainty_kmh = max(0.8, min(15.0, total_uncertainty_kmh))

        tot_sq = max(1e-5, cal_uncertainty_kmh**2 + jitter_uncertainty_kmh**2 + time_uncertainty_kmh**2)
        err_components = {
            "homography_perspective_pct": round(float((cal_uncertainty_kmh**2 / tot_sq) * 100), 1),
            "centroid_jitter_pct": round(float((jitter_uncertainty_kmh**2 / tot_sq) * 100), 1),
            "timestamp_variance_pct": round(float((time_uncertainty_kmh**2 / tot_sq) * 100), 1)
        }

        ci_low = max(0.0, smoothed_kmh - 1.96 * total_uncertainty_kmh)
        ci_high = smoothed_kmh + 1.96 * total_uncertainty_kmh

        return SpeedEstimate(
            speed_method="REGRESSION",
            distance_m=dist_m,
            elapsed_time_s=dt,
            speed_mps=speed_mps,
            speed_kmh=smoothed_kmh,
            uncertainty_kmh=total_uncertainty_kmh,
            validity="VALID" if smoothed_kmh <= 180.0 else "LOW_CONFIDENCE",
            sample_count=len(window_pts),
            instantaneous_kmh=inst_kmh,
            smoothed_kmh=smoothed_kmh,
            confidence_low_kmh=ci_low,
            confidence_high_kmh=ci_high,
            error_components=err_components
        )

    def _calculate_instantaneous_speed(
        self,
        trajectory: VehicleTrajectory,
        target_frame_idx: int
    ) -> Optional[SpeedEstimate]:
        idx_list = [i for i, p in enumerate(trajectory.points) if p.frame_index == target_frame_idx]
        if not idx_list:
            return None

        curr_i = idx_list[0]
        if curr_i == 0:
            if len(trajectory.points) < 2:
                return None
            p1 = trajectory.points[0]
            p2 = trajectory.points[1]
        else:
            p1 = trajectory.points[curr_i - 1]
            p2 = trajectory.points[curr_i]

        if p1.world_pos is None or p2.world_pos is None:
            return None

        dt = p2.timestamp - p1.timestamp
        if dt < 1e-4:
            return None

        dist = float(np.hypot(p2.world_pos[0] - p1.world_pos[0], p2.world_pos[1] - p1.world_pos[1]))
        speed_mps = float(dist / dt)
        inst_kmh = float(speed_mps * 3.6)

        return SpeedEstimate(
            speed_method="INSTANTANEOUS",
            distance_m=dist,
            elapsed_time_s=dt,
            speed_mps=speed_mps,
            speed_kmh=inst_kmh,
            uncertainty_kmh=3.5,
            validity="VALID" if inst_kmh <= 180.0 else "LOW_CONFIDENCE",
            sample_count=2,
            instantaneous_kmh=inst_kmh,
            smoothed_kmh=inst_kmh,
            confidence_low_kmh=max(0.0, inst_kmh - 3.5),
            confidence_high_kmh=inst_kmh + 3.5,
            error_components={
                "homography_perspective_pct": 50.0,
                "centroid_jitter_pct": 35.0,
                "timestamp_variance_pct": 15.0
            }
        )
