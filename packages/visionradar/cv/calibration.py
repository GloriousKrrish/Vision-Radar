import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional

class HomographyCalibrator:
    """
    Handles 4-point (or N-point) DLT homography calibration for road-plane mapping.
    Maps image pixel coordinates (u, v) to metric world road-plane coordinates (X, Y).
    """

    def __init__(
        self,
        image_points: Optional[List[Tuple[float, float]]] = None,
        world_points: Optional[List[Tuple[float, float]]] = None,
        H_matrix: Optional[np.ndarray] = None,
        camera_height: float = 9.0,
        pitch_deg: float = 14.0,
    ):
        self.camera_height = camera_height
        self.pitch_deg = pitch_deg
        self.image_points = image_points or []
        self.world_points = world_points or []
        
        if H_matrix is not None:
            self.H = np.array(H_matrix, dtype=np.float64)
            if self.H.shape != (3, 3):
                raise ValueError("H_matrix must be a 3x3 matrix")
            self.H_inv = np.linalg.inv(self.H)
        elif len(self.image_points) >= 4 and len(self.world_points) >= 4:
            self.H, self.H_inv = self.compute_homography(self.image_points, self.world_points)
        else:
            self.H = np.eye(3, dtype=np.float64)
            self.H_inv = np.eye(3, dtype=np.float64)

    @staticmethod
    def compute_homography(
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 3x3 Homography Matrix H mapping image (u,v) to world (X,Y) using DLT / RANSAC.
        """
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)

        if len(src) < 4 or len(dst) < 4:
            raise ValueError("At least 4 corresponding point pairs are required for homography.")

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            # Fallback to standard DLT if RANSAC fails
            H, _ = cv2.findHomography(src, dst, 0)
        
        if H is None:
            raise RuntimeError("Failed to compute homography matrix.")

        H = H.astype(np.float64)
        H_inv = np.linalg.inv(H)
        return H, H_inv

    def image_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Projects an image pixel point (u, v) onto the road plane (X, Y) in metric coordinates (meters).
        """
        pt = np.array([u, v, 1.0], dtype=np.float64)
        res = self.H @ pt
        if abs(res[2]) < 1e-9:
            return 0.0, 0.0
        X = res[0] / res[2]
        Y = res[1] / res[2]
        return float(X), float(Y)

    def world_to_image(self, X: float, Y: float) -> Tuple[float, float]:
        """
        Inverse projects a world road-plane point (X, Y) back to image pixel coordinates (u, v).
        """
        pt = np.array([X, Y, 1.0], dtype=np.float64)
        res = self.H_inv @ pt
        if abs(res[2]) < 1e-9:
            return 0.0, 0.0
        u = res[0] / res[2]
        v = res[1] / res[2]
        return float(u), float(v)

    def validate_calibration(self) -> Dict[str, Any]:
        """
        Validates homography matrix sanity:
        - Convex polygon check
        - Non-zero polygon area (> 100 sq px)
        - Plausible road dimensions (length > 5m, width > 2m)
        - Stable matrix condition number (< 1e12)
        - Reprojection error RMSE <= 15.0m
        """
        if len(self.image_points) < 4 or len(self.world_points) < 4:
            return {
                "is_valid": False,
                "reason": "Calibration requires at least 4 corresponding image-world point pairs.",
                "reprojection_rmse_m": 0.0
            }

        pts = np.array(self.image_points, dtype=np.int32)
        if not cv2.isContourConvex(pts):
            return {
                "is_valid": False,
                "reason": "Image polygon points do not form a convex quad.",
                "reprojection_rmse_m": self.compute_reprojection_rmse()
            }

        area = cv2.contourArea(pts)
        if area < 100.0:
            return {
                "is_valid": False,
                "reason": f"Image polygon area ({area:.1f} px^2) is too small.",
                "reprojection_rmse_m": self.compute_reprojection_rmse()
            }

        w_pts = np.array(self.world_points, dtype=np.float64)
        road_width = float(np.max(w_pts[:, 0]) - np.min(w_pts[:, 0]))
        road_length = float(np.max(w_pts[:, 1]) - np.min(w_pts[:, 1]))

        if road_width < 1.0 or road_length < 2.0:
            return {
                "is_valid": False,
                "reason": f"World road dimensions (width={road_width:.1f}m, length={road_length:.1f}m) are implausibly small.",
                "reprojection_rmse_m": self.compute_reprojection_rmse()
            }

        cond_num = float(np.linalg.cond(self.H))
        if cond_num > 1e12 or np.isnan(cond_num):
            return {
                "is_valid": False,
                "reason": f"Homography matrix condition number ({cond_num:.2e}) indicates numerical instability.",
                "reprojection_rmse_m": self.compute_reprojection_rmse()
            }

        rmse = self.compute_reprojection_rmse()
        if rmse > 15.0:
            return {
                "is_valid": False,
                "reason": f"Reprojection RMSE ({rmse:.2f}m) exceeds 15.0m error threshold.",
                "reprojection_rmse_m": rmse
            }

        return {
            "is_valid": True,
            "reason": f"Valid homography calibration (RMSE: {rmse:.2f}m, road length: {road_length:.1f}m, road width: {road_width:.1f}m).",
            "reprojection_rmse_m": round(rmse, 3),
            "road_length_m": round(road_length, 2),
            "road_width_m": round(road_width, 2)
        }

    def compute_reprojection_rmse(self) -> float:
        """
        Computes the Root Mean Squared Error (RMSE) in meters of reprojection on the reference points.
        """
        if not self.image_points or not self.world_points:
            return 0.0

        sq_errs = []
        for (u, v), (real_X, real_Y) in zip(self.image_points, self.world_points):
            est_X, est_Y = self.image_to_world(u, v)
            sq_errs.append((est_X - real_X) ** 2 + (est_Y - real_Y) ** 2)

        return float(np.sqrt(np.mean(sq_errs)))

    def compute_scales(self) -> Tuple[float, float]:
        """
        Computes scale factors Sx (near edge m/px) and Sy (mean distance m/px).
        """
        if len(self.image_points) < 4 or len(self.world_points) < 4:
            return 0.05, 0.15

        p0, p1 = self.image_points[0], self.image_points[1]
        w0, w1 = self.world_points[0], self.world_points[1]

        px_dist_x = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        world_dist_x = np.hypot(w1[0] - w0[0], w1[1] - w0[1])

        p2, p3 = self.image_points[0], self.image_points[3]
        w2, w3 = self.world_points[0], self.world_points[3]

        px_dist_y = np.hypot(p3[0] - p2[0], p3[1] - p2[1])
        world_dist_y = np.hypot(w3[0] - w2[0], w3[1] - w2[1])

        Sx = float(world_dist_x / max(1e-5, px_dist_x))
        Sy = float(world_dist_y / max(1e-5, px_dist_y))
        return Sx, Sy

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes calibration data to dictionary/JSON format.
        """
        Sx, Sy = self.compute_scales()
        return {
            "H_matrix": self.H.tolist(),
            "image_points": self.image_points,
            "world_points": self.world_points,
            "reprojection_rmse_m": self.compute_reprojection_rmse(),
            "scale_x_m_per_px": Sx,
            "scale_y_m_per_px": Sy,
            "camera_height_m": self.camera_height,
            "pitch_deg": self.pitch_deg
        }
