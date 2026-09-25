import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple

class VehicleDensityEngine:
    """
    Computes physical vehicle density (vehicles/km) and instantaneous road occupancy.
    Strictly distinguishes total unique vehicles observed throughout video from simultaneous road segment occupancy.
    """
    def __init__(
        self,
        road_length_m: Optional[float] = 150.0,
        image_points: Optional[List[Tuple[float, float]]] = None
    ):
        self.road_length_m = road_length_m or 150.0
        self.image_points = image_points
        self.occupancy_history: List[int] = []
        self.density_history: List[float] = []
        self.unique_track_ids: set = set()

    def is_inside_segment(
        self,
        bbox: Optional[List[float]],
        anchor_pixel: Optional[Tuple[float, float]],
        world_pos: Optional[Tuple[float, float]] = None
    ) -> bool:
        """
        Determines whether a vehicle track is physically inside the calibrated measurement segment.
        """
        if world_pos is not None and len(world_pos) >= 2 and world_pos[1] is not None:
            if 0.0 <= world_pos[1] <= self.road_length_m:
                return True

        if anchor_pixel is not None and self.image_points and len(self.image_points) >= 4:
            pts = np.array(self.image_points, dtype=np.int32)
            u, v = anchor_pixel
            res = cv2.pointPolygonTest(pts, (float(u), float(v)), False)
            return res >= 0

        if bbox is not None and self.image_points and len(self.image_points) >= 4:
            bx, by, bw, bh = bbox
            u, v = bx + bw / 2.0, by + bh
            pts = np.array(self.image_points, dtype=np.int32)
            res = cv2.pointPolygonTest(pts, (float(u), float(v)), False)
            return res >= 0

        return True

    def update_frame_occupancy(
        self,
        frame_idx: int,
        timestamp: float,
        active_tracks: List[Dict[str, Any]],
        calibrator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluates instantaneous occupancy at frame timestamp t.
        """
        current_occupancy = 0
        road_length_km = max(0.001, self.road_length_m / 1000.0)

        for trk in active_tracks:
            tid = trk.get("track_id")
            if tid is not None:
                self.unique_track_ids.add(tid)

            bbox = trk.get("bbox")
            anchor = trk.get("anchor_pixel")
            world_pos = trk.get("world_pos")

            if self.is_inside_segment(bbox, anchor, world_pos):
                current_occupancy += 1

        instantaneous_density = current_occupancy / road_length_km
        self.occupancy_history.append(current_occupancy)
        self.density_history.append(instantaneous_density)

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "current_road_occupancy": current_occupancy,
            "instantaneous_density_veh_km": round(instantaneous_density, 1)
        }

    def compute_density(
        self,
        total_unique_tracks: Optional[int] = None,
        calibration_active: bool = True,
        vehicle_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes aggregate density and occupancy statistics across all evaluated frames.
        """
        obs_vehicles = vehicle_count if vehicle_count is not None else (total_unique_tracks if total_unique_tracks is not None else len(self.unique_track_ids))

        if not calibration_active or not self.road_length_m or self.road_length_m <= 0:
            return {
                "density_status": "NOT_AVAILABLE",
                "total_unique_vehicles_observed": obs_vehicles,
                "current_road_occupancy": self.occupancy_history[-1] if self.occupancy_history else 0,
                "mean_road_occupancy": None,
                "median_road_occupancy": None,
                "p85_road_occupancy": None,
                "max_road_occupancy": None,
                "mean_density_veh_km": None,
                "median_density_veh_km": None,
                "p85_density_veh_km": None,
                "max_density_veh_km": None,
                "road_length_m": self.road_length_m,
                "explanation": "Road geometry scale or homography calibration is not active."
            }

        if not self.occupancy_history:
            road_length_km = max(0.001, self.road_length_m / 1000.0)
            density_veh_km = obs_vehicles / road_length_km
            return {
                "density_status": "VALID",
                "density_veh_km": round(density_veh_km, 1),
                "total_unique_vehicles_observed": obs_vehicles,
                "current_road_occupancy": obs_vehicles,
                "mean_road_occupancy": float(obs_vehicles),
                "median_road_occupancy": float(obs_vehicles),
                "p85_road_occupancy": float(obs_vehicles),
                "max_road_occupancy": obs_vehicles,
                "mean_density_veh_km": round(density_veh_km, 1),
                "median_density_veh_km": round(density_veh_km, 1),
                "p85_density_veh_km": round(density_veh_km, 1),
                "max_density_veh_km": round(density_veh_km, 1),
                "road_length_m": self.road_length_m,
                "explanation": f"Snapshot count used for road occupancy over {self.road_length_m}m segment."
            }

        occ_arr = np.array(self.occupancy_history, dtype=np.float64)
        dens_arr = np.array(self.density_history, dtype=np.float64)

        mean_occ = float(np.mean(occ_arr))
        median_occ = float(np.median(occ_arr))
        p85_occ = float(np.percentile(occ_arr, 85))
        max_occ = int(np.max(occ_arr))

        mean_dens = float(np.mean(dens_arr))
        median_dens = float(np.median(dens_arr))
        p85_dens = float(np.percentile(dens_arr, 85))
        max_dens = float(np.max(dens_arr))

        return {
            "density_status": "VALID",
            "density_veh_km": round(mean_dens, 1),
            "total_unique_vehicles_observed": obs_vehicles,
            "current_road_occupancy": self.occupancy_history[-1],
            "mean_road_occupancy": round(mean_occ, 2),
            "median_road_occupancy": round(median_occ, 2),
            "p85_road_occupancy": round(p85_occ, 2),
            "max_road_occupancy": max_occ,
            "mean_density_veh_km": round(mean_dens, 1),
            "median_density_veh_km": round(median_dens, 1),
            "p85_density_veh_km": round(p85_dens, 1),
            "max_density_veh_km": round(max_dens, 1),
            "road_length_m": self.road_length_m,
            "explanation": f"Mean instantaneous occupancy is {mean_occ:.1f} vehicles ({mean_dens:.1f} veh/km) over {self.road_length_m}m segment. Total observed unique vehicles: {obs_vehicles}."
        }

