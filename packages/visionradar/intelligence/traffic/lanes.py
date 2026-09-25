from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class LanePolygon:
    """
    Defines a lane boundary polygon in pixel coordinates.
    """
    def __init__(self, lane_id: str, name: str, polygon_points: List[Tuple[float, float]], speed_limit_kmh: float = 80.0):
        self.lane_id = lane_id
        self.name = name
        self.polygon = np.array(polygon_points, dtype=np.int32)
        self.speed_limit_kmh = speed_limit_kmh

    def contains_point(self, pt: Tuple[float, float]) -> bool:
        """
        Point-in-polygon test using OpenCV.
        """
        res = cv2.pointPolygonTest(self.polygon, (float(pt[0]), float(pt[1])), False)
        return res >= 0

import cv2

class LaneIntelligenceEngine:
    """
    Assigns vehicle tracks to lane polygons, calculates lane volume, average lane speed, and speed distribution.
    """
    def __init__(self, lanes: Optional[List[LanePolygon]] = None):
        self.lanes = lanes or [
            LanePolygon("lane_1", "Lane 1 (Left)", [(0, 110), (280, 110), (280, 450), (0, 450)], speed_limit_kmh=80.0),
            LanePolygon("lane_2", "Lane 2 (Center)", [(280, 110), (520, 110), (520, 450), (280, 450)], speed_limit_kmh=80.0),
            LanePolygon("lane_3", "Lane 3 (Right)", [(520, 110), (800, 110), (800, 450), (520, 450)], speed_limit_kmh=80.0)
        ]

    def assign_lane(self, anchor_pixel: Tuple[float, float]) -> str:
        for lane in self.lanes:
            if lane.contains_point(anchor_pixel):
                return lane.name
        return "UNKNOWN"

    def compute_lane_metrics(self, active_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lane_stats = {lane.name: {"count": 0, "speeds": [], "limit": lane.speed_limit_kmh} for lane in self.lanes}
        lane_stats["UNKNOWN"] = {"count": 0, "speeds": [], "limit": 80.0}

        for trk in active_tracks:
            lane_name = trk.get("lane") or "UNKNOWN"
            if lane_name not in lane_stats:
                lane_stats[lane_name] = {"count": 0, "speeds": [], "limit": 80.0}

            lane_stats[lane_name]["count"] += 1
            spd = trk.get("speed_kmh")
            if spd is not None and spd > 0:
                lane_stats[lane_name]["speeds"].append(spd)

        result = []
        for lane_name, stats in lane_stats.items():
            speeds = stats["speeds"]
            avg_spd = float(np.mean(speeds)) if speeds else 0.0
            p85_spd = float(np.percentile(speeds, 85)) if speeds else 0.0
            max_spd = float(np.max(speeds)) if speeds else 0.0

            result.append({
                "lane_name": lane_name,
                "vehicle_count": stats["count"],
                "avg_speed_kmh": round(avg_spd, 1),
                "p85_speed_kmh": round(p85_spd, 1),
                "max_speed_kmh": round(max_spd, 1),
                "speed_limit_kmh": stats["limit"]
            })

        return result
