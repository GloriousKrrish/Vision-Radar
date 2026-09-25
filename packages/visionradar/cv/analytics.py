import numpy as np
from typing import List, Dict, Any

class TrafficAnalyticsEngine:
    """
    Computes traffic flow, speed distribution, median/P85 speeds, and class breakdown metrics.
    """

    @staticmethod
    def compute_summary(speed_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not speed_records:
            return {
                "total_vehicles": 0,
                "mean_speed_kmh": 0.0,
                "median_speed_kmh": 0.0,
                "p85_speed_kmh": 0.0,
                "max_speed_kmh": 0.0,
                "class_distribution": {},
                "speed_histogram": {"bins": [], "counts": []}
            }

        speeds = [r["smoothed_kmh"] for r in speed_records if r.get("smoothed_kmh") is not None and r["smoothed_kmh"] > 0]
        classes = [r.get("vehicle_class", "Car") for r in speed_records]

        if not speeds:
            arr = np.array([0.0], dtype=np.float64)
            mean_v = 0.0
            median_v = 0.0
            p85_v = 0.0
            max_v = 0.0
        else:
            arr = np.array(speeds, dtype=np.float64)
            mean_v = float(np.mean(arr))
            median_v = float(np.median(arr))
            p85_v = float(np.percentile(arr, 85))
            max_v = float(np.max(arr))

        # Class counts
        class_counts = {}
        for c in classes:
            class_counts[c] = class_counts.get(c, 0) + 1

        # Speed histogram (10 km/h bins from 0 to 140)
        hist, bin_edges = np.histogram(arr, bins=range(0, 150, 10))
        bins_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(hist))]

        return {
            "total_vehicles": len(speed_records),
            "mean_speed_kmh": round(mean_v, 1),
            "median_speed_kmh": round(median_v, 1),
            "p85_speed_kmh": round(p85_v, 1),
            "max_speed_kmh": round(max_v, 1),
            "class_distribution": class_counts,
            "speed_histogram": {
                "bins": bins_labels,
                "counts": hist.tolist()
            }
        }
