from typing import List, Dict, Any

class TrafficFlowEngine:
    """
    Computes traffic flow rates (vehicles/hour) over configurable time intervals.
    Supported intervals: 1m (60s), 5m (300s), 15m (900s), 1h (3600s).
    """
    def __init__(self, interval_sec: float = 60.0):
        self.interval_sec = interval_sec

    def compute_flow_rate(self, vehicle_count: int, duration_sec: float) -> Dict[str, Any]:
        valid_duration = max(1.0, duration_sec)
        flow_vph = float(vehicle_count * (3600.0 / valid_duration))

        return {
            "vehicle_count": vehicle_count,
            "duration_sec": round(valid_duration, 2),
            "flow_rate_vph": round(flow_vph, 1),
            "interval_sec": self.interval_sec,
            "provenance": {
                "formula": "vehicles * (3600.0 / duration_sec)",
                "interval_name": f"{int(self.interval_sec // 60)}m" if self.interval_sec >= 60 else f"{int(self.interval_sec)}s"
            }
        }
