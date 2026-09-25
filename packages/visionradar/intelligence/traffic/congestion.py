from typing import Dict, Any, List, Optional

class CongestionEngine:
    """
    Evaluates roadway congestion states (FREE_FLOW, MODERATE, CONGESTED, SEVERE) based on average speed,
    road density, and queue length. Provides reproducible policy governance with explicit classification reasons.
    """
    def __init__(self, policy_version: str = "v1.0.0"):
        self.policy_version = policy_version
        self.free_flow_speed_kmh = 50.0
        self.moderate_speed_kmh = 35.0
        self.congested_speed_kmh = 20.0
        self.density_threshold_high = 80.0
        self.queue_threshold_m = 50.0

    def evaluate_congestion(
        self,
        avg_speed_kmh: float,
        mean_density_veh_km: Optional[float] = 0.0,
        mean_occupancy: Optional[float] = 0.0,
        queue_length_m: float = 0.0,
        flow_veh_h: float = 0.0,
        active_queue_count: int = 0
    ) -> Dict[str, Any]:
        density = mean_density_veh_km or 0.0
        occ = mean_occupancy or 0.0

        if avg_speed_kmh == 0.0 and active_queue_count == 0 and flow_veh_h == 0.0:
            state = "NOT_AVAILABLE"
            reason = "NOT_AVAILABLE because vehicle speed, density, and flow data are unavailable or zero."
        elif active_queue_count >= 3 or avg_speed_kmh < 10.0 or queue_length_m >= 100.0:
            state = "SEVERE"
            reason = f"SEVERE because average speed ({avg_speed_kmh:.1f} km/h) < 10.0 km/h threshold OR active queue count ({active_queue_count}) >= 3."
        elif active_queue_count > 0 or avg_speed_kmh < self.congested_speed_kmh or density >= 120.0:
            state = "CONGESTED"
            reason = f"CONGESTED because average speed ({avg_speed_kmh:.1f} km/h) < {self.congested_speed_kmh:.1f} km/h threshold OR density ({density:.1f} veh/km) >= 120.0 veh/km."
        elif avg_speed_kmh < self.free_flow_speed_kmh or density >= self.density_threshold_high:
            state = "MODERATE"
            reason = f"MODERATE because average speed ({avg_speed_kmh:.1f} km/h) < {self.free_flow_speed_kmh:.1f} km/h threshold OR density ({density:.1f} veh/km) >= {self.density_threshold_high:.1f} veh/km."
        else:
            state = "FREE_FLOW"
            reason = f"FREE_FLOW because average speed ({avg_speed_kmh:.1f} km/h) >= {self.free_flow_speed_kmh:.1f} km/h threshold AND density ({density:.1f} veh/km) < {self.density_threshold_high:.1f} veh/km AND queue length ({queue_length_m:.1f} m) < {self.queue_threshold_m:.1f} m."

        return {
            "congestion_state": state,
            "average_speed_kmh": round(avg_speed_kmh, 1),
            "avg_speed_kmh": round(avg_speed_kmh, 1),
            "mean_density_veh_km": round(density, 1),
            "occupancy": round(occ, 2),
            "queue_length_m": round(queue_length_m, 1),
            "flow_veh_h": round(flow_veh_h, 1),
            "active_queue_count": active_queue_count,
            "policy_version": self.policy_version,
            "classification_reason": reason,
            "explanation": reason
        }

