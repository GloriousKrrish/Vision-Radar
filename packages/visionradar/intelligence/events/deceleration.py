from typing import List, Dict, Any
from visionradar.intelligence.events.base import BaseEventDetector

class SuddenDecelerationDetector(BaseEventDetector):
    """
    Evaluates sudden deceleration triggers (speed drop >= 25.0 km/h over short interval).
    """
    def __init__(self, drop_threshold_kmh: float = 25.0):
        self.drop_threshold_kmh = drop_threshold_kmh
        self.prev_speeds: Dict[int, float] = {}

    def detect_frame_triggers(self, active_tracks: List[Dict[str, Any]], frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        triggers = []

        for trk in active_tracks:
            tid = trk["track_id"]
            curr_spd = trk.get("speed_kmh")

            if curr_spd is not None and tid in self.prev_speeds:
                prev_spd = self.prev_speeds[tid]
                drop = prev_spd - curr_spd
                if drop >= self.drop_threshold_kmh:
                    triggers.append({
                        "event_type": "sudden_deceleration",
                        "track_id": tid,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "metric_value": round(drop, 1),
                        "confidence": 0.88,
                        "severity": "HIGH" if drop > 35.0 else "MEDIUM",
                        "explanation": f"Track #{tid} ({trk.get('vehicle_class')}) decelerated by {drop:.1f} km/h (from {prev_spd:.1f} to {curr_spd:.1f} km/h).",
                        "evidence": {
                            "track_id": tid,
                            "previous_speed_kmh": round(prev_spd, 1),
                            "current_speed_kmh": round(curr_spd, 1),
                            "deceleration_kmh": round(drop, 1),
                            "bbox": trk.get("bbox")
                        }
                    })

            if curr_spd is not None:
                self.prev_speeds[tid] = curr_spd

        return triggers

    def detect_events(self, active_tracks: List[Dict[str, Any]], frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        # Backward compatibility helper
        return self.detect_frame_triggers(active_tracks, frame_index, timestamp)

