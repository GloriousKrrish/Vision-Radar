from typing import List, Dict, Any
from visionradar.intelligence.events.base import BaseEventDetector

class StoppedVehicleDetector(BaseEventDetector):
    """
    Detects vehicles remaining stationary (speed <= 3 km/h) in active roadway.
    """
    def __init__(self, stopped_speed_thresh: float = 3.0, min_stopped_frames: int = 30):
        self.stopped_speed_thresh = stopped_speed_thresh
        self.min_stopped_frames = min_stopped_frames
        self.stopped_tracker: Dict[int, int] = {}

    def detect_frame_triggers(self, active_tracks: List[Dict[str, Any]], frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        triggers = []

        for trk in active_tracks:
            tid = trk["track_id"]
            spd = trk.get("speed_kmh")

            if spd is not None and spd <= self.stopped_speed_thresh:
                self.stopped_tracker[tid] = self.stopped_tracker.get(tid, 0) + 1
            else:
                self.stopped_tracker[tid] = 0

            if self.stopped_tracker[tid] >= self.min_stopped_frames:
                triggers.append({
                    "event_type": "stopped_vehicle",
                    "track_id": tid,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "metric_value": float(self.stopped_tracker[tid]),
                    "confidence": 0.89,
                    "severity": "MEDIUM",
                    "explanation": f"Track #{tid} ({trk.get('vehicle_class')}) stationary in lane '{trk.get('lane')}' for {self.stopped_tracker[tid]} frames.",
                    "evidence": {
                        "track_id": tid,
                        "lane": trk.get("lane"),
                        "speed_kmh": spd,
                        "bbox": trk.get("bbox")
                    }
                })

        return triggers

    def detect_events(self, active_tracks: List[Dict[str, Any]], frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        return self.detect_frame_triggers(active_tracks, frame_index, timestamp)

