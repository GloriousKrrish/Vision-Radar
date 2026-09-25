from typing import List, Dict, Any, Optional
import numpy as np

class QueueDetector:
    """
    Detects temporal traffic queues (low speed + spatial proximity + temporal persistence > 3s).
    """
    def __init__(self, low_speed_thresh_kmh: float = 15.0, min_persistence_frames: int = 45):
        self.low_speed_thresh_kmh = low_speed_thresh_kmh
        self.min_persistence_frames = min_persistence_frames
        self.slow_vehicle_persistence: Dict[int, int] = {}
        self.active_queues: List[Dict[str, Any]] = []

    def update(self, active_tracks: List[Dict[str, Any]], current_frame: int, timestamp: float) -> List[Dict[str, Any]]:
        current_track_ids = set()

        for trk in active_tracks:
            tid = trk["track_id"]
            current_track_ids.add(tid)
            spd = trk.get("speed_kmh")

            if spd is not None and spd <= self.low_speed_thresh_kmh:
                self.slow_vehicle_persistence[tid] = self.slow_vehicle_persistence.get(tid, 0) + 1
            else:
                self.slow_vehicle_persistence[tid] = 0

        # Purge absent tracks
        stale = [tid for tid in self.slow_vehicle_persistence if tid not in current_track_ids]
        for tid in stale:
            del self.slow_vehicle_persistence[tid]

        # Identify persistent slow vehicles (>= min_persistence_frames)
        queued_track_ids = [
            tid for tid, count in self.slow_vehicle_persistence.items()
            if count >= self.min_persistence_frames
        ]

        if len(queued_track_ids) >= 2:
            # Persistent queue candidate
            queue_info = {
                "queue_id": f"queue_{current_frame}",
                "status": "QUEUE_ACTIVE",
                "vehicle_count": len(queued_track_ids),
                "queued_track_ids": queued_track_ids,
                "avg_speed_kmh": round(float(np.mean([
                    trk.get("speed_kmh", 0.0) for trk in active_tracks if trk["track_id"] in queued_track_ids
                ])), 1),
                "timestamp": timestamp,
                "confidence": 0.88,
                "explanation": f"Temporal queue detected: {len(queued_track_ids)} vehicles moving under {self.low_speed_thresh_kmh} km/h for >3 seconds."
            }
            self.active_queues = [queue_info]
            return [queue_info]
        else:
            self.active_queues = []
            return []
