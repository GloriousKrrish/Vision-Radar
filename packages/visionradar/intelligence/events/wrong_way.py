import numpy as np
from typing import List, Dict, Any, Tuple
from visionradar.intelligence.events.base import BaseEventDetector

class WrongWayDetector(BaseEventDetector):
    """
    Detects vehicles traveling opposite to designated roadway flow over temporal persistence and displacement.
    """
    def __init__(
        self,
        expected_direction: str = "SOUTHWARD",
        expected_vector: Tuple[float, float] = (0.0, 1.0),
        min_persistence_frames: int = 15,
        min_displacement_px: float = 30.0
    ):
        self.expected_direction = expected_direction
        self.expected_vector = np.array(expected_vector, dtype=np.float64)
        if np.linalg.norm(self.expected_vector) > 1e-5:
            self.expected_vector = self.expected_vector / np.linalg.norm(self.expected_vector)
        self.min_persistence_frames = min_persistence_frames
        self.min_displacement_px = min_displacement_px
        self.track_history: Dict[int, List[Dict[str, Any]]] = {}

    def detect_frame_triggers(self, active_tracks: List[Dict[str, Any]], frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        triggers = []

        for trk in active_tracks:
            tid = trk["track_id"]
            anchor = trk.get("anchor_pixel")
            if not anchor:
                continue

            if tid not in self.track_history:
                self.track_history[tid] = []

            self.track_history[tid].append({
                "frame": frame_index,
                "timestamp": timestamp,
                "anchor": anchor
            })

            pts = self.track_history[tid]
            if len(pts) < self.min_persistence_frames:
                continue

            # Calculate total displacement vector over window
            p_start = np.array(pts[0]["anchor"], dtype=np.float64)
            p_end = np.array(pts[-1]["anchor"], dtype=np.float64)
            disp_vec = p_end - p_start
            disp_mag = float(np.linalg.norm(disp_vec))

            if disp_mag < self.min_displacement_px:
                continue

            unit_dir = disp_vec / disp_mag
            dot_prod = float(np.dot(unit_dir, self.expected_vector))

            # Dot product < -0.5 indicates motion > 120 degrees opposite to expected flow direction
            if dot_prod < -0.5:
                direction_label = "NORTHWARD" if self.expected_direction == "SOUTHWARD" else "OPPOSITE"
                dir_confidence = min(0.99, float(abs(dot_prod)))

                triggers.append({
                    "event_type": "wrong_way",
                    "track_id": tid,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "metric_value": disp_mag,
                    "confidence": round(dir_confidence, 2),
                    "severity": "CRITICAL",
                    "explanation": f"Track #{tid} ({trk.get('vehicle_class')}) moving {direction_label}, contrary to expected roadway flow {self.expected_direction} (displacement: {disp_mag:.1f}px over {len(pts)} frames).",
                    "direction_vector": [round(float(unit_dir[0]), 3), round(float(unit_dir[1]), 3)],
                    "expected_lane_direction": self.expected_direction,
                    "observed_direction": direction_label,
                    "direction_confidence": round(dir_confidence, 2),
                    "evidence": {
                        "track_id": tid,
                        "observed_direction": direction_label,
                        "expected_direction": self.expected_direction,
                        "displacement_px": round(disp_mag, 1),
                        "bbox": trk.get("bbox"),
                        "speed_kmh": trk.get("speed_kmh")
                    }
                })

        return triggers

    def detect_events(self, active_tracks: List[Dict[str, Any]], frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        return self.detect_frame_triggers(active_tracks, frame_index, timestamp)

