from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class VirtualCountingLine:
    """
    Virtual counting line/gate defining a directional crossing segment in pixel coordinates.
    """
    def __init__(self, line_id: str, p1: Tuple[float, float], p2: Tuple[float, float], direction_name: str = "SOUTHWARD", lane_name: str = "Lane 1"):
        self.line_id = line_id
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.direction_name = direction_name
        self.lane_name = lane_name

    def check_crossing(self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float]) -> bool:
        """
        Determines if segment (prev_pos, curr_pos) intersects virtual line segment (p1, p2).
        """
        A = self.p1
        B = self.p2
        C = np.array(prev_pos, dtype=np.float64)
        D = np.array(curr_pos, dtype=np.float64)

        def ccw(pX, pY, pZ):
            return (pZ[1] - pX[1]) * (pY[0] - pX[0]) > (pY[1] - pX[1]) * (pZ[0] - pX[0])

        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


class VehicleCountingEngine:
    """
    Manages virtual counting lines, detects vehicle crossings, and prevents double counting.
    """
    def __init__(self, lines: Optional[List[VirtualCountingLine]] = None):
        self.lines = lines or [
            VirtualCountingLine("gate_1", (50, 260), (750, 260), direction_name="SOUTHWARD", lane_name="Main Roadway")
        ]
        self.counted_tracks: Dict[str, set] = {line.line_id: set() for line in self.lines}
        self.crossing_records: List[Dict[str, Any]] = []

    def process_track_update(self, track_id: int, vehicle_class: str, trajectory_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluates track trajectory against virtual counting lines.
        """
        if len(trajectory_points) < 2:
            return []

        prev_pt = trajectory_points[-2].get("anchor_pixel")
        curr_pt = trajectory_points[-1].get("anchor_pixel")
        timestamp = trajectory_points[-1].get("timestamp", 0.0)

        if not prev_pt or not curr_pt:
            return []

        new_events = []

        for line in self.lines:
            if track_id in self.counted_tracks[line.line_id]:
                continue

            if line.check_crossing(prev_pt, curr_pt):
                self.counted_tracks[line.line_id].add(track_id)
                event_id = f"cnt_{line.line_id}_trk{track_id}_{int(timestamp * 1000)}"
                event = {
                    "count_event_id": event_id,
                    "gate_id": line.line_id,
                    "line_id": line.line_id,
                    "track_id": track_id,
                    "vehicle_class": vehicle_class,
                    "crossing_direction": line.direction_name,
                    "direction": line.direction_name,
                    "lane": line.lane_name,
                    "crossing_timestamp": round(timestamp, 3),
                    "timestamp": round(timestamp, 3),
                    "position": [round(float(curr_pt[0]), 1), round(float(curr_pt[1]), 1)],
                    "already_counted": True
                }
                self.crossing_records.append(event)
                new_events.append(event)

        return new_events

    def get_counts_summary(self) -> Dict[str, Any]:
        total_count = len(self.crossing_records)
        class_counts = {}
        lane_counts = {}

        for rec in self.crossing_records:
            c = rec["vehicle_class"]
            l = rec["lane"]
            class_counts[c] = class_counts.get(c, 0) + 1
            lane_counts[l] = lane_counts.get(l, 0) + 1

        return {
            "total_vehicle_count": total_count,
            "counts_by_class": class_counts,
            "counts_by_lane": lane_counts,
            "records": self.crossing_records
        }
