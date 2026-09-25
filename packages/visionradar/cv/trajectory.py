import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class TrajectoryPoint:
    """
    Represents a single trajectory sample for a tracked vehicle at a given frame.
    """
    def __init__(
        self,
        frame_index: int,
        timestamp: float,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        world_pos: Optional[Tuple[float, float]] = None,
        confidence: float = 1.0,
        anchor_pixel: Optional[Tuple[float, float]] = None
    ):
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.bbox = bbox or (0.0, 0.0, 50.0, 50.0)
        self.confidence = confidence
        self.world_pos = world_pos

        if anchor_pixel is not None:
            self.anchor_pixel = anchor_pixel
        else:
            x1, y1, x2, y2 = self.bbox
            self.anchor_pixel = ((x1 + x2) / 2.0, y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "bbox": list(self.bbox),
            "anchor_pixel": list(self.anchor_pixel),
            "world_pos": list(self.world_pos) if self.world_pos else None,
            "confidence": self.confidence
        }

class VehicleTrajectory:
    """
    Represents the full space-time trajectory of a single tracked vehicle.
    """
    def __init__(self, track_id: int, vehicle_class: str):
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.points: List[TrajectoryPoint] = []

    def add_point(self, point: TrajectoryPoint):
        self.points.append(point)

    @property
    def start_timestamp(self) -> float:
        return self.points[0].timestamp if self.points else 0.0

    @property
    def end_timestamp(self) -> float:
        return self.points[-1].timestamp if self.points else 0.0

    @property
    def duration(self) -> float:
        return self.end_timestamp - self.start_timestamp

    def get_world_trajectory(self) -> List[Tuple[float, float, float]]:
        """
        Returns list of (timestamp, X, Y) for world-projected points.
        """
        return [
            (p.timestamp, p.world_pos[0], p.world_pos[1])
            for p in self.points
            if p.world_pos is not None
        ]

