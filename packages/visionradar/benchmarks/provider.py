import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from visionradar.benchmarks.schema import GroundTruthObservation

class GroundTruthProvider(ABC):
    """
    Abstract interface for vehicle ground-truth trajectory and speed data providers.
    """
    
    @abstractmethod
    def load_ground_truth(self) -> List[GroundTruthObservation]:
        pass

    @abstractmethod
    def get_vehicle_speed(self, vehicle_id: str, timestamp: float) -> Optional[float]:
        pass

    @abstractmethod
    def get_vehicle_trajectory(self, vehicle_id: str) -> List[GroundTruthObservation]:
        pass

    @abstractmethod
    def validate_annotations(self) -> bool:
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        pass


class SyntheticGroundTruthProvider(GroundTruthProvider):
    """
    Controlled synthetic traffic benchmark provider with mathematically exact trajectory and ground-truth speed.
    Ground-truth vehicle configurations:
    - Vehicle #1 (Lane 1): 60.0 km/h (16.67 m/s)
    - Vehicle #2 (Lane 2): 80.0 km/h (22.22 m/s)
    - Vehicle #3 (Lane 3): 105.0 km/h (29.17 m/s)
    - Vehicle #4 (Lane 1): 45.0 km/h (12.50 m/s)
    - Vehicle #5 (Lane 2): 90.0 km/h (25.00 m/s)
    """

    def __init__(
        self,
        fps: float = 29.97,
        num_frames: int = 300,
        road_length_m: float = 150.0,
        source_name: str = "SYNTHETIC_HIGHWAY_CONTROLLED_V1"
    ):
        self.fps = fps
        self.num_frames = num_frames
        self.duration_sec = num_frames / fps
        self.road_length_m = road_length_m
        self.source_name = source_name
        self.observations: List[GroundTruthObservation] = []
        self.vehicles_config = [
            {"id": "gt_veh_1", "lane": "Lane 1", "lane_x": 2.0, "speed_kmh": 60.0, "start_y": 140.0},
            {"id": "gt_veh_2", "lane": "Lane 2", "lane_x": 6.0, "speed_kmh": 80.0, "start_y": 145.0},
            {"id": "gt_veh_3", "lane": "Lane 3", "lane_x": 10.0, "speed_kmh": 105.0, "start_y": 150.0},
            {"id": "gt_veh_4", "lane": "Lane 1", "lane_x": 2.0, "speed_kmh": 45.0, "start_y": 120.0},
            {"id": "gt_veh_5", "lane": "Lane 2", "lane_x": 6.0, "speed_kmh": 90.0, "start_y": 135.0},
        ]
        self._generate_data()

    def _generate_data(self):
        self.observations = []
        for f in range(self.num_frames):
            t = f / self.fps
            for v in self.vehicles_config:
                v_ms = v["speed_kmh"] / 3.6
                y_pos = v["start_y"] - v_ms * t
                if -10.0 <= y_pos <= 160.0:
                    obs = GroundTruthObservation(
                        ground_truth_id=f"{v['id']}_f{f}",
                        timestamp=round(t, 4),
                        vehicle_id=v["id"],
                        speed_mps=round(v_ms, 4),
                        speed_kmh=round(v["speed_kmh"], 2),
                        position_x=v["lane_x"],
                        position_y=round(y_pos, 4),
                        source=self.source_name,
                        confidence=1.0,
                        annotation_timestamp=round(t, 4)
                    )
                    self.observations.append(obs)

    def load_ground_truth(self) -> List[GroundTruthObservation]:
        return self.observations

    def get_vehicle_speed(self, vehicle_id: str, timestamp: float) -> Optional[float]:
        for obs in self.observations:
            if obs.vehicle_id == vehicle_id and abs(obs.timestamp - timestamp) < 0.5:
                return obs.speed_kmh
        # Fallback to vehicle config constant speed
        for v in self.vehicles_config:
            if v["id"] == vehicle_id:
                return v["speed_kmh"]
        return None

    def get_vehicle_trajectory(self, vehicle_id: str) -> List[GroundTruthObservation]:
        return [obs for obs in self.observations if obs.vehicle_id == vehicle_id]

    def validate_annotations(self) -> bool:
        return len(self.observations) > 0

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "Synthetic Highway Controlled Speed Benchmark",
            "source": self.source_name,
            "license": "CC-BY-4.0 (Synthetic Benchmark)",
            "video_ids": ["synthetic_highway.mp4"],
            "ground_truth_format": "Mathematically exact trajectory equations",
            "units": "meters, meters/second, km/h",
            "coordinate_system": "Metric Road-Plane (X: lane meters, Y: distance meters)",
            "annotation_methodology": "Synthetic trajectory generator with constant velocity equations",
            "number_of_vehicles": len(self.vehicles_config),
            "number_of_usable_samples": len(self.observations)
        }


class JSONFileGroundTruthProvider(GroundTruthProvider):
    """
    JSON File Ground Truth Provider for annotated public/local traffic speed datasets.
    """

    def __init__(self, json_filepath: str):
        self.json_filepath = json_filepath
        self.observations: List[GroundTruthObservation] = []
        self.metadata: Dict[str, Any] = {}
        if os.path.exists(json_filepath):
            self._load_from_file()

    def _load_from_file(self):
        with open(self.json_filepath, 'r') as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {
            "dataset_name": os.path.basename(self.json_filepath),
            "source": "JSON File Annotation",
            "license": "Public Domain",
            "units": "km/h"
        })

        obs_list = data.get("observations", [])
        for item in obs_list:
            self.observations.append(GroundTruthObservation(
                ground_truth_id=str(item.get("ground_truth_id", "")),
                timestamp=float(item.get("timestamp", 0.0)),
                vehicle_id=str(item.get("vehicle_id", "")),
                speed_mps=float(item.get("speed_mps", item.get("speed_kmh", 0.0) / 3.6)),
                speed_kmh=float(item.get("speed_kmh", 0.0)),
                position_x=float(item.get("position_x", 0.0)),
                position_y=float(item.get("position_y", 0.0)),
                source=self.metadata.get("source", "JSON Annotation"),
                confidence=float(item.get("confidence", 1.0)),
                annotation_timestamp=float(item.get("annotation_timestamp", 0.0))
            ))

    def load_ground_truth(self) -> List[GroundTruthObservation]:
        return self.observations

    def get_vehicle_speed(self, vehicle_id: str, timestamp: float) -> Optional[float]:
        matches = [obs for obs in self.observations if obs.vehicle_id == vehicle_id and abs(obs.timestamp - timestamp) < 0.1]
        if matches:
            return matches[0].speed_kmh
        return None

    def get_vehicle_trajectory(self, vehicle_id: str) -> List[GroundTruthObservation]:
        return [obs for obs in self.observations if obs.vehicle_id == vehicle_id]

    def validate_annotations(self) -> bool:
        return len(self.observations) > 0 and all(obs.speed_kmh >= 0 for obs in self.observations)

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata
