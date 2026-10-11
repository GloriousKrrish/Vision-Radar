"""
VisionRadar Phase 15 — Configuration Snapshot Generator

Captures immutable configuration snapshots for computer vision, traffic intelligence,
and anomaly engines with deterministic SHA256 hashing.
"""

import json
import hashlib
import datetime
from typing import Dict, Any, Optional
from visionradar.research.schemas import ConfigurationSnapshot


class ConfigurationSnapshotEngine:
    """
    Captures complete configuration state of an experiment and computes a deterministic SHA256 hash.
    """

    @staticmethod
    def create_snapshot(
        detector_cfg: Optional[Dict[str, Any]] = None,
        tracker_cfg: Optional[Dict[str, Any]] = None,
        speed_cfg: Optional[Dict[str, Any]] = None,
        traffic_cfg: Optional[Dict[str, Any]] = None,
        anomaly_cfg: Optional[Dict[str, Any]] = None,
        accident_cfg: Optional[Dict[str, Any]] = None
    ) -> ConfigurationSnapshot:

        detector = detector_cfg or {
            "model_name": "YOLOX-Nano",
            "model_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "input_resolution": [640, 640],
            "conf_threshold": 0.30,
            "nms_threshold": 0.45,
            "classes": ["Car", "Bus", "Truck", "Motorcycle"],
            "backend": "ONNXRuntime",
            "device": "CPU"
        }

        tracker = tracker_cfg or {
            "tracker_name": "ByteTrack",
            "track_thresh": 0.50,
            "track_buffer": 30,
            "match_thresh": 0.80,
            "min_box_area": 10.0,
            "max_age": 30
        }

        speed = speed_cfg or {
            "calibration_id": 1,
            "speed_estimator": "MonocularSpeedEstimator",
            "smoothing": "LOWESS",
            "window_size": 5,
            "camera_height_m": 9.0,
            "pitch_deg": 14.0,
            "uncertainty_method": "ErrorPropagationMatrix"
        }

        traffic = traffic_cfg or {
            "occupancy_grid": "8x8_spatial_cells",
            "density_road_length_m": 150.0,
            "queue_stop_velocity_kmh": 3.0,
            "congestion_policy_version": "v1.0.0"
        }

        anomaly = anomaly_cfg or {
            "baseline_method": "STATISTICAL_ROBUST_MAD",
            "sensitivity_threshold": 0.65,
            "mad_multiplier": 3.0,
            "persistence_min_observed_frames": 3,
            "cooldown_frames": 15,
            "proximity_threshold_px": 45.0,
            "speed_delta_threshold_kmh": 30.0,
            "policy_version": "v1.0.0"
        }

        accident = accident_cfg or {
            "classifier_name": "ResNet18_AccidentClassifier",
            "candidate_frame_policy": "RING_BUFFER_3_SHOT",
            "temporal_window_frames": 90,
            "aggregate_score_threshold": 0.65
        }

        raw_str = json.dumps({
            "detector": detector,
            "tracker": tracker,
            "speed": speed,
            "traffic": traffic,
            "anomaly": anomaly,
            "accident": accident
        }, sort_keys=True)

        config_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

        return ConfigurationSnapshot(
            config_hash=config_hash,
            detector=detector,
            tracker=tracker,
            speed=speed,
            traffic_intelligence=traffic,
            anomaly_intelligence=anomaly,
            accident_intelligence=accident,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
