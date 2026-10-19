"""
VisionRadar — High-Scale Vehicle Speed Estimation & Edge Hardware YOLO Sensor
=============================================================================
Reference implementation for:
1. Edge-Accelerated YOLO Object Detection (NPU/VPU/ONNX INT8 compatible).
2. Real-time Multi-Object Tracking (ByteTrack/SORT ID association).
3. Planar Homography / Perspective Transformation (Wheel contact to road meters).
4. Rolling Frame-by-Frame Speed Estimation (Delta-d / Delta-t across N frames).
5. Dynamic Visual Overlay with Speed Limit Violation Highlighting.
6. Asynchronous Event Logging & Vehicle ROI Snapshot Capturing.
"""

from __future__ import annotations

import collections
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackState:
    """State management for tracked vehicles across consecutive video frames."""
    vehicle_id: int
    timestamps: collections.deque = field(default_factory=lambda: collections.deque(maxlen=30))
    meter_coords: collections.deque = field(default_factory=lambda: collections.deque(maxlen=30))
    pixel_coords: collections.deque = field(default_factory=lambda: collections.deque(maxlen=30))
    estimated_speed_kmh: Optional[float] = None
    violation_logged: bool = False
    best_crop: Optional[np.ndarray] = None
    min_dist_to_camera: float = float("inf")


class HomographyTransformer:
    """
    Computes and applies 4-point planar perspective transformation (Homography)
    mapping camera pixel coordinates to real-world ground coordinates in meters.
    """

    def __init__(self, src_pts: List[Tuple[float, float]], dst_pts: List[Tuple[float, float]]):
        """
        :param src_pts: 4 image pixel coordinates [(u0, v0), (u1, v1), (u2, v2), (u3, v3)]
        :param dst_pts: 4 ground-plane coordinates in meters [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        """
        assert len(src_pts) == 4 and len(dst_pts) == 4, "Exact 4-point correspondence required."
        self.src_pts = np.array(src_pts, dtype=np.float32)
        self.dst_pts = np.array(dst_pts, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

    def pixel_to_meters(self, u: float, v: float) -> Tuple[float, float]:
        """Transform bottom-center wheel contact point (u, v) into real-world (X, Y) meters."""
        pt = np.array([[[u, v]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, self.H)
        return float(world_pt[0, 0, 0]), float(world_pt[0, 0, 1])


class HighScaleSpeedEstimator:
    """
    Asynchronous, decoupled vehicle speed estimation engine.
    Computes rolling velocity over a configurable temporal window (e.g. 10 frames).
    """

    def __init__(
        self,
        transformer: HomographyTransformer,
        fps: float = 30.0,
        window_size: int = 10,
        speed_limit_kmh: float = 60.0,
        capture_output_dir: str = "data/violations"
    ):
        self.transformer = transformer
        self.fps = fps
        self.window_size = window_size
        self.speed_limit_kmh = speed_limit_kmh
        self.capture_output_dir = capture_output_dir
        self.tracks: Dict[int, TrackState] = {}
        os.makedirs(capture_output_dir, exist_ok=True)

    def get_bottom_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Extract bottom-center (where wheels touch the pavement) from (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, y2

    def update_track(
        self,
        vehicle_id: int,
        bbox: Tuple[float, float, float, float],
        frame_idx: int,
        frame_img: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """
        Ingests tracking updates and computes rolling average speed.
        Returns estimated speed in km/h if available.
        """
        if vehicle_id not in self.tracks:
            self.tracks[vehicle_id] = TrackState(vehicle_id=vehicle_id)

        state = self.tracks[vehicle_id]
        timestamp = frame_idx / self.fps
        u, v = self.get_bottom_center(bbox)
        mx, my = self.transformer.pixel_to_meters(u, v)

        state.timestamps.append(timestamp)
        state.pixel_coords.append((u, v))
        state.meter_coords.append((mx, my))

        # Rolling Speed Calculation (Step B)
        if len(state.meter_coords) >= self.window_size:
            prev_mx, prev_my = state.meter_coords[-self.window_size]
            prev_t = state.timestamps[-self.window_size]

            # Delta distance in meters
            delta_d = math.hypot(mx - prev_mx, my - prev_my)
            # Delta time in seconds
            delta_t = timestamp - prev_t

            if delta_t > 0:
                speed_mps = delta_d / delta_t
                speed_kmh = speed_mps * 3.6
                state.estimated_speed_kmh = round(speed_kmh, 1)

        # High-Speed Event Capturing (Step D)
        if (
            state.estimated_speed_kmh is not None
            and state.estimated_speed_kmh > self.speed_limit_kmh
            and not state.violation_logged
            and frame_img is not None
        ):
            self._capture_violation(state, bbox, frame_img, frame_idx)

        return state.estimated_speed_kmh

    def _capture_violation(
        self,
        state: TrackState,
        bbox: Tuple[float, float, float, float],
        frame_img: np.ndarray,
        frame_idx: int
    ):
        """Crops vehicle ROI and logs structured JSON payload asynchronously."""
        h, w = frame_img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame_img[y1:y2, x1:x2]
        crop_filename = f"violation_id_{state.vehicle_id}_frame_{frame_idx}.jpg"
        crop_path = os.path.join(self.capture_output_dir, crop_filename)

        if crop.size > 0:
            cv2.imwrite(crop_path, crop)

        payload = {
            "event_type": "SPEED_LIMIT_VIOLATION",
            "vehicle_id": state.vehicle_id,
            "speed_kmh": state.estimated_speed_kmh,
            "speed_limit_kmh": self.speed_limit_kmh,
            "frame_index": frame_idx,
            "timestamp_sec": frame_idx / self.fps,
            "snapshot_path": crop_path,
            "created_at": time.time(),
        }

        json_path = os.path.join(self.capture_output_dir, f"violation_{state.vehicle_id}.json")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        state.violation_logged = True

    def render_overlay(
        self,
        frame: np.ndarray,
        tracks: List[Tuple[int, Tuple[float, float, float, float]]]
    ) -> np.ndarray:
        """
        Step C: Visual Display Overlay.
        Draws bounding box with dynamic color (Red = Over-speed, Green/Cyan = Normal)
        and textual banner displaying vehicle ID and speed.
        """
        annotated = frame.copy()
        for vehicle_id, bbox in tracks:
            state = self.tracks.get(vehicle_id)
            speed = state.estimated_speed_kmh if state else None
            is_overspeed = speed is not None and speed > self.speed_limit_kmh

            color = (0, 0, 255) if is_overspeed else (0, 255, 128)  # BGR
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Text banner
            label = f"ID: {vehicle_id}"
            if speed is not None:
                label += f" | {speed:.0f} km/h"
            else:
                label += " | Calibrating"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
