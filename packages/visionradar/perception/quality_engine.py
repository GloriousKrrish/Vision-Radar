"""
VisionRadar Perception Quality Engine — Phase 6 Core Module

Provides diagnostic quality assessment across:
1. Detection Quality (Confidence, size bins, edge clipping, ROI membership, geometry, duplicate flags)
2. Temporal & Track Quality (Class consensus, gap categorization, recovery, fragmentation, track quality score)
3. Trajectory & Motion Quality (Spatial jitter, path continuity, homography ROI safety, impossible motion guard)
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from visionradar.perception.schemas import (
    VehicleDetection, VehicleTrack, BoundingBox, VehicleClass, TrackState,
    DetectionQualityFlag, TrajectoryStatus, CalibrationStatus
)


def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm to test if point (x, y) is inside polygon [(x0,y0), (x1,y1), ...]."""
    if not polygon or len(polygon) < 3:
        return True
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def compute_iou_boxes(b1: BoundingBox, b2: BoundingBox) -> float:
    x1 = max(b1.x1, b2.x1)
    y1 = max(b1.y1, b2.y1)
    x2 = min(b1.x2, b2.x2)
    y2 = min(b1.y2, b2.y2)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
    area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


class DetectionQualityEngine:
    """
    Evaluates raw detections per frame and attaches diagnostic metadata flags without dropping detections.
    """
    def __init__(self, low_conf_thresh: float = 0.35, roi_polygon: Optional[List[Tuple[float, float]]] = None):
        self.low_conf_thresh = low_conf_thresh
        self.roi_polygon = roi_polygon

    def categorize_size(self, width: float, height: float) -> str:
        max_dim = max(width, height)
        if max_dim < 16:
            return "<16x16"
        elif max_dim <= 32:
            return "16-32"
        elif max_dim <= 64:
            return "32-64"
        elif max_dim <= 128:
            return "64-128"
        else:
            return ">128"

    def evaluate_detections(
        self,
        detections: List[VehicleDetection],
        frame_width: int = 1920,
        frame_height: int = 1080
    ) -> List[VehicleDetection]:
        if not detections:
            return []

        # Find duplicate candidates via pairwise IoU
        duplicate_indices = set()
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if detections[i].vehicle_class == detections[j].vehicle_class:
                    iou = compute_iou_boxes(detections[i].bbox, detections[j].bbox)
                    if iou > 0.70:
                        if detections[i].confidence >= detections[j].confidence:
                            duplicate_indices.add(j)
                        else:
                            duplicate_indices.add(i)

        evaluated_detections = []
        for idx, det in enumerate(detections):
            b = det.bbox
            w = max(0.0, b.x2 - b.x1)
            h = max(0.0, b.y2 - b.y1)
            area = w * h
            aspect_ratio = w / (h + 1e-6)
            bottom_center_x = (b.x1 + b.x2) / 2.0
            bottom_center_y = b.y2

            size_cat = self.categorize_size(w, h)
            flags = []

            if det.confidence < self.low_conf_thresh:
                flags.append(DetectionQualityFlag.LOW_CONFIDENCE.value)

            if w < 16 or h < 16:
                flags.append(DetectionQualityFlag.TOO_SMALL.value)

            if b.x1 <= 5 or b.y1 <= 5 or b.x2 >= (frame_width - 5) or b.y2 >= (frame_height - 5):
                flags.append(DetectionQualityFlag.EDGE_CLIPPED.value)

            if aspect_ratio < 0.20 or aspect_ratio > 4.50 or area < 64:
                flags.append(DetectionQualityFlag.SUSPICIOUS_GEOMETRY.value)

            if idx in duplicate_indices:
                flags.append(DetectionQualityFlag.DUPLICATE_CANDIDATE.value)

            if self.roi_polygon and not point_in_polygon(bottom_center_x, bottom_center_y, self.roi_polygon):
                flags.append(DetectionQualityFlag.OUTSIDE_ROI.value)

            if not flags:
                flags.append(DetectionQualityFlag.VALID.value)

            is_valid = DetectionQualityFlag.SUSPICIOUS_GEOMETRY.value not in flags and DetectionQualityFlag.TOO_SMALL.value not in flags

            evaluated_det = VehicleDetection(
                detection_id=det.detection_id,
                frame_id=det.frame_id,
                bbox=det.bbox,
                confidence=det.confidence,
                vehicle_class=det.vehicle_class,
                class_confidence=det.class_confidence,
                quality_flags=flags,
                size_category=size_cat,
                detector_validity=is_valid
            )
            evaluated_detections.append(evaluated_det)

        return evaluated_detections


class TrackQualityEngine:
    """
    Maintains persistent track diagnostics:
    - Class consensus history
    - Gap classification (SHORT_GAP: 1-3, MEDIUM_GAP: 4-10, LONG_GAP: >10)
    - Track recovery & true fragmentation counters
    - Overall track quality score (0-100)
    """
    def __init__(self, short_gap_max: int = 3, medium_gap_max: int = 10):
        self.short_gap_max = short_gap_max
        self.medium_gap_max = medium_gap_max

        # Per track state dictionaries
        self.class_histories: Dict[int, List[VehicleClass]] = {}
        self.class_flip_counts: Dict[int, int] = {}
        self.gap_histories: Dict[int, List[Tuple[int, str]]] = {}
        self.consecutive_missing: Dict[int, int] = {}
        self.total_missing: Dict[int, int] = {}
        self.recovery_counts: Dict[int, int] = {}
        self.fragmentation_counts: Dict[int, int] = {}
        self.prev_raw_classes: Dict[int, VehicleClass] = {}

    def update_track_quality(self, tracks: List[VehicleTrack], current_frame: int) -> List[VehicleTrack]:
        updated_tracks = []

        for trk in tracks:
            tid = trk.track_id
            raw_c = trk.vehicle_class

            # Initialize history for new track
            if tid not in self.class_histories:
                self.class_histories[tid] = [raw_c]
                self.class_flip_counts[tid] = 0
                self.gap_histories[tid] = []
                self.consecutive_missing[tid] = 0
                self.total_missing[tid] = 0
                self.recovery_counts[tid] = 0
                self.fragmentation_counts[tid] = 0
                self.prev_raw_classes[tid] = raw_c
            else:
                self.class_histories[tid].append(raw_c)
                if raw_c != self.prev_raw_classes[tid]:
                    self.class_flip_counts[tid] += 1
                self.prev_raw_classes[tid] = raw_c

            # Check detection hit vs miss
            if trk.frames_since_last_detection == 0:
                # Active detection hit
                if self.consecutive_missing[tid] > 0:
                    gap_len = self.consecutive_missing[tid]
                    if gap_len <= self.short_gap_max:
                        cat = "SHORT_GAP"
                    elif gap_len <= self.medium_gap_max:
                        cat = "MEDIUM_GAP"
                    else:
                        cat = "LONG_GAP"

                    self.gap_histories[tid].append((gap_len, cat))
                    self.recovery_counts[tid] += 1
                    # True fragmentation: unobserved for >= 1 frame before re-matching
                    if gap_len >= 1:
                        self.fragmentation_counts[tid] += 1

                    self.consecutive_missing[tid] = 0
            else:
                # Missed detection frame
                self.consecutive_missing[tid] += 1
                self.total_missing[tid] += 1

            # Compute Class Consensus
            hist = self.class_histories[tid]
            class_counts: Dict[VehicleClass, int] = {}
            for c in hist:
                class_counts[c] = class_counts.get(c, 0) + 1
            consensus_c = max(class_counts, key=class_counts.get)
            class_conf = float(class_counts[consensus_c] / len(hist))

            # Compute Track Quality Score (0-100)
            conf_score = trk.confidence * 100.0
            continuity_score = max(0.0, 100.0 - (self.total_missing[tid] * 5.0))
            class_stability = class_conf * 100.0
            lifetime_score = min(100.0, trk.frames_tracked * 5.0)
            spatial_stability = 100.0 if trk.velocity_px_per_frame is not None else 80.0

            q_score = (
                0.30 * conf_score +
                0.25 * continuity_score +
                0.20 * class_stability +
                0.15 * spatial_stability +
                0.10 * lifetime_score
            )
            q_score = max(0.0, min(100.0, q_score))

            if q_score >= 80.0:
                q_level = "HIGH"
            elif q_score >= 50.0:
                q_level = "MEDIUM"
            else:
                q_level = "LOW"

            # Update fields on VehicleTrack instance
            trk.raw_class = raw_c
            trk.consensus_class = consensus_c
            trk.vehicle_class = consensus_c
            trk.class_confidence = class_conf
            trk.class_flip_count = self.class_flip_counts[tid]
            trk.track_quality_score = q_score
            trk.track_quality_level = q_level
            trk.missing_frames = self.total_missing[tid]
            trk.recovery_count = self.recovery_counts[tid]
            trk.fragmentation_count = self.fragmentation_counts[tid]
            trk.observed_internal_id_changes = 0 # Default observed internal ID continuity

            updated_tracks.append(trk)

        return updated_tracks


class TrajectoryQualityEngine:
    """
    Evaluates trajectory motion safety, spatial jitter, calibration boundary membership,
    numerical homography stability, and impossible motion anomalies.
    """
    def __init__(self, roi_polygon: Optional[List[Tuple[float, float]]] = None):
        self.roi_polygon = roi_polygon

    def evaluate_track_trajectory(
        self,
        track: VehicleTrack,
        trajectory_points: List[Any],
        H_matrix: Optional[np.ndarray] = None
    ) -> VehicleTrack:
        if not trajectory_points or len(trajectory_points) < 2:
            track.trajectory_status = TrajectoryStatus.TRAJECTORY_VALID.value
            track.calibration_status = CalibrationStatus.CALIBRATION_VALID.value
            track.motion_anomaly = False
            track.motion_anomaly_reason = None
            return track

        # Bounding box bottom-center anchor
        b = track.bbox
        bc_x = (b.x1 + b.x2) / 2.0
        bc_y = b.y2

        # 1. Calibration Boundary & Homography Check
        calib_status = CalibrationStatus.CALIBRATION_VALID.value
        if self.roi_polygon and not point_in_polygon(bc_x, bc_y, self.roi_polygon):
            calib_status = CalibrationStatus.OUT_OF_ROI.value

        if H_matrix is not None:
            try:
                cond = np.linalg.cond(H_matrix)
                if cond > 1e8 or math.isnan(cond):
                    calib_status = CalibrationStatus.CALIBRATION_UNSTABLE.value
            except Exception:
                calib_status = CalibrationStatus.CALIBRATION_UNSTABLE.value

        # 2. Motion Anomaly & Jitter Analysis
        motion_anomaly = False
        anomaly_reason = None
        traj_status = TrajectoryStatus.TRAJECTORY_VALID.value

        pts = trajectory_points
        recent_pts = pts[-5:]
        deltas = []
        for i in range(1, len(recent_pts)):
            p1 = recent_pts[i - 1].anchor_pixel
            p2 = recent_pts[i].anchor_pixel
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            df = max(1, recent_pts[i].frame_index - recent_pts[i - 1].frame_index)
            velocity_px = dist / df
            deltas.append(velocity_px)

        if deltas:
            max_disp = max(deltas)
            avg_disp = sum(deltas) / len(deltas)

            # Impossible single frame jump (> 150 pixels in 1 frame)
            if max_disp > 150.0:
                motion_anomaly = True
                anomaly_reason = f"IMPOSSIBLE_SINGLE_FRAME_DISPLACEMENT_{max_disp:.1f}PX"
                traj_status = TrajectoryStatus.TRAJECTORY_INVALID.value

            # Sudden acceleration / velocity jump (> 4x average displacement)
            elif max_disp > 4.0 * max(10.0, avg_disp):
                motion_anomaly = True
                anomaly_reason = f"EXTREME_VELOCITY_SPIKE_{max_disp:.1f}PX"
                traj_status = TrajectoryStatus.TRAJECTORY_SUSPICIOUS.value

            # Oscillation check
            if len(deltas) >= 3:
                var_disp = np.var(deltas)
                if var_disp > 500.0:
                    traj_status = TrajectoryStatus.TRAJECTORY_SUSPICIOUS.value

        track.trajectory_status = traj_status
        track.calibration_status = calib_status
        track.motion_anomaly = motion_anomaly
        track.motion_anomaly_reason = anomaly_reason

        return track
