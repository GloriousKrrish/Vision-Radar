from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any
from visionradar.cv.detector import Detection
from visionradar.cv.trajectory import TrajectoryPoint, VehicleTrajectory

class TrackedVehicle:
    """
    Structured active vehicle track container adhering to Phase 2C schema.
    """
    def __init__(self, track_id: int, detection: Detection, frame_index: int, timestamp: float):
        self.track_id = track_id
        self.vehicle_class = detection.class_name
        self.confidence = float(detection.confidence)
        self.bbox = detection.bbox
        self.first_frame = frame_index
        self.last_frame = frame_index
        self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        self.hits = 1
        self.detection_count = 1
        self.time_since_update = 0
        self.direction = "UNKNOWN"
        self.lane = "UNKNOWN"
        self.anpr_status = "NOT_CONFIGURED"
        self.speed_kmh = None
        self.speed_uncertainty_kmh = None

        self.trajectory = VehicleTrajectory(track_id, self.vehicle_class)
        self.trajectory.add_point(TrajectoryPoint(frame_index, timestamp, detection.bbox, confidence=detection.confidence))

    def update(self, detection: Detection, frame_index: int, timestamp: float):
        self.bbox = detection.bbox
        self.confidence = float(detection.confidence)
        self.last_frame = frame_index
        self.last_timestamp = timestamp
        self.hits += 1
        self.detection_count += 1
        self.time_since_update = 0

        # Estimate simple direction from vertical delta
        if len(self.trajectory.points) > 0:
            first_y = self.trajectory.points[0].anchor_pixel[1]
            curr_y = (detection.bbox[1] + detection.bbox[3]) / 2.0
            if curr_y - first_y > 15:
                self.direction = "SOUTHWARD"
            elif first_y - curr_y > 15:
                self.direction = "NORTHWARD"

            curr_x = (detection.bbox[0] + detection.bbox[2]) / 2.0
            if curr_x < 280:
                self.lane = "Lane 1"
            elif curr_x < 520:
                self.lane = "Lane 2"
            else:
                self.lane = "Lane 3"

        self.trajectory.add_point(TrajectoryPoint(frame_index, timestamp, detection.bbox, confidence=detection.confidence))

    @property
    def duration(self) -> float:
        return max(0.0, self.last_timestamp - self.first_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "duration_sec": round(self.duration, 2),
            "detection_count": self.detection_count,
            "direction": self.direction,
            "lane": self.lane,
            "anpr_status": self.anpr_status,
            "speed_kmh": round(self.speed_kmh, 2) if self.speed_kmh is not None else None,
            "speed_uncertainty_kmh": round(self.speed_uncertainty_kmh, 2) if self.speed_uncertainty_kmh is not None else None
        }


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection], frame_index: int, timestamp: float) -> List[TrackedVehicle]:
        pass

class ByteTrackTracker(BaseTracker):
    """
    ByteTrack implementation for robust association across high and low confidence detections.
    Uses IoU greedy matching, center-distance fallback, and standard MOT metric tracking.
    """
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.active_tracks: Dict[int, TrackedVehicle] = {}
        self.total_id_switches = 0
        self.total_class_switches = 0
        self.total_fragmentations = 0
        self.total_detections_processed = 0

    def update(self, detections: List[Detection], frame_index: int, timestamp: float) -> List[TrackedVehicle]:
        self.total_detections_processed += len(detections)

        # Increment time since update for existing tracks (without falsely counting consecutive frames as fragmentations)
        for t in self.active_tracks.values():
            t.time_since_update += 1

        matched_dets = set()
        matched_tracks = set()

        # Compute IoU matrix between existing tracks and new detections
        track_ids = list(self.active_tracks.keys())
        if track_ids and detections:
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                t_bbox = self.active_tracks[tid].bbox
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = compute_iou(t_bbox, det.bbox)

            # Greedy IoU matching
            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                if max_iou < self.iou_threshold:
                    break

                t_idx, d_idx = max_idx
                tid = track_ids[t_idx]
                if tid not in matched_tracks and d_idx not in matched_dets:
                    track_obj = self.active_tracks[tid]

                    # Track Fragmentation (FM): Track was unobserved for >= 1 frame (time_since_update > 1) before re-matching
                    if track_obj.time_since_update > 1:
                        self.total_fragmentations += 1

                    # Class Flip / Switch: Detector label change for the same track_id
                    if track_obj.vehicle_class != detections[d_idx].class_name:
                        self.total_class_switches += 1

                    track_obj.update(detections[d_idx], frame_index, timestamp)
                    matched_tracks.add(tid)
                    matched_dets.add(d_idx)

                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Center distance fallback matching for fast moving vehicles
        for i, tid in enumerate(track_ids):
            if tid in matched_tracks:
                continue
            t_bbox = self.active_tracks[tid].bbox
            tc_x, tc_y = (t_bbox[0] + t_bbox[2]) / 2.0, (t_bbox[1] + t_bbox[3]) / 2.0

            best_d_idx = -1
            min_dist = 120.0  # max pixel distance search radius

            for j, det in enumerate(detections):
                if j in matched_dets:
                    continue
                dc_x, dc_y = (det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0
                dist = np.hypot(dc_x - tc_x, dc_y - tc_y)
                if dist < min_dist:
                    min_dist = dist
                    best_d_idx = j

            if best_d_idx >= 0:
                track_obj = self.active_tracks[tid]
                if track_obj.time_since_update > 1:
                    self.total_fragmentations += 1
                if track_obj.vehicle_class != detections[best_d_idx].class_name:
                    self.total_class_switches += 1

                track_obj.update(detections[best_d_idx], frame_index, timestamp)
                matched_tracks.add(tid)
                matched_dets.add(best_d_idx)

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_track = TrackedVehicle(self.next_id, det, frame_index, timestamp)
                self.active_tracks[self.next_id] = new_track
                self.next_id += 1

        # Purge stale tracks
        stale_ids = [tid for tid, t in self.active_tracks.items() if t.time_since_update > self.max_age]
        for tid in stale_ids:
            del self.active_tracks[tid]

        return list(self.active_tracks.values())

    def get_tracking_metrics(self) -> Dict[str, Any]:
        total_tracks = self.next_id - 1
        active_count = len(self.active_tracks)
        return {
            "total_tracks_created": total_tracks,
            "active_tracks": active_count,
            "id_switches": self.total_id_switches,
            "class_switches": self.total_class_switches,
            "track_fragmentations": self.total_fragmentations,
            "total_detections_processed": self.total_detections_processed
        }


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return float(iou)

