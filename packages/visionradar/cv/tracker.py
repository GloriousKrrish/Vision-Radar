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
    Uses two-stage association (D_high and D_low), IoU greedy matching, center-distance fallback,
    and standard MOT metric tracking (ID switches, class switches, fragmentations).
    """
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3, high_threshold: float = 0.40):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.next_id = 1
        self.active_tracks: Dict[int, TrackedVehicle] = {}
        self.recently_lost_tracks: Dict[int, Dict[str, Any]] = {}
        self.total_id_switches = 0
        self.total_class_switches = 0
        self.total_fragmentations = 0
        self.total_detections_processed = 0

    def _associate_pool(
        self,
        track_ids: List[int],
        pool_dets: List[Detection],
        matched_tracks: set,
        matched_dets: set,
        frame_index: int,
        timestamp: float
    ):
        available_tids = [tid for tid in track_ids if tid not in matched_tracks]
        available_didxs = [i for i in range(len(pool_dets)) if i not in matched_dets]
        if not available_tids or not available_didxs:
            return

        # 1. IoU Matching
        iou_matrix = np.zeros((len(available_tids), len(available_didxs)), dtype=np.float32)
        for i, tid in enumerate(available_tids):
            t_bbox = self.active_tracks[tid].bbox
            for j, d_idx in enumerate(available_didxs):
                iou_matrix[i, j] = compute_iou(t_bbox, pool_dets[d_idx].bbox)

        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            if max_iou < self.iou_threshold:
                break

            t_i, d_j = max_idx
            tid = available_tids[t_i]
            d_idx = available_didxs[d_j]

            track_obj = self.active_tracks[tid]
            if track_obj.time_since_update > 1:
                self.total_fragmentations += 1
            if track_obj.vehicle_class != pool_dets[d_idx].class_name:
                self.total_class_switches += 1

            track_obj.update(pool_dets[d_idx], frame_index, timestamp)
            matched_tracks.add(tid)
            matched_dets.add(d_idx)

            iou_matrix[t_i, :] = -1.0
            iou_matrix[:, d_j] = -1.0

        # 2. Center distance fallback matching for fast moving vehicles
        for tid in available_tids:
            if tid in matched_tracks:
                continue
            t_bbox = self.active_tracks[tid].bbox
            tc_x, tc_y = (t_bbox[0] + t_bbox[2]) / 2.0, (t_bbox[1] + t_bbox[3]) / 2.0

            best_d_idx = -1
            min_dist = 120.0  # max pixel distance search radius

            for d_idx in available_didxs:
                if d_idx in matched_dets:
                    continue
                det = pool_dets[d_idx]
                dc_x, dc_y = (det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0
                dist = np.hypot(dc_x - tc_x, dc_y - tc_y)
                if dist < min_dist:
                    min_dist = dist
                    best_d_idx = d_idx

            if best_d_idx >= 0:
                track_obj = self.active_tracks[tid]
                if track_obj.time_since_update > 1:
                    self.total_fragmentations += 1
                if track_obj.vehicle_class != pool_dets[best_d_idx].class_name:
                    self.total_class_switches += 1

                track_obj.update(pool_dets[best_d_idx], frame_index, timestamp)
                matched_tracks.add(tid)
                matched_dets.add(best_d_idx)

    def update(self, detections: List[Detection], frame_index: int, timestamp: float) -> List[TrackedVehicle]:
        self.total_detections_processed += len(detections)

        # Increment time since update for existing tracks
        for t in self.active_tracks.values():
            t.time_since_update += 1

        # Partition detections into High and Low confidence pools (genuine ByteTrack)
        dets_high: List[Detection] = []
        dets_low: List[Detection] = []
        for d in detections:
            if d.confidence >= self.high_threshold:
                dets_high.append(d)
            else:
                dets_low.append(d)

        track_ids = list(self.active_tracks.keys())
        matched_tracks = set()
        matched_high_dets = set()
        matched_low_dets = set()

        # Stage 1: Associate active tracks with High-confidence detections
        self._associate_pool(track_ids, dets_high, matched_tracks, matched_high_dets, frame_index, timestamp)

        # Stage 2: Associate remaining active tracks with Low-confidence detections
        self._associate_pool(track_ids, dets_low, matched_tracks, matched_low_dets, frame_index, timestamp)

        # Stage 3: Initiate new tracks ONLY from unmatched High-confidence detections
        for j, det in enumerate(dets_high):
            if j not in matched_high_dets:
                # Check for ID Switch against recently lost tracks
                for lost_tid, lost_meta in list(self.recently_lost_tracks.items()):
                    ref_frame = lost_meta.get("lost_at_frame", lost_meta["last_frame"])
                    if frame_index - ref_frame <= self.max_age:
                        iou_lost = compute_iou(lost_meta["last_bbox"], det.bbox)
                        if iou_lost >= 0.35:
                            self.total_id_switches += 1
                            del self.recently_lost_tracks[lost_tid]
                            break

                new_track = TrackedVehicle(self.next_id, det, frame_index, timestamp)
                self.active_tracks[self.next_id] = new_track
                self.next_id += 1

        # Purge stale tracks and record them into recently_lost_tracks for ID switch accounting
        stale_ids = [tid for tid, t in self.active_tracks.items() if t.time_since_update > self.max_age]
        for tid in stale_ids:
            stale_trk = self.active_tracks[tid]
            self.recently_lost_tracks[tid] = {
                "last_bbox": stale_trk.bbox,
                "last_frame": stale_trk.last_frame,
                "lost_at_frame": frame_index,
                "vehicle_class": stale_trk.vehicle_class
            }
            del self.active_tracks[tid]

        # Evict old history from recently_lost_tracks
        evict_lost = [tid for tid, m in self.recently_lost_tracks.items() if frame_index - m.get("lost_at_frame", m["last_frame"]) > self.max_age * 2]
        for tid in evict_lost:
            del self.recently_lost_tracks[tid]

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

