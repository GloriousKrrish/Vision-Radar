from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from visionradar.cv.detector import Detection
from visionradar.cv.trajectory import TrajectoryPoint, VehicleTrajectory


class KalmanBoxTracker:
    """
    8-state Kalman Filter tracking bounding box state:
    State: [xc, yc, w, h, vxc, vyc, vw, vh]
    where (xc, yc) is center, w is width, h is height, and derivatives represent velocities.
    """
    def __init__(self, bbox: Tuple[float, float, float, float]):
        xc = (bbox[0] + bbox[2]) / 2.0
        yc = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])

        self.x = np.array([xc, yc, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Initial state covariance
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 10.0, 10.0]).astype(np.float32)

        # State transition matrix F (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Process noise covariance Q
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0]).astype(np.float32)

        # Measurement noise covariance R
        self.R = np.diag([4.0, 4.0, 8.0, 8.0]).astype(np.float32)

    def predict(self) -> Tuple[float, float, float, float]:
        """
        Predict state for next time step.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Ensure width and height remain physically positive
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.get_state()

    def update(self, bbox: Tuple[float, float, float, float]):
        """
        Kalman measurement update with observed bounding box.
        """
        xc = (bbox[0] + bbox[2]) / 2.0
        yc = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])

        z = np.array([xc, yc, w, h], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])

    def get_state(self) -> Tuple[float, float, float, float]:
        """
        Returns bounding box [x1, y1, x2, y2] from state.
        """
        xc, yc, w, h = self.x[:4]
        return (
            float(xc - w / 2.0),
            float(yc - h / 2.0),
            float(xc + w / 2.0),
            float(yc + h / 2.0)
        )


class TrackedVehicle:
    """
    Structured active vehicle track container adhering to Phase 2C schema.
    Includes Kalman Filter motion estimation to prevent static freezing on missed detections.
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

        # Kalman Filter state tracker
        self.kf = KalmanBoxTracker(detection.bbox)

        self.trajectory = VehicleTrajectory(track_id, self.vehicle_class)
        self.trajectory.add_point(TrajectoryPoint(frame_index, timestamp, detection.bbox, confidence=detection.confidence))

    def predict(self) -> Tuple[float, float, float, float]:
        """
        Advances Kalman state model. Updates bbox with predicted state so bounding
        boxes propagate continuously even when detector drops frames.
        """
        self.time_since_update += 1
        predicted_bbox = self.kf.predict()
        self.bbox = predicted_bbox
        return predicted_bbox

    def update(self, detection: Detection, frame_index: int, timestamp: float):
        """
        Updates track with matched detector observation.
        """
        self.kf.update(detection.bbox)
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


def _bipartite_match(
    track_boxes: List[Tuple[float, float, float, float]],
    det_boxes: List[Tuple[float, float, float, float]],
    iou_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Computes global optimal bipartite matching between track bounding boxes and detections.
    Returns: (matches [(track_idx, det_idx)], unmatched_tracks, unmatched_dets)
    """
    n_tracks = len(track_boxes)
    n_dets = len(det_boxes)

    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
    for i, tb in enumerate(track_boxes):
        for j, db in enumerate(det_boxes):
            iou_matrix[i, j] = compute_iou(tb, db)

    matches = []
    unmatched_tracks = list(range(n_tracks))
    unmatched_dets = list(range(n_dets))

    if _HAS_SCIPY:
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_r = set()
        matched_c = set()
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((int(r), int(c)))
                matched_r.add(int(r))
                matched_c.add(int(c))
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_r]
        unmatched_dets = [j for j in range(n_dets) if j not in matched_c]
    else:
        # Greedy fallback
        temp_iou = iou_matrix.copy()
        matched_r = set()
        matched_c = set()
        while True:
            if temp_iou.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(temp_iou), temp_iou.shape)
            max_val = temp_iou[max_idx]
            if max_val < iou_threshold:
                break
            r, c = max_idx
            matches.append((int(r), int(c)))
            matched_r.add(int(r))
            matched_c.add(int(c))
            temp_iou[r, :] = -1.0
            temp_iou[:, c] = -1.0
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_r]
        unmatched_dets = [j for j in range(n_dets) if j not in matched_c]

    return matches, unmatched_tracks, unmatched_dets


class ByteTrackTracker(BaseTracker):
    """
    ByteTrack implementation for robust association across high and low confidence detections.
    Features:
    - 8-State Kalman Filter motion prediction (xc, yc, w, h, vxc, vyc, vw, vh)
    - Optimal bipartite linear matching for association (via SciPy linear_sum_assignment)
    - Two-stage association: high-confidence detections first, low-confidence recovery second
    - Center-distance fallback matching for fast-moving vehicles
    - Duplicate track suppression to prevent duplicate ID proliferation
    - Deterministic ID switch and track fragmentation accounting
    """
    def __init__(
        self,
        max_age: int = 30,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.40,
        low_threshold: float = 0.10
    ):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.next_id = 1
        self.active_tracks: Dict[int, TrackedVehicle] = {}
        self.recently_lost_tracks: Dict[int, Dict[str, Any]] = {}
        self.total_id_switches = 0
        self.total_class_switches = 0
        self.total_fragmentations = 0
        self.total_detections_processed = 0

    def update(self, detections: List[Detection], frame_index: int, timestamp: float) -> List[TrackedVehicle]:
        self.total_detections_processed += len(detections)

        # Step 1: Advance Kalman Filter state prediction for all active tracks
        active_tids = list(self.active_tracks.keys())
        for tid in active_tids:
            self.active_tracks[tid].predict()

        # Step 2: Partition detections into High and Low confidence pools
        dets_high: List[Detection] = []
        dets_low: List[Detection] = []
        for d in detections:
            if d.confidence >= self.high_threshold:
                dets_high.append(d)
            elif d.confidence >= self.low_threshold:
                dets_low.append(d)

        # Step 3: Stage 1 association (Active tracks with High-confidence detections)
        matched_tids = set()
        matched_high_dets = set()

        if active_tids and dets_high:
            track_boxes = [self.active_tracks[tid].bbox for tid in active_tids]
            det_boxes = [d.bbox for d in dets_high]
            matches_s1, unassigned_t_idxs, unassigned_dh_idxs = _bipartite_match(
                track_boxes, det_boxes, self.iou_threshold
            )

            for t_idx, d_idx in matches_s1:
                tid = active_tids[t_idx]
                det = dets_high[d_idx]
                trk = self.active_tracks[tid]
                if trk.time_since_update > 1:
                    self.total_fragmentations += 1
                if trk.vehicle_class != det.class_name:
                    self.total_class_switches += 1
                trk.update(det, frame_index, timestamp)
                matched_tids.add(tid)
                matched_high_dets.add(d_idx)

            # Center distance fallback for fast-moving vehicles remaining unassigned
            for t_idx in unassigned_t_idxs:
                tid = active_tids[t_idx]
                if tid in matched_tids:
                    continue
                t_bbox = self.active_tracks[tid].bbox
                tc_x, tc_y = (t_bbox[0] + t_bbox[2]) / 2.0, (t_bbox[1] + t_bbox[3]) / 2.0

                best_dh_idx = -1
                min_dist = 120.0
                for d_idx in unassigned_dh_idxs:
                    if d_idx in matched_high_dets:
                        continue
                    det = dets_high[d_idx]
                    dc_x, dc_y = (det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0
                    dist = float(np.hypot(dc_x - tc_x, dc_y - tc_y))
                    if dist < min_dist:
                        min_dist = dist
                        best_dh_idx = d_idx

                if best_dh_idx >= 0:
                    trk = self.active_tracks[tid]
                    det = dets_high[best_dh_idx]
                    if trk.time_since_update > 1:
                        self.total_fragmentations += 1
                    if trk.vehicle_class != det.class_name:
                        self.total_class_switches += 1
                    trk.update(det, frame_index, timestamp)
                    matched_tids.add(tid)
                    matched_high_dets.add(best_dh_idx)

        # Step 4: Stage 2 association (Unmatched active tracks with Low-confidence detections)
        remaining_tids = [tid for tid in active_tids if tid not in matched_tids]
        if remaining_tids and dets_low:
            rem_track_boxes = [self.active_tracks[tid].bbox for tid in remaining_tids]
            low_det_boxes = [d.bbox for d in dets_low]
            matches_s2, _, _ = _bipartite_match(
                rem_track_boxes, low_det_boxes, max(0.20, self.iou_threshold - 0.10)
            )

            for rt_idx, ld_idx in matches_s2:
                tid = remaining_tids[rt_idx]
                det = dets_low[ld_idx]
                trk = self.active_tracks[tid]
                if trk.time_since_update > 1:
                    self.total_fragmentations += 1
                if trk.vehicle_class != det.class_name:
                    self.total_class_switches += 1
                trk.update(det, frame_index, timestamp)
                matched_tids.add(tid)

        # Step 5: Initiate new tracks ONLY from unmatched High-confidence detections
        for j, det in enumerate(dets_high):
            if j not in matched_high_dets:
                # Check for ID Switch against recently lost tracks
                for lost_tid, lost_meta in list(self.recently_lost_tracks.items()):
                    ref_frame = lost_meta.get("lost_at_frame", lost_meta["last_frame"])
                    if frame_index - ref_frame <= self.max_age * 2:
                        iou_lost = compute_iou(lost_meta["last_bbox"], det.bbox)
                        if iou_lost >= 0.35:
                            self.total_id_switches += 1
                            del self.recently_lost_tracks[lost_tid]
                            break

                new_track = TrackedVehicle(self.next_id, det, frame_index, timestamp)
                self.active_tracks[self.next_id] = new_track
                self.next_id += 1

        # Step 6: Duplicate track suppression (merge overlapping tracks of same class)
        active_list = list(self.active_tracks.values())
        duplicate_tids = set()
        for i in range(len(active_list)):
            for k in range(i + 1, len(active_list)):
                t1 = active_list[i]
                t2 = active_list[k]
                if t1.track_id in duplicate_tids or t2.track_id in duplicate_tids:
                    continue
                if t1.vehicle_class != t2.vehicle_class:
                    continue
                if compute_iou(t1.bbox, t2.bbox) > 0.85:
                    # Keep track with more hits / earlier origin
                    if t1.hits >= t2.hits:
                        duplicate_tids.add(t2.track_id)
                    else:
                        duplicate_tids.add(t1.track_id)

        for dup_id in duplicate_tids:
            if dup_id in self.active_tracks:
                del self.active_tracks[dup_id]

        # Step 7: Purge stale tracks exceeding max_age into recently_lost_tracks
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

        # Step 8: Evict old history from recently_lost_tracks
        evict_lost = [
            tid for tid, m in self.recently_lost_tracks.items()
            if frame_index - m.get("lost_at_frame", m["last_frame"]) > self.max_age * 2
        ]
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

