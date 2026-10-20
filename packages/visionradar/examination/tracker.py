from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from visionradar.examination.schemas import PersonDetection, ExaminationObjectDetection

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(areaA + areaB - interArea)

def calculate_center_distance(boxA: List[float], boxB: List[float]) -> float:
    c1 = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
    c2 = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

def calculate_containment(inner_box: List[float], outer_box: List[float]) -> float:
    """Computes fraction of inner_box contained within outer_box."""
    xA = max(inner_box[0], outer_box[0])
    yA = max(inner_box[1], outer_box[1])
    xB = min(inner_box[2], outer_box[2])
    yB = min(inner_box[3], outer_box[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    innerArea = max(1e-6, (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1]))
    return float(interArea / innerArea)

class ExaminationPersonTrack:
    """Represents a single continuous visual observation track of a person."""
    def __init__(self, track_id: int, bbox: List[float], confidence: float, frame_index: int):
        self.track_id = track_id
        self.bbox = list(bbox)
        self.confidence = confidence
        self.first_frame = frame_index
        self.last_frame = frame_index
        self.time_since_update = 0
        self.history: List[Dict[str, Any]] = [{"frame": frame_index, "bbox": list(bbox)}]
        self.assigned_zone: Optional[str] = None
        self.zone_confidence: Optional[float] = None
        self.orientation: Optional[str] = "forward"

    def update(self, bbox: List[float], confidence: float, frame_index: int):
        self.bbox = list(bbox)
        self.confidence = confidence
        self.last_frame = frame_index
        self.time_since_update = 0
        self.history.append({"frame": frame_index, "bbox": list(bbox)})
        if len(self.history) > 300:
            self.history.pop(0)

    def mark_missed(self):
        self.time_since_update += 1

class PersonTracker:
    """
    Dedicated tracker for maintaining temporary visual track continuity in examination settings.
    Isolates person tracking state from vehicle ByteTrack instances.
    """
    def __init__(self, max_lost_frames: int = 30, iou_threshold: float = 0.30):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.tracks: List[ExaminationPersonTrack] = []

    def update(self, raw_person_detections: List[Dict[str, Any]], frame_index: int) -> List[PersonDetection]:
        """
        Associates incoming person detections with existing tracks using greedy IoU matching.
        """
        for t in self.tracks:
            t.mark_missed()

        unmatched_dets = list(range(len(raw_person_detections)))
        matched_tracks = set()

        if self.tracks and raw_person_detections:
            cost_matrix = np.zeros((len(self.tracks), len(raw_person_detections)), dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                for j, det in enumerate(raw_person_detections):
                    cost_matrix[i, j] = calculate_iou(trk.bbox, det["bbox"])

            while True:
                max_val = np.max(cost_matrix) if cost_matrix.size > 0 else 0.0
                if max_val < self.iou_threshold:
                    break
                trk_idx, det_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                if trk_idx in matched_tracks or det_idx not in unmatched_dets:
                    cost_matrix[trk_idx, det_idx] = -1.0
                    continue

                det = raw_person_detections[det_idx]
                self.tracks[trk_idx].update(det["bbox"], det["confidence"], frame_index)
                matched_tracks.add(trk_idx)
                unmatched_dets.remove(det_idx)
                cost_matrix[trk_idx, :] = -1.0
                cost_matrix[:, det_idx] = -1.0

        for det_idx in unmatched_dets:
            det = raw_person_detections[det_idx]
            new_trk = ExaminationPersonTrack(
                track_id=self.next_track_id,
                bbox=det["bbox"],
                confidence=det["confidence"],
                frame_index=frame_index
            )
            self.next_track_id += 1
            self.tracks.append(new_trk)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]

        active_persons = []
        for t in self.tracks:
            if t.time_since_update == 0:
                active_persons.append(
                    PersonDetection(
                        track_id=t.track_id,
                        bbox=t.bbox,
                        confidence=t.confidence,
                        assigned_zone=t.assigned_zone,
                        zone_confidence=t.zone_confidence,
                        posture_orientation=t.orientation
                    )
                )

        return active_persons

    def predict_step(self, frame_index: int) -> List[PersonDetection]:
        """
        Propagates active person tracks during skipped frames for speed optimization.
        Keeps visual boxes active without running expensive neural network inference.
        """
        active_persons = []
        for t in self.tracks:
            if t.time_since_update <= 3:
                active_persons.append(
                    PersonDetection(
                        track_id=t.track_id,
                        bbox=t.bbox,
                        confidence=round(t.confidence * 0.99, 4),
                        assigned_zone=t.assigned_zone,
                        zone_confidence=t.zone_confidence,
                        posture_orientation=t.orientation
                    )
                )
        return active_persons


class ExaminationObjectTrack:
    """Represents a continuous track of an examination object (e.g. cell phone, paper, laptop)."""
    def __init__(self, object_track_id: int, object_class: str, bbox: List[float], confidence: float, frame_index: int):
        self.object_track_id = object_track_id
        self.object_class = object_class
        self.bbox = list(bbox)
        self.confidence = confidence
        self.first_frame = frame_index
        self.last_frame = frame_index
        self.time_since_update = 0
        self.observation_count = 1
        self.history: List[Dict[str, Any]] = [{"frame": frame_index, "bbox": list(bbox), "conf": confidence}]
        self.associated_person_track_id: Optional[int] = None
        self.association_status: str = "UNASSOCIATED"
        self.association_confidence: float = 0.0
        self.uncertainty_reasons: List[str] = []

    def update(self, bbox: List[float], confidence: float, frame_index: int):
        self.bbox = list(bbox)
        self.confidence = confidence
        self.last_frame = frame_index
        self.time_since_update = 0
        self.observation_count += 1
        self.history.append({"frame": frame_index, "bbox": list(bbox), "conf": confidence})
        if len(self.history) > 150:
            self.history.pop(0)

    def mark_missed(self):
        self.time_since_update += 1


class ExaminationObjectTracker:
    """
    Dedicated tracker for physical examination objects across sequential frames.
    Maintains track continuity under hand occlusions and movements.
    """
    def __init__(self, max_lost_frames: int = 20, iou_threshold: float = 0.20, distance_threshold_px: float = 90.0):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.distance_threshold_px = distance_threshold_px
        self.next_object_track_id = 1
        self.tracks: List[ExaminationObjectTrack] = []

    def update(self, raw_objects: List[ExaminationObjectDetection], frame_index: int) -> List[ExaminationObjectDetection]:
        for t in self.tracks:
            t.mark_missed()

        unmatched = list(range(len(raw_objects)))
        matched_tracks = set()

        if self.tracks and raw_objects:
            cost_matrix = np.full((len(self.tracks), len(raw_objects)), -1.0, dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                for j, obj in enumerate(raw_objects):
                    if trk.object_class != obj.object_class:
                        continue
                    iou = calculate_iou(trk.bbox, obj.bbox)
                    dist = calculate_center_distance(trk.bbox, obj.bbox)
                    if iou >= self.iou_threshold:
                        cost_matrix[i, j] = iou
                    elif dist <= self.distance_threshold_px:
                        cost_matrix[i, j] = max(0.05, 1.0 - (dist / self.distance_threshold_px) * 0.8)

            while True:
                max_val = np.max(cost_matrix) if cost_matrix.size > 0 else -1.0
                if max_val < 0.05:
                    break
                trk_idx, obj_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                if trk_idx in matched_tracks or obj_idx not in unmatched:
                    cost_matrix[trk_idx, obj_idx] = -1.0
                    continue

                obj = raw_objects[obj_idx]
                self.tracks[trk_idx].update(obj.bbox, obj.confidence, frame_index)
                obj.object_track_id = self.tracks[trk_idx].object_track_id
                matched_tracks.add(trk_idx)
                unmatched.remove(obj_idx)
                cost_matrix[trk_idx, :] = -1.0
                cost_matrix[:, obj_idx] = -1.0

        for obj_idx in unmatched:
            obj = raw_objects[obj_idx]
            new_trk = ExaminationObjectTrack(
                object_track_id=self.next_object_track_id,
                object_class=obj.object_class,
                bbox=obj.bbox,
                confidence=obj.confidence,
                frame_index=frame_index
            )
            obj.object_track_id = self.next_object_track_id
            self.next_object_track_id += 1
            self.tracks.append(new_trk)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]
        return raw_objects

    def predict_step(self, frame_index: int) -> List[ExaminationObjectDetection]:
        """
        Propagates active object tracks during skipped frames for speed optimization.
        """
        active_objects = []
        for t in self.tracks:
            if t.time_since_update <= 3:
                active_objects.append(
                    ExaminationObjectDetection(
                        object_id=t.object_track_id,
                        object_track_id=t.object_track_id,
                        object_class=t.object_class,
                        bbox=t.bbox,
                        confidence=round(t.confidence * 0.99, 4),
                        near_person_track_id=t.associated_person_track_id,
                        association_status=t.association_status,
                        association_confidence=t.association_confidence,
                        distance_to_person_px=None
                    )
                )
        return active_objects


def associate_persons_and_objects(
    persons: List[PersonDetection],
    objects: List[ExaminationObjectDetection],
    object_tracks: Optional[List[ExaminationObjectTrack]] = None
):
    """
    Computes evidence-based spatial and geometric association between detected objects and person tracks.
    Never equates simple proximity with interaction; represents association with explicit uncertainty.
    """
    if not objects or not persons:
        for obj in objects:
            obj.association_status = "UNASSOCIATED"
            obj.association_confidence = 0.0
        return

    obj_track_map = {t.object_track_id: t for t in (object_tracks or [])}

    for obj in objects:
        candidates = []
        for p in persons:
            c_dist = calculate_center_distance(obj.bbox, p.bbox)
            containment = calculate_containment(obj.bbox, p.bbox)
            p_w = p.bbox[2] - p.bbox[0]
            p_h = p.bbox[3] - p.bbox[1]
            p_diag = float(np.hypot(p_w, p_h))

            # In desk-seated examinations, interaction zone is lower 65% of person bbox or within reach radius
            desk_reach_radius = max(90.0, p_diag * 0.70)
            p_desk_y = p.bbox[1] + p_h * 0.50
            obj_cy = (obj.bbox[1] + obj.bbox[3]) / 2.0
            is_lower_or_desk = obj_cy >= p_desk_y - 40.0

            if c_dist <= desk_reach_radius:
                # Heuristic evidence score based on containment, reach, and distance
                proximity_score = max(0.1, 1.0 - (c_dist / desk_reach_radius))
                if containment >= 0.20 or (c_dist < 80.0 and is_lower_or_desk):
                    status = "INTERACTING"
                    assoc_conf = min(0.95, 0.65 + 0.25 * containment + 0.10 * proximity_score)
                else:
                    status = "NEARBY"
                    assoc_conf = round(0.40 + 0.40 * proximity_score, 3)

                candidates.append({
                    "person": p,
                    "distance": c_dist,
                    "containment": containment,
                    "assoc_conf": assoc_conf,
                    "status": status
                })

        if not candidates:
            obj.association_status = "UNASSOCIATED"
            obj.association_confidence = 0.0
            obj.near_person_track_id = None
            obj.distance_to_person_px = None
            continue

        candidates.sort(key=lambda c: c["assoc_conf"], reverse=True)
        top = candidates[0]

        # Ambiguity check: if 2nd candidate is within 15% association score, mark as AMBIGUOUS
        if len(candidates) >= 2 and (top["assoc_conf"] - candidates[1]["assoc_conf"]) < 0.15:
            obj.association_status = "AMBIGUOUS"
            obj.association_confidence = round(top["assoc_conf"] * 0.85, 3)
            obj.near_person_track_id = top["person"].track_id
            obj.distance_to_person_px = round(top["distance"], 1)
        else:
            obj.association_status = top["status"]
            obj.association_confidence = round(top["assoc_conf"], 3)
            obj.near_person_track_id = top["person"].track_id
            obj.distance_to_person_px = round(top["distance"], 1)

        # Sync to object track if available
        if obj.object_track_id and obj.object_track_id in obj_track_map:
            t = obj_track_map[obj.object_track_id]
            t.associated_person_track_id = obj.near_person_track_id
            t.association_status = obj.association_status
            t.association_confidence = obj.association_confidence

        # Populate person's associated objects list
        for p in persons:
            if p.track_id == obj.near_person_track_id:
                p.associated_objects.append({
                    "object_id": obj.object_id,
                    "object_track_id": obj.object_track_id,
                    "object_class": obj.object_class,
                    "association_status": obj.association_status,
                    "association_confidence": obj.association_confidence,
                    "bbox": obj.bbox
                })

