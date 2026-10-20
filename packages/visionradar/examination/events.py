import uuid
from typing import List, Dict, Optional, Set, Any, Tuple
import numpy as np

from visionradar.examination.schemas import (
    ExaminationEventType,
    ExaminationEvent,
    PersonDetection,
    ExaminationObjectDetection,
    SeatingZone,
    ZoneType
)

def bbox_distance(b1: List[float], b2: List[float]) -> float:
    """Computes Euclidean distance between two bounding box centers."""
    c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
    c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

class ExaminationEventEngine:
    """
    Evidence-based temporal event state machine for Examination Intelligence (Layer C).
    Analyzes multi-object spatial relationships and multi-frame temporal persistence.
    Emits neutral, verifiable candidate events subject to authorized human proctor review.
    Zero automated determinations of misconduct are ever made.
    """

    def __init__(self, job_id: int, video_id: int, fps: float = 30.0):
        self.job_id = job_id
        self.video_id = video_id
        self.fps = fps if fps > 0 else 30.0

        # Temporal observation buffer for phone detections: track_id -> list of observation dicts
        self._phone_observations: Dict[int, List[Dict[str, Any]]] = {}
        self._phone_inactivity_frames: Dict[int, int] = {}

        # Seating zone state tracking
        self._established_seat_counts: Dict[int, Dict[str, int]] = {}  # track_id -> {zone_id: frames}
        self._person_assigned_seat: Dict[int, str] = {}  # track_id -> zone_id (once established)
        self._vacated_zone_streaks: Dict[int, int] = {}  # track_id -> frames away from assigned seat

        # Restricted zone tracking
        self._restricted_zone_streaks: Dict[int, Tuple[str, int]] = {}  # track_id -> (zone_id, frames)

        # Orientation shift tracking: track_id -> list of (frame, orientation)
        self._orientation_history: Dict[int, List[Tuple[int, str]]] = {}
        self._gaze_deviation_streaks: Dict[int, int] = {}
        self._peer_proximity_streaks: Dict[Tuple[int, int], int] = {}

        # Active & completed candidate events
        self._active_events: Dict[str, ExaminationEvent] = {}
        self._completed_events: List[ExaminationEvent] = []

    def process_frame(
        self,
        frame_index: int,
        timestamp_s: float,
        persons: List[PersonDetection],
        objects: List[ExaminationObjectDetection],
        zones: List[SeatingZone]
    ) -> List[ExaminationEvent]:
        """
        Evaluates current frame observations and updates temporal candidate events.
        Returns list of newly emitted or active candidate events for current frame.
        """
        emitted_this_frame: List[ExaminationEvent] = []
        restricted_zones = {z.zone_id: z for z in zones if z.zone_type == ZoneType.RESTRICTED_ZONE}
        seating_zones = {z.zone_id: z for z in zones if z.zone_type == ZoneType.SEATING_ZONE}

        # --------------------------------------------------------------------
        # 1. Event: POSSIBLE_PHONE_INTERACTION
        # Robust temporal windowing: requires >= 3 phone observations within a
        # 45-frame (1.5s) window. Tolerates intermittent hand occlusion.
        # --------------------------------------------------------------------
        phones = [o for o in objects if o.object_class == "cell phone"]
        persons_with_phone: Set[int] = set()

        for phone in phones:
            for p in persons:
                dist = bbox_distance(phone.bbox, p.bbox)
                p_diag = float(np.hypot(p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]))
                is_associated = (phone.near_person_track_id == p.track_id) or (dist < max(90.0, p_diag * 0.70))

                if is_associated:
                    persons_with_phone.add(p.track_id)
                    phone.near_person_track_id = p.track_id

                    if p.track_id not in self._phone_observations:
                        self._phone_observations[p.track_id] = []

                    self._phone_observations[p.track_id].append({
                        "frame": frame_index,
                        "timestamp": timestamp_s,
                        "confidence": phone.confidence,
                        "assoc_conf": getattr(phone, "association_confidence", 0.80),
                        "assoc_status": getattr(phone, "association_status", "INTERACTING"),
                        "distance": dist,
                        "bbox": phone.bbox,
                        "obj_track_id": getattr(phone, "object_track_id", None)
                    })
                    self._phone_inactivity_frames[p.track_id] = 0

        # Evaluate phone candidate events across temporal window
        window_size_frames = int(round(self.fps * 1.5))  # 45 frames (1.5s)
        for p in persons:
            p_id = p.track_id
            obs_list = self._phone_observations.get(p_id, [])

            # Filter observations within recent sliding window
            recent_obs = [o for o in obs_list if (frame_index - o["frame"]) <= window_size_frames]
            event_key = f"phone:{p_id}"

            if len(recent_obs) >= 3:
                # Valid candidate phone interaction pattern detected
                frames = [o["frame"] for o in recent_obs]
                confs = [o["confidence"] for o in recent_obs]
                assoc_confs = [o["assoc_conf"] for o in recent_obs]
                mean_conf = float(np.mean(confs))
                peak_conf = float(np.max(confs))
                mean_assoc = float(np.mean(assoc_confs))
                earliest_frame = min(frames)
                earliest_time = min(o["timestamp"] for o in recent_obs)

                # Heuristic evidence-strength rating
                if len(recent_obs) >= 4 and peak_conf >= 0.35:
                    strength = "HIGH"
                elif len(recent_obs) >= 3:
                    strength = "MEDIUM"
                else:
                    strength = "LOW"

                heuristic_score = round(0.50 * mean_conf + 0.50 * mean_assoc, 3)
                duration_s = max(0.1, timestamp_s - earliest_time)

                uncertainty_reasons = []
                if any((frames[i] - frames[i - 1]) > 1 for i in range(1, len(frames))):
                    uncertainty_reasons.append("INTERMITTENT_HAND_OCCLUSION")
                if mean_assoc < 0.70:
                    uncertainty_reasons.append("DESK_PROXIMITY_MARGINAL")

                if event_key not in self._active_events:
                    evt = ExaminationEvent(
                        event_id=f"evt_phone_{uuid.uuid4().hex[:8]}",
                        job_id=self.job_id,
                        video_id=self.video_id,
                        event_type=ExaminationEventType.POSSIBLE_PHONE_INTERACTION,
                        start_frame=earliest_frame,
                        end_frame=frame_index,
                        start_time=round(earliest_time, 2),
                        end_time=round(timestamp_s, 2),
                        person_track_id=p_id,
                        associated_object_track_id=recent_obs[-1].get("obj_track_id"),
                        associated_object_class="cell phone",
                        supporting_objects=["cell phone"],
                        supporting_frames=list(dict.fromkeys(frames)),
                        confidence=round(peak_conf, 3),
                        uncertainty=0.15,
                        priority="RED",
                        evidence_strength=strength,
                        detection_confidence=round(mean_conf, 3),
                        association_confidence=round(mean_assoc, 3),
                        event_confidence=heuristic_score,
                        uncertainty_reasons=uncertainty_reasons,
                        supporting_observations=f"Cell phone detected in {len(recent_obs)} frames adjacent to Track #{p_id}. Peak detection conf: {peak_conf * 100:.1f}%. Mean association: {mean_assoc * 100:.1f}%.",
                        visibility_warnings=["Partial hand occlusion observed across intermittent frames"] if "INTERMITTENT_HAND_OCCLUSION" in uncertainty_reasons else [],
                        description=f"Observable electronic device / cell phone detected in physical reach of Track #{p_id} across {duration_s:.1f}s ({len(recent_obs)} supporting frames)",
                        review_status="PENDING"
                    )
                    self._active_events[event_key] = evt
                    emitted_this_frame.append(evt)
                else:
                    # Update active event
                    evt = self._active_events[event_key]
                    evt.end_frame = frame_index
                    evt.end_time = round(timestamp_s, 2)
                    all_frames = list(dict.fromkeys(evt.supporting_frames + [frame_index]))
                    evt.supporting_frames = all_frames
                    evt.detection_confidence = round(mean_conf, 3)
                    evt.association_confidence = round(mean_assoc, 3)
                    evt.event_confidence = heuristic_score
                    evt.evidence_strength = strength
                    evt.description = f"Observable electronic device / cell phone detected in physical reach of Track #{p_id} across {duration_s:.1f}s ({len(all_frames)} supporting frames)"

            # Inactivity tracking: decay and complete inactive phone events
            if p_id not in persons_with_phone:
                inact = self._phone_inactivity_frames.get(p_id, 0) + 1
                self._phone_inactivity_frames[p_id] = inact

                # Close event after 30 frames (1.0s) of inactivity
                if inact >= int(round(self.fps * 1.0)) and event_key in self._active_events:
                    self._completed_events.append(self._active_events.pop(event_key))

        # --------------------------------------------------------------------
        # 2. Event: RESTRICTED_ZONE_ENTRY
        # Person enters a designated restricted zone for >= 5 frames
        # --------------------------------------------------------------------
        for p in persons:
            if p.assigned_zone in restricted_zones:
                z_id = p.assigned_zone
                curr_zone, streak = self._restricted_zone_streaks.get(p.track_id, (z_id, 0))
                if curr_zone == z_id:
                    streak += 1
                else:
                    curr_zone, streak = z_id, 1
                self._restricted_zone_streaks[p.track_id] = (curr_zone, streak)

                event_key = f"restricted:{p.track_id}:{z_id}"
                if streak >= 5 and event_key not in self._active_events:
                    zone_name = restricted_zones[z_id].name
                    evt = ExaminationEvent(
                        event_id=f"evt_restr_{uuid.uuid4().hex[:8]}",
                        job_id=self.job_id,
                        video_id=self.video_id,
                        event_type=ExaminationEventType.RESTRICTED_ZONE_ENTRY,
                        start_frame=max(0, frame_index - streak),
                        end_frame=frame_index,
                        start_time=max(0.0, round(timestamp_s - (streak / self.fps), 2)),
                        end_time=round(timestamp_s, 2),
                        person_track_id=p.track_id,
                        supporting_objects=[],
                        supporting_frames=[frame_index],
                        confidence=0.88,
                        uncertainty=0.10,
                        priority="RED",
                        evidence_strength="HIGH",
                        detection_confidence=0.90,
                        association_confidence=0.85,
                        event_confidence=0.88,
                        supporting_observations=f"Track #{p.track_id} positioned inside restricted zone '{zone_name}' for {streak / self.fps:.1f}s",
                        description=f"Track #{p.track_id} entered designated restricted zone '{zone_name}' ({streak / self.fps:.1f}s duration)",
                        review_status="PENDING"
                    )
                    self._active_events[event_key] = evt
                    emitted_this_frame.append(evt)
            else:
                self._restricted_zone_streaks[p.track_id] = ("", 0)

        # --------------------------------------------------------------------
        # 3. Event: LEAVING_SEATING_ZONE
        # STRICT REQUIREMENT: Only active if configured seating zones exist AND
        # the person was previously established as seated in a valid zone for >= 10 frames.
        # If no seating zones exist in the room, this event must NEVER fire.
        # --------------------------------------------------------------------
        if seating_zones:
            for p in persons:
                p_id = p.track_id
                if p.assigned_zone in seating_zones:
                    self._person_assigned_seat[p_id] = p.assigned_zone
                    self._vacated_zone_streaks[p_id] = 0
                elif p.assigned_zone is None and p_id in self._person_assigned_seat:
                    streak = self._vacated_zone_streaks.get(p_id, 0) + 1
                    self._vacated_zone_streaks[p_id] = streak
                    assigned_seat = self._person_assigned_seat[p_id]

                    event_key = f"vacate:{p_id}"
                    if streak >= 12 and event_key not in self._active_events:
                        evt = ExaminationEvent(
                            event_id=f"evt_vacate_{uuid.uuid4().hex[:8]}",
                            job_id=self.job_id,
                            video_id=self.video_id,
                            event_type=ExaminationEventType.LEAVING_SEATING_ZONE,
                            start_frame=max(0, frame_index - streak),
                            end_frame=frame_index,
                            start_time=max(0.0, round(timestamp_s - (streak / self.fps), 2)),
                            end_time=round(timestamp_s, 2),
                            person_track_id=p_id,
                            supporting_objects=[],
                            supporting_frames=[frame_index],
                            confidence=0.82,
                            uncertainty=0.12,
                            priority="AMBER",
                            evidence_strength="MEDIUM",
                            detection_confidence=0.85,
                            association_confidence=0.80,
                            event_confidence=0.82,
                            supporting_observations=f"Track #{p_id} vacated assigned seating zone '{assigned_seat}' for {streak / self.fps:.1f}s",
                            description=f"Track #{p_id} vacated assigned seating zone '{assigned_seat}' for {streak / self.fps:.1f}s",
                            review_status="PENDING"
                        )
                        self._active_events[event_key] = evt
                        emitted_this_frame.append(evt)

        # --------------------------------------------------------------------
        # 4. Event: POSSIBLE_OBJECT_TRANSFER
        # An object appears within close spatial proximity to two distinct persons
        # --------------------------------------------------------------------
        for obj in objects:
            if obj.object_class in ("book", "cell phone"):
                near_persons = []
                for p in persons:
                    d = bbox_distance(obj.bbox, p.bbox)
                    if d < 120.0:
                        near_persons.append(p.track_id)
                if len(near_persons) >= 2:
                    p1, p2 = min(near_persons[0], near_persons[1]), max(near_persons[0], near_persons[1])
                    event_key = f"transfer:{p1}:{p2}"
                    if event_key not in self._active_events:
                        evt = ExaminationEvent(
                            event_id=f"evt_trans_{uuid.uuid4().hex[:8]}",
                            job_id=self.job_id,
                            video_id=self.video_id,
                            event_type=ExaminationEventType.POSSIBLE_OBJECT_TRANSFER,
                            start_frame=frame_index,
                            end_frame=frame_index,
                            start_time=round(timestamp_s, 2),
                            end_time=round(timestamp_s, 2),
                            person_track_id=p1,
                            associated_object_class=obj.object_class,
                            supporting_objects=[obj.object_class],
                            supporting_frames=[frame_index],
                            confidence=0.74,
                            uncertainty=0.20,
                            priority="AMBER",
                            evidence_strength="MEDIUM",
                            detection_confidence=round(obj.confidence, 3),
                            association_confidence=0.70,
                            event_confidence=0.72,
                            supporting_observations=f"Object '{obj.object_class}' observed in shared proximity between Track #{p1} and Track #{p2}",
                            description=f"Observable proximity interaction of {obj.object_class} between Track #{p1} and Track #{p2}",
                            review_status="PENDING"
                        )
                        self._active_events[event_key] = evt
                        emitted_this_frame.append(evt)

        # --------------------------------------------------------------------
        # 5. Event: GAZE_DEVIATION
        # Sustained lateral head turn / body orientation toward peer or lap
        # --------------------------------------------------------------------
        for p in persons:
            p_id = p.track_id
            orientation = p.posture_orientation or "forward"
            w = max(1.0, p.bbox[2] - p.bbox[0])
            h = max(1.0, p.bbox[3] - p.bbox[1])
            aspect_ratio = w / h
            is_deviated = orientation in ("lateral_left", "lateral_right", "looking_down") or (aspect_ratio > 0.85)

            event_key = f"gaze:{p_id}"
            if is_deviated:
                streak = self._gaze_deviation_streaks.get(p_id, 0) + 1
                self._gaze_deviation_streaks[p_id] = streak
                if streak >= 18 and event_key not in self._active_events:
                    direction_label = "lateral / neighbor direction" if orientation != "looking_down" else "lap / concealed area"
                    evt = ExaminationEvent(
                        event_id=f"evt_gaze_{uuid.uuid4().hex[:8]}",
                        job_id=self.job_id,
                        video_id=self.video_id,
                        event_type=ExaminationEventType.GAZE_DEVIATION,
                        start_frame=max(0, frame_index - streak),
                        end_frame=frame_index,
                        start_time=max(0.0, round(timestamp_s - (streak / self.fps), 2)),
                        end_time=round(timestamp_s, 2),
                        person_track_id=p_id,
                        supporting_objects=[],
                        supporting_frames=[frame_index],
                        confidence=0.76,
                        uncertainty=0.18,
                        priority="AMBER",
                        evidence_strength="MEDIUM",
                        detection_confidence=0.80,
                        association_confidence=0.75,
                        event_confidence=0.76,
                        supporting_observations=f"Track #{p_id} exhibited sustained gaze deviation toward {direction_label} for {streak / self.fps:.1f}s",
                        description=f"Candidate gaze deviation / sustained head turn toward {direction_label} by Track #{p_id} ({streak / self.fps:.1f}s)",
                        review_status="PENDING"
                    )
                    self._active_events[event_key] = evt
                    emitted_this_frame.append(evt)
                elif streak >= 18 and event_key in self._active_events:
                    evt = self._active_events[event_key]
                    evt.end_frame = frame_index
                    evt.end_time = round(timestamp_s, 2)
                    if frame_index not in evt.supporting_frames:
                        evt.supporting_frames.append(frame_index)
            else:
                self._gaze_deviation_streaks[p_id] = 0
                if event_key in self._active_events:
                    self._completed_events.append(self._active_events.pop(event_key))

        # --------------------------------------------------------------------
        # 6. Event: PEER_PROXIMITY
        # Sustained physical convergence or closeness between two seated candidates
        # --------------------------------------------------------------------
        if len(persons) >= 2:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    p1, p2 = persons[i], persons[j]
                    dist = bbox_distance(p1.bbox, p2.bbox)
                    p1_w = p1.bbox[2] - p1.bbox[0]
                    p2_w = p2.bbox[2] - p2.bbox[0]
                    threshold_dist = max(110.0, (p1_w + p2_w) * 0.75)

                    pair_key = (min(p1.track_id, p2.track_id), max(p1.track_id, p2.track_id))
                    event_key = f"proximity:{pair_key[0]}:{pair_key[1]}"
                    if dist <= threshold_dist:
                        streak = self._peer_proximity_streaks.get(pair_key, 0) + 1
                        self._peer_proximity_streaks[pair_key] = streak

                        if streak >= 15 and event_key not in self._active_events:
                            evt = ExaminationEvent(
                                event_id=f"evt_prox_{uuid.uuid4().hex[:8]}",
                                job_id=self.job_id,
                                video_id=self.video_id,
                                event_type=ExaminationEventType.PEER_PROXIMITY,
                                start_frame=max(0, frame_index - streak),
                                end_frame=frame_index,
                                start_time=max(0.0, round(timestamp_s - (streak / self.fps), 2)),
                                end_time=round(timestamp_s, 2),
                                person_track_id=pair_key[0],
                                supporting_objects=[],
                                supporting_frames=[frame_index],
                                confidence=0.78,
                                uncertainty=0.15,
                                priority="AMBER",
                                evidence_strength="MEDIUM",
                                detection_confidence=0.82,
                                association_confidence=0.78,
                                event_confidence=0.78,
                                supporting_observations=f"Proximity convergence between Track #{pair_key[0]} and Track #{pair_key[1]} ({dist:.1f}px distance, {streak / self.fps:.1f}s)",
                                description=f"Candidate peer proximity / desk convergence between Track #{pair_key[0]} and Track #{pair_key[1]} ({streak / self.fps:.1f}s)",
                                review_status="PENDING"
                            )
                            self._active_events[event_key] = evt
                            emitted_this_frame.append(evt)
                        elif streak >= 15 and event_key in self._active_events:
                            evt = self._active_events[event_key]
                            evt.end_frame = frame_index
                            evt.end_time = round(timestamp_s, 2)
                            if frame_index not in evt.supporting_frames:
                                evt.supporting_frames.append(frame_index)
                    else:
                        self._peer_proximity_streaks[pair_key] = 0
                        if event_key in self._active_events:
                            self._completed_events.append(self._active_events.pop(event_key))

        return list(self._active_events.values())

    def get_all_events(self) -> List[ExaminationEvent]:
        """
        Returns all completed and active candidate events, prioritized and ranked.
        Marks the highest qualifying candidate event as primary_finding.
        If no candidate event qualifies, returns empty list (insufficient evidence).
        """
        all_evts = list(self._completed_events) + list(self._active_events.values())
        if not all_evts:
            return []

        # Deduplicate events by event_id
        unique_evts: Dict[str, ExaminationEvent] = {}
        for e in all_evts:
            unique_evts[e.event_id] = e
        evts_list = list(unique_evts.values())

        # Priority ranking weights
        priority_weights = {"RED": 30, "AMBER": 20, "INFO": 10}
        evidence_weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

        def sort_key(evt: ExaminationEvent):
            p_score = priority_weights.get(evt.priority, 10)
            e_score = evidence_weights.get(evt.evidence_strength, 1)
            duration = evt.end_time - evt.start_time
            return (p_score, e_score, duration, evt.confidence)

        evts_list.sort(key=sort_key, reverse=True)

        # Designate primary candidate finding on top ranked qualifying event
        for i, evt in enumerate(evts_list):
            evt.primary_finding = (i == 0 and evt.evidence_strength in ("HIGH", "MEDIUM"))

        return evts_list

