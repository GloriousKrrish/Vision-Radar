from typing import List, Dict, Any, Optional
import uuid
from visionradar.intelligence.events.base import BaseEventDetector, EventCandidate
from visionradar.intelligence.events.wrong_way import WrongWayDetector
from visionradar.intelligence.events.stopped_vehicle import StoppedVehicleDetector
from visionradar.intelligence.events.deceleration import SuddenDecelerationDetector

class TrackEventState:
    """
    Manages temporal state machine (IDLE -> ACTIVE -> COOLDOWN -> IDLE) for a single track and event type.
    Enforces trigger hysteresis, release thresholding, minimum event duration, and cooldown deduplication.
    """
    def __init__(
        self,
        track_id: int,
        event_type: str,
        release_threshold_frames: int = 5,
        min_duration_frames: int = 5,
        min_cooldown_frames: int = 30,
        min_confidence: float = 0.70
    ):
        self.track_id = track_id
        self.event_type = event_type
        self.release_threshold_frames = release_threshold_frames
        self.min_duration_frames = min_duration_frames
        self.min_cooldown_frames = min_cooldown_frames
        self.min_confidence = min_confidence

        self.state = "IDLE"  # IDLE, ACTIVE, COOLDOWN
        self.start_frame = 0
        self.start_timestamp = 0.0
        self.peak_frame = 0
        self.peak_timestamp = 0.0
        self.peak_metric_value = 0.0
        self.end_frame = 0
        self.end_timestamp = 0.0
        self.evidence_frames: List[Dict[str, Any]] = []
        self.release_counter = 0
        self.cooldown_counter = 0
        self.latest_trigger: Optional[Dict[str, Any]] = None

    def update(
        self,
        has_trigger: bool,
        trigger_info: Optional[Dict[str, Any]],
        frame_idx: int,
        timestamp: float
    ) -> Optional[EventCandidate]:
        finalized_event = None

        if self.state == "IDLE":
            if has_trigger and trigger_info:
                self.state = "ACTIVE"
                self.start_frame = frame_idx
                self.start_timestamp = timestamp
                self.peak_frame = frame_idx
                self.peak_timestamp = timestamp
                self.peak_metric_value = trigger_info.get("metric_value", 0.0)
                self.release_counter = 0
                self.evidence_frames = [trigger_info.get("evidence", {})]
                self.latest_trigger = trigger_info

        elif self.state == "ACTIVE":
            if has_trigger and trigger_info:
                self.release_counter = 0
                self.evidence_frames.append(trigger_info.get("evidence", {}))
                self.latest_trigger = trigger_info
                metric_val = trigger_info.get("metric_value", 0.0)
                if metric_val > self.peak_metric_value:
                    self.peak_metric_value = metric_val
                    self.peak_frame = frame_idx
                    self.peak_timestamp = timestamp
            else:
                self.release_counter += 1
                if self.release_counter >= self.release_threshold_frames:
                    # Finalize Active Event
                    self.end_frame = max(self.start_frame, frame_idx - self.release_counter)
                    self.end_timestamp = timestamp
                    duration_frames = self.end_frame - self.start_frame + 1
                    duration_sec = max(0.1, self.end_timestamp - self.start_timestamp)

                    conf = self.latest_trigger.get("confidence", 0.85) if self.latest_trigger else 0.85

                    if duration_frames >= self.min_duration_frames and conf >= self.min_confidence and self.latest_trigger:
                        ev_id = f"evt_{self.event_type}_trk{self.track_id}_f{self.start_frame}-{self.end_frame}"
                        finalized_event = EventCandidate(
                            event_id=ev_id,
                            event_type=self.event_type,
                            track_id=self.track_id,
                            severity=self.latest_trigger.get("severity", "MEDIUM"),
                            confidence=conf,
                            start_frame=self.start_frame,
                            peak_frame=self.peak_frame,
                            end_frame=self.end_frame,
                            start_time=self.start_timestamp,
                            peak_time=self.peak_timestamp,
                            end_time=self.end_timestamp,
                            duration=duration_sec,
                            explanation=self.latest_trigger.get("explanation", ""),
                            evidence_frames=self.evidence_frames,
                            status="FINALIZED",
                            direction_vector=self.latest_trigger.get("direction_vector"),
                            expected_lane_direction=self.latest_trigger.get("expected_lane_direction"),
                            observed_direction=self.latest_trigger.get("observed_direction"),
                            direction_confidence=self.latest_trigger.get("direction_confidence")
                        )

                    self.state = "COOLDOWN"
                    self.cooldown_counter = self.min_cooldown_frames

        elif self.state == "COOLDOWN":
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.state = "IDLE"

        return finalized_event

    def force_finalize(self, frame_idx: int, timestamp: float) -> Optional[EventCandidate]:
        if self.state == "ACTIVE" and self.latest_trigger:
            self.end_frame = frame_idx
            self.end_timestamp = timestamp
            duration_frames = self.end_frame - self.start_frame + 1
            duration_sec = max(0.1, self.end_timestamp - self.start_timestamp)
            conf = self.latest_trigger.get("confidence", 0.85)

            if duration_frames >= self.min_duration_frames and conf >= self.min_confidence:
                ev_id = f"evt_{self.event_type}_trk{self.track_id}_f{self.start_frame}-{self.end_frame}"
                event = EventCandidate(
                    event_id=ev_id,
                    event_type=self.event_type,
                    track_id=self.track_id,
                    severity=self.latest_trigger.get("severity", "MEDIUM"),
                    confidence=conf,
                    start_frame=self.start_frame,
                    peak_frame=self.peak_frame,
                    end_frame=self.end_frame,
                    start_time=self.start_timestamp,
                    peak_time=self.peak_timestamp,
                    end_time=self.end_timestamp,
                    duration=duration_sec,
                    explanation=self.latest_trigger.get("explanation", ""),
                    evidence_frames=self.evidence_frames,
                    status="FINALIZED",
                    direction_vector=self.latest_trigger.get("direction_vector"),
                    expected_lane_direction=self.latest_trigger.get("expected_lane_direction"),
                    observed_direction=self.latest_trigger.get("observed_direction"),
                    direction_confidence=self.latest_trigger.get("direction_confidence")
                )
                self.state = "COOLDOWN"
                return event
        self.state = "IDLE"
        return None


class TrafficEventEngine:
    """
    Executes modular event detectors, drives temporal hysteresis state machines, and yields deduplicated physical events.
    """
    def __init__(self, detectors: List[BaseEventDetector] = None):
        self.detectors = detectors or [
            WrongWayDetector(),
            StoppedVehicleDetector(),
            SuddenDecelerationDetector()
        ]
        self.state_machines: Dict[str, TrackEventState] = {}
        self.finalized_events: List[EventCandidate] = []
        self.candidate_trigger_count = 0

    def process_frame(self, active_tracks: List[Dict[str, Any]], frame_index: int, timestamp: float) -> List[EventCandidate]:
        frame_triggers = []
        for det in self.detectors:
            if hasattr(det, "detect_frame_triggers"):
                trigs = det.detect_frame_triggers(active_tracks, frame_index, timestamp)
            else:
                trigs = det.detect_events(active_tracks, frame_index, timestamp)
            frame_triggers.extend(trigs)

        self.candidate_trigger_count += len(frame_triggers)
        newly_finalized = []

        # Index frame triggers by (track_id, event_type)
        trig_map = {(t["track_id"], t["event_type"]): t for t in frame_triggers if isinstance(t, dict)}

        # Update state machines for all active tracks
        active_track_ids = {trk["track_id"] for trk in active_tracks}

        for trk in active_tracks:
            tid = trk["track_id"]
            for etype in ["sudden_deceleration", "wrong_way", "stopped_vehicle"]:
                key = f"{tid}_{etype}"
                if key not in self.state_machines:
                    self.state_machines[key] = TrackEventState(track_id=tid, event_type=etype)

                sm = self.state_machines[key]
                trig_info = trig_map.get((tid, etype))
                has_trig = trig_info is not None

                final_ev = sm.update(has_trig, trig_info, frame_index, timestamp)
                if final_ev:
                    self.finalized_events.append(final_ev)
                    newly_finalized.append(final_ev)

        return newly_finalized

    def finalize(self, last_frame_index: int, last_timestamp: float):
        """
        Finalizes any remaining active events at end of video stream.
        """
        for sm in self.state_machines.values():
            ev = sm.force_finalize(last_frame_index, last_timestamp)
            if ev:
                self.finalized_events.append(ev)

    def get_events_summary(self) -> Dict[str, Any]:
        tot = len(self.finalized_events) + self.candidate_trigger_count
        return {
            "total_event_candidates": tot,
            "total_raw_candidate_triggers": self.candidate_trigger_count,
            "total_finalized_events": len(self.finalized_events),
            "unreviewed_count": len([e for e in self.finalized_events if e.review_status == "UNREVIEWED"]),
            "accepted_count": len([e for e in self.finalized_events if e.review_status == "ACCEPTED"]),
            "rejected_count": len([e for e in self.finalized_events if e.review_status == "REJECTED"]),
            "events_by_type": {
                "sudden_deceleration": len([e for e in self.finalized_events if e.event_type == "sudden_deceleration"]),
                "wrong_way": len([e for e in self.finalized_events if e.event_type == "wrong_way"]),
                "stopped_vehicle": len([e for e in self.finalized_events if e.event_type == "stopped_vehicle"])
            },
            "events": [e.to_dict() for e in self.finalized_events]
        }

