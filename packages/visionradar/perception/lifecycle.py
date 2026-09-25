"""
VisionRadar Perception Engine — Object Lifecycle Manager

Consumes tracker output over time and manages track state transitions:
NEW -> ACTIVE -> LOST -> REMOVED per the exact lifecycle rules spec.
Emits ObjectLifecycleEvent transition records for auditing.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from visionradar.perception.schemas import (
    VehicleTrack, TrackState, ObjectLifecycleEvent
)

logger = logging.getLogger(__name__)


class ObjectLifecycleManager:
    """
    Manages state transitions for active vehicle tracks over time.

    Rules:
    - NEW -> ACTIVE: Promoted after detected in K consecutive frames (default K=2).
    - ACTIVE -> LOST: Transitioned on first missed detection.
    - LOST -> ACTIVE: Re-associated if detected within grace_window (default 15 frames).
    - LOST -> REMOVED: Retired permanently if missed > grace_window frames. Retired IDs are never reused.
    """
    def __init__(self, confirmation_frames: int = 2, grace_window: int = 15):
        self.confirmation_frames = confirmation_frames
        self.grace_window = grace_window
        self.track_states: Dict[int, TrackState] = {}
        self.consecutive_hits: Dict[int, int] = {}
        self.last_known_tracks: Dict[int, VehicleTrack] = {}
        self.retired_track_ids: Set[int] = set()
        self.lifecycle_events: List[ObjectLifecycleEvent] = []

    def is_id_retired(self, track_id: int) -> bool:
        """Returns True if track_id has been retired and cannot be reused."""
        return track_id in self.retired_track_ids

    def update(
        self,
        raw_tracks: List[VehicleTrack],
        frame_id: int
    ) -> Tuple[List[VehicleTrack], List[ObjectLifecycleEvent]]:
        """
        Processes current frame raw tracks from tracker and applies lifecycle rules.

        Returns:
            confirmed_tracks: list of active VehicleTrack objects (NEW, ACTIVE, or LOST within grace)
            events: list of ObjectLifecycleEvent state transition events emitted this frame
        """
        frame_events: List[ObjectLifecycleEvent] = []
        current_frame_ids = {t.track_id for t in raw_tracks}

        # 1. Update detected tracks
        for trk in raw_tracks:
            tid = trk.track_id
            assert tid not in self.retired_track_ids, f"Determinism failure: Track ID #{tid} was previously retired and reused!"

            self.last_known_tracks[tid] = trk

            if tid not in self.track_states:
                # Track newly created by tracker -> NEW
                self.track_states[tid] = TrackState.NEW
                self.consecutive_hits[tid] = 1

                event = ObjectLifecycleEvent(
                    track_id=tid,
                    frame_id=frame_id,
                    from_state=TrackState.NEW,
                    to_state=TrackState.NEW,
                    reason="first_detection_seen"
                )
                frame_events.append(event)

                if self.confirmation_frames <= 1:
                    self.track_states[tid] = TrackState.ACTIVE
                    event_active = ObjectLifecycleEvent(
                        track_id=tid,
                        frame_id=frame_id,
                        from_state=TrackState.NEW,
                        to_state=TrackState.ACTIVE,
                        reason=f"confirmed_{self.confirmation_frames}_consecutive_frames"
                    )
                    frame_events.append(event_active)

            else:
                curr_state = self.track_states[tid]
                if trk.frames_since_last_detection == 0:
                    self.consecutive_hits[tid] = self.consecutive_hits.get(tid, 0) + 1

                    if curr_state == TrackState.NEW:
                        if self.consecutive_hits[tid] >= self.confirmation_frames:
                            self.track_states[tid] = TrackState.ACTIVE
                            event = ObjectLifecycleEvent(
                                track_id=tid,
                                frame_id=frame_id,
                                from_state=TrackState.NEW,
                                to_state=TrackState.ACTIVE,
                                reason=f"confirmed_{self.confirmation_frames}_consecutive_frames"
                            )
                            frame_events.append(event)

                    elif curr_state == TrackState.LOST:
                        # Reassociated within grace window
                        self.track_states[tid] = TrackState.ACTIVE
                        event = ObjectLifecycleEvent(
                            track_id=tid,
                            frame_id=frame_id,
                            from_state=TrackState.LOST,
                            to_state=TrackState.ACTIVE,
                            reason="reassociated_within_grace_window"
                        )
                        frame_events.append(event)
                else:
                    self.consecutive_hits[tid] = 0

            trk.state = self.track_states[tid]

        # 2. Process missed tracks (ACTIVE/NEW -> LOST, LOST -> REMOVED)
        known_tids = list(self.track_states.keys())
        for tid in known_tids:
            if tid not in current_frame_ids:
                curr_state = self.track_states[tid]
                last_trk = self.last_known_tracks.get(tid)

                if last_trk is not None:
                    last_trk.frames_since_last_detection += 1
                    missed_count = last_trk.frames_since_last_detection

                    if missed_count > self.grace_window:
                        # Exceeded grace window -> REMOVED & retire track_id
                        self.track_states[tid] = TrackState.REMOVED
                        self.retired_track_ids.add(tid)
                        event = ObjectLifecycleEvent(
                            track_id=tid,
                            frame_id=frame_id,
                            from_state=curr_state,
                            to_state=TrackState.REMOVED,
                            reason=f"exceeded_grace_window_{self.grace_window}_frames"
                        )
                        frame_events.append(event)
                        del self.track_states[tid]
                        del self.last_known_tracks[tid]

                    elif curr_state in (TrackState.ACTIVE, TrackState.NEW):
                        self.track_states[tid] = TrackState.LOST
                        last_trk.state = TrackState.LOST
                        event = ObjectLifecycleEvent(
                            track_id=tid,
                            frame_id=frame_id,
                            from_state=curr_state,
                            to_state=TrackState.LOST,
                            reason="no_detection_missed_frame"
                        )
                        frame_events.append(event)

        # 3. Assemble active output tracks
        confirmed_output_tracks: List[VehicleTrack] = [
            trk for trk in self.last_known_tracks.values()
            if trk.state in (TrackState.ACTIVE, TrackState.NEW, TrackState.LOST)
        ]

        self.lifecycle_events.extend(frame_events)
        return confirmed_output_tracks, frame_events
