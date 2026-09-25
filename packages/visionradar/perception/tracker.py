"""
VisionRadar Perception Engine — ByteTrack Tracker Wrapper

Wraps ByteTrack multi-object tracking to associate per-frame VehicleDetection instances
into persistent VehicleTrack instances across frames.
"""

from typing import List, Dict, Optional
from visionradar.perception.schemas import VehicleDetection, VehicleTrack, BoundingBox, TrackState, VehicleClass
from visionradar.cv.tracker import ByteTrackTracker, Detection as LegacyDetection


class PerceptionTracker:
    """
    ByteTrack wrapper adapting VehicleDetection list to persistent VehicleTrack list.
    """
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.tracker = ByteTrackTracker(max_age=max_age, iou_threshold=iou_threshold)

    def update(
        self,
        detections: List[VehicleDetection],
        frame_id: int,
        capture_ts: float
    ) -> List[VehicleTrack]:
        """
        Updates ByteTrack association with current frame detections.

        Args:
            detections: List of VehicleDetection instances for frame_id
            frame_id: monotonic frame index
            capture_ts: perf_counter timestamp

        Returns:
            List[VehicleTrack]: active tracked vehicles
        """
        # Convert VehicleDetection to legacy Detection format expected by ByteTrackTracker
        legacy_dets = []
        det_map: Dict[int, VehicleDetection] = {}

        for idx, det in enumerate(detections):
            # Legacy format bbox: (x1, y1, x2, y2)
            legacy_dets.append(LegacyDetection(
                class_id=0,
                class_name=det.vehicle_class.value.capitalize(),
                confidence=det.confidence,
                bbox=det.bbox.to_tuple()
            ))
            det_map[idx] = det

        # Update ByteTrack tracker instance
        tracked_objs = self.tracker.update(legacy_dets, frame_id, capture_ts)

        # Convert ByteTrack active tracks into VehicleTrack schema
        tracks: List[VehicleTrack] = []
        for trk in tracked_objs:
            vclass_str = trk.vehicle_class.lower()
            try:
                vclass = VehicleClass(vclass_str)
            except ValueError:
                vclass = VehicleClass.CAR

            bx1, by1, bx2, by2 = trk.bbox

            # Estimate velocity in pixel space from trajectory points
            velocity = None
            if hasattr(trk, 'trajectory') and len(trk.trajectory.points) >= 2:
                p_curr = trk.trajectory.points[-1]
                p_prev = trk.trajectory.points[-2]
                df = max(1, p_curr.frame_index - p_prev.frame_index)
                dx = (p_curr.anchor_pixel[0] - p_prev.anchor_pixel[0]) / df
                dy = (p_curr.anchor_pixel[1] - p_prev.anchor_pixel[1]) / df
                velocity = (dx, dy)

            tracks.append(VehicleTrack(
                track_id=trk.track_id,
                state=TrackState.ACTIVE,
                bbox=BoundingBox(x1=float(bx1), y1=float(by1), x2=float(bx2), y2=float(by2)),
                confidence=float(trk.confidence),
                vehicle_class=vclass,
                first_seen_frame=trk.first_frame,
                last_seen_frame=trk.last_frame,
                frames_tracked=trk.hits,
                frames_since_last_detection=trk.time_since_update,
                velocity_px_per_frame=velocity
            ))

        return tracks
