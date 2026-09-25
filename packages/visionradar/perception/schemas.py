"""
VisionRadar Real-Time Perception Engine Schemas

Defines typed dataclasses and enums for raw detections, persistent vehicle tracks,
object lifecycle states, frame results, stage latency statistics, and perception metrics.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


class VehicleClass(str, Enum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"


class TrackState(str, Enum):
    NEW = "new"           # First frame(s) seen, unconfirmed
    ACTIVE = "active"      # Confirmed active track
    LOST = "lost"            # Missed detection this frame, within grace window
    REMOVED = "removed"        # Exceeded grace window, track terminated


@dataclass(frozen=True)
class BoundingBox:
    x1: float  # Pixel coords, top-left origin
    y1: float
    x2: float
    y2: float

    def to_tuple(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class VehicleDetection:
    """Raw per-frame detection output, pre-tracking. Immutable."""
    detection_id: str          # UUID or unique identifier per detection instance
    frame_id: int
    bbox: BoundingBox
    confidence: float           # 0.0 - 1.0, raw detector confidence
    vehicle_class: VehicleClass
    class_confidence: float       # 0.0 - 1.0, classification confidence

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "frame_id": self.frame_id,
            "bbox": self.bbox.to_dict(),
            "confidence": round(self.confidence, 4),
            "vehicle_class": self.vehicle_class.value,
            "class_confidence": round(self.class_confidence, 4)
        }


@dataclass
class VehicleTrack:
    """Tracked vehicle, persists across frames. Mutable, owned by tracker/lifecycle."""
    track_id: int                  # Stable across lifetime, assigned by ByteTrack
    state: TrackState
    bbox: BoundingBox               # Current frame's bbox
    confidence: float                # Current frame's association confidence
    vehicle_class: VehicleClass
    first_seen_frame: int
    last_seen_frame: int
    frames_tracked: int                # Count of frames where state was ACTIVE
    frames_since_last_detection: int     # Resets to 0 on ACTIVE frame; drives NEW/LOST/REMOVED
    velocity_px_per_frame: Optional[tuple] = None  # (dx, dy) in pixel space

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "bbox": self.bbox.to_dict(),
            "confidence": round(self.confidence, 4),
            "vehicle_class": self.vehicle_class.value,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "frames_tracked": self.frames_tracked,
            "frames_since_last_detection": self.frames_since_last_detection,
            "velocity_px_per_frame": [round(v, 2) for v in self.velocity_px_per_frame] if self.velocity_px_per_frame else None
        }


@dataclass
class ObjectLifecycleEvent:
    """Emitted on state transitions for logging and debugging."""
    track_id: int
    frame_id: int
    from_state: TrackState
    to_state: TrackState
    reason: str

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason
        }


@dataclass
class FrameResult:
    """The unit sent to the frontend per processed frame."""
    frame_id: int
    capture_ts: float             # perf_counter seconds at capture
    result_ts: float                # perf_counter seconds at assembly
    detections: List[VehicleDetection] = field(default_factory=list)
    tracks: List[VehicleTrack] = field(default_factory=list)
    degraded: bool = False           # True if any stage was skipped/failed
    degraded_reason: Optional[str] = None
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "capture_ts": self.capture_ts,
            "result_ts": self.result_ts,
            "server_pipeline_latency_ms": round((self.result_ts - self.capture_ts) * 1000.0, 2),
            "detections": [d.to_dict() for d in self.detections],
            "tracks": [t.to_dict() for t in self.tracks],
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "stage_timings_ms": {k: round(v, 2) for k, v in self.stage_timings_ms.items()}
        }


@dataclass
class StageLatency:
    p50: float
    p95: float
    p99: float
    max: float

    def to_dict(self) -> dict:
        return {
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "max": round(self.max, 2)
        }


@dataclass
class PerceptionMetrics:
    window: str  # "last_100" | "last_1000"
    fps_capture: float
    fps_processed: float
    stage_latency_ms: Dict[str, StageLatency]
    frames_processed: int
    frames_dropped: int
    drop_reasons: Dict[str, int]
    active_track_count: int
    first_box_latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "fps_capture": round(self.fps_capture, 2),
            "fps_processed": round(self.fps_processed, 2),
            "stage_latency_ms": {k: v.to_dict() for k, v in self.stage_latency_ms.items()},
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "drop_reasons": self.drop_reasons,
            "active_track_count": self.active_track_count,
            "first_box_latency_ms": round(self.first_box_latency_ms, 2) if self.first_box_latency_ms else None
        }
