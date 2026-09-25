from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time

class EventCandidate:
    """
    Structured Traffic Event adhering to Stage K-P schema.
    Represents a finalized or active temporal physical traffic event.
    """
    def __init__(
        self,
        event_id: str,
        event_type: str,
        track_id: int,
        severity: str = "MEDIUM",
        confidence: float = 0.85,
        start_frame: int = 0,
        peak_frame: int = 0,
        end_frame: int = 0,
        start_time: float = 0.0,
        peak_time: float = 0.0,
        end_time: float = 0.0,
        duration: float = 0.0,
        explanation: str = "",
        evidence_frames: Optional[List[Dict[str, Any]]] = None,
        status: str = "FINALIZED",
        model_version: str = "v1.0.0",
        rule_version: str = "v1.0.0",
        direction_vector: Optional[List[float]] = None,
        expected_lane_direction: Optional[str] = None,
        observed_direction: Optional[str] = None,
        direction_confidence: Optional[float] = None
    ):
        self.event_id = event_id
        self.event_type = event_type
        self.track_id = track_id
        self.severity = severity
        self.confidence = float(confidence)
        self.start_frame = start_frame
        self.peak_frame = peak_frame
        self.end_frame = end_frame
        self.start_time = start_time
        self.peak_time = peak_time or start_time
        self.end_time = end_time or start_time
        self.duration = duration or max(0.0, self.end_time - self.start_time)
        self.explanation = explanation
        self.evidence_frames = evidence_frames or []
        self.status = status
        self.model_version = model_version
        self.rule_version = rule_version
        self.review_status = "UNREVIEWED"
        self.reviewer = None
        self.review_timestamp = None
        self.review_reason = None
        
        # Wrong-way specific temporal metadata
        self.direction_vector = direction_vector
        self.expected_lane_direction = expected_lane_direction
        self.observed_direction = observed_direction
        self.direction_confidence = direction_confidence

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "track_id": self.track_id,
            "affected_tracks": [self.track_id],
            "severity": self.severity,
            "severity_metric": self.severity,
            "confidence": round(self.confidence, 4),
            "start_frame": self.start_frame,
            "peak_frame": self.peak_frame,
            "end_frame": self.end_frame,
            "start_timestamp": round(self.start_time, 2),
            "peak_timestamp": round(self.peak_time, 2),
            "end_timestamp": round(self.end_time, 2),
            "duration_s": round(self.duration, 2),
            "explanation": self.explanation,
            "evidence_frames": self.evidence_frames,
            "status": self.status,
            "model_version": self.model_version,
            "rule_version": self.rule_version,
            "review_status": self.review_status
        }
        if self.direction_vector is not None:
            res["direction_vector"] = self.direction_vector
            res["expected_lane_direction"] = self.expected_lane_direction
            res["observed_direction"] = self.observed_direction
            res["direction_confidence"] = self.direction_confidence
        return res


class BaseEventDetector(ABC):
    @abstractmethod
    def detect_frame_triggers(self, active_tracks: List[Dict[str, Any]], frame_index: int, timestamp: float) -> List[Dict[str, Any]]:
        """
        Evaluates per-frame trigger candidates without emitting duplicate events.
        """
        pass

