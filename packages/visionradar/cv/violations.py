from typing import List, Dict, Any, Optional
from visionradar.cv.speed import SpeedEstimate

class CandidateViolation:
    """
    Container for candidate speed limit violation events awaiting human review.
    """
    def __init__(
        self,
        violation_id: str,
        track_id: int,
        vehicle_class: str,
        frame_index: int,
        timestamp: float,
        estimated_speed_kmh: float,
        speed_limit_kmh: float,
        uncertainty_kmh: float,
        location_label: str = "Lane 1"
    ):
        self.violation_id = violation_id
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.estimated_speed_kmh = estimated_speed_kmh
        self.speed_limit_kmh = speed_limit_kmh
        self.uncertainty_kmh = uncertainty_kmh
        self.location_label = location_label
        self.review_status = "PENDING"  # PENDING, ACCEPTED, REJECTED
        self.reviewer_notes = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 2),
            "estimated_speed_kmh": round(self.estimated_speed_kmh, 1),
            "speed_limit_kmh": round(self.speed_limit_kmh, 1),
            "uncertainty_kmh": round(self.uncertainty_kmh, 1),
            "location_label": self.location_label,
            "review_status": self.review_status,
            "reviewer_notes": self.reviewer_notes
        }

class ViolationRuleEngine:
    """
    Evaluates vehicle speeds against speed limit rules and extracts candidate violations.
    """

    def __init__(self, speed_limit_kmh: float = 80.0):
        self.speed_limit_kmh = speed_limit_kmh

    def evaluate_track(
        self,
        track_id: int,
        vehicle_class: str,
        frame_index: int,
        timestamp: float,
        estimate: SpeedEstimate,
        location_label: str = "Lane 1"
    ) -> Optional[CandidateViolation]:
        # Check if estimated speed minus uncertainty exceeds the limit
        if estimate.smoothed_kmh > self.speed_limit_kmh:
            v_id = f"viol_t{track_id}_f{frame_index}"
            return CandidateViolation(
                violation_id=v_id,
                track_id=track_id,
                vehicle_class=vehicle_class,
                frame_index=frame_index,
                timestamp=timestamp,
                estimated_speed_kmh=estimate.smoothed_kmh,
                speed_limit_kmh=self.speed_limit_kmh,
                uncertainty_kmh=estimate.uncertainty_kmh,
                location_label=location_label
            )
        return None
