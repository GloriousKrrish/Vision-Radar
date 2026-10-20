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
        warning_tolerance_kmh: float = 0.0,
        uncertainty_kmh: float = 0.0,
        location_label: str = "Lane 1",
        evidence_available: bool = True,
        evidence_reason: Optional[str] = None,
        job_id: Optional[int] = None,
        violation_type: str = "SPEED_LIMIT"
    ):
        self.violation_id = violation_id
        self.violation_type = violation_type
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.estimated_speed_kmh = estimated_speed_kmh
        self.speed_limit_kmh = speed_limit_kmh
        self.warning_tolerance_kmh = warning_tolerance_kmh
        self.effective_threshold_kmh = round(speed_limit_kmh + warning_tolerance_kmh, 2)
        self.amount_over_threshold_kmh = max(0.0, round(estimated_speed_kmh - self.effective_threshold_kmh, 1))
        self.uncertainty_kmh = uncertainty_kmh
        self.location_label = location_label
        self.review_status = "PENDING"  # PENDING, ACCEPTED, REJECTED
        if violation_type == "WRONG_WAY_DRIVER":
            self.reviewer_notes = f"CRITICAL: Vehicle moving counter to designated lane flow ({location_label})"
        else:
            self.reviewer_notes = f"Configured limit: {speed_limit_kmh:.1f} km/h, Tolerance: {warning_tolerance_kmh:.1f} km/h (+{self.amount_over_threshold_kmh:.1f} km/h over)"
        self.evidence_available = evidence_available
        self.evidence_reason = evidence_reason
        self.job_id = job_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type,
            "job_id": self.job_id,
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 2),
            "estimated_speed_kmh": round(self.estimated_speed_kmh, 1),
            "speed_limit_kmh": round(self.speed_limit_kmh, 1),
            "warning_tolerance_kmh": round(self.warning_tolerance_kmh, 1),
            "effective_threshold_kmh": round(self.effective_threshold_kmh, 1),
            "amount_over_threshold_kmh": round(self.amount_over_threshold_kmh, 1),
            "uncertainty_kmh": round(self.uncertainty_kmh, 1),
            "location_label": self.location_label,
            "review_status": self.review_status,
            "reviewer_notes": self.reviewer_notes,
            "evidence_available": self.evidence_available,
            "evidence_reason": self.evidence_reason
        }

class ViolationRuleEngine:
    """
    Evaluates vehicle speeds against speed limit rules and extracts candidate violations.
    Enforces metric calibration validity, effective threshold logic, and cooldown deduplication.
    """

    def __init__(
        self,
        speed_limit_kmh: float = 80.0,
        warning_tolerance_kmh: float = 0.0,
        cooldown_sec: float = 2.0
    ):
        if speed_limit_kmh <= 0:
            raise ValueError("Speed limit must be greater than zero.")
        if warning_tolerance_kmh < 0:
            raise ValueError("Warning tolerance must be non-negative.")
        
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.warning_tolerance_kmh = float(warning_tolerance_kmh)
        self.effective_threshold_kmh = round(self.speed_limit_kmh + self.warning_tolerance_kmh, 2)
        self.cooldown_sec = float(cooldown_sec)
        self.last_violation_timestamp: Dict[int, float] = {}

    def reset(self):
        """Reset deduplication state across jobs."""
        self.last_violation_timestamp.clear()

    def evaluate_track(
        self,
        track_id: int,
        vehicle_class: str,
        frame_index: int,
        timestamp: float,
        estimate: SpeedEstimate,
        location_label: str = "Lane 1",
        job_id: Optional[int] = None
    ) -> Optional[CandidateViolation]:
        # Condition 1: Strict reliability check - must be calibrated and VALID
        if not estimate or getattr(estimate, "validity", None) != "VALID":
            return None

        # Condition 2: Valid positive smoothed speed measurement
        smoothed = getattr(estimate, "smoothed_kmh", None)
        if smoothed is None or smoothed <= 0:
            return None

        # Condition 3: Must exceed effective threshold (speed_limit + tolerance)
        if smoothed <= self.effective_threshold_kmh:
            return None

        # Condition 4: Deduplicate repeated alerts for the same track using cooldown
        last_t = self.last_violation_timestamp.get(track_id)
        if last_t is not None and (timestamp - last_t) < self.cooldown_sec:
            return None

        self.last_violation_timestamp[track_id] = timestamp
        v_id = f"viol_t{track_id}_f{frame_index}"
        return CandidateViolation(
            violation_id=v_id,
            track_id=track_id,
            vehicle_class=vehicle_class,
            frame_index=frame_index,
            timestamp=timestamp,
            estimated_speed_kmh=smoothed,
            speed_limit_kmh=self.speed_limit_kmh,
            warning_tolerance_kmh=self.warning_tolerance_kmh,
            uncertainty_kmh=getattr(estimate, "uncertainty_kmh", 0.0),
            location_label=location_label,
            evidence_available=True,
            job_id=job_id
        )

    def evaluate_wrong_way(
        self,
        track_id: int,
        vehicle_class: str,
        frame_index: int,
        timestamp: float,
        trajectory: Any,
        expected_direction: str = "SOUTHWARD",
        location_label: str = "Lane 1",
        job_id: Optional[int] = None
    ) -> Optional[CandidateViolation]:
        """
        Detects vehicles moving counter to the designated roadway/lane flow direction.
        Requires sustained contrary trajectory displacement over >= 12 frames to eliminate jitter.
        """
        pts = getattr(trajectory, "points", [])
        if len(pts) < 12:
            return None

        # Check net vertical displacement over trajectory window
        first_y = pts[0].anchor_pixel[1]
        curr_y = pts[-1].anchor_pixel[1]
        delta_y = curr_y - first_y

        is_wrong_way = False
        if expected_direction == "SOUTHWARD" and delta_y < -35.0:
            is_wrong_way = True
        elif expected_direction == "NORTHWARD" and delta_y > 35.0:
            is_wrong_way = True

        if not is_wrong_way:
            return None

        last_t = self.last_violation_timestamp.get(track_id)
        if last_t is not None and (timestamp - last_t) < self.cooldown_sec:
            return None

        self.last_violation_timestamp[track_id] = timestamp
        v_id = f"wrongway_t{track_id}_f{frame_index}"
        return CandidateViolation(
            violation_id=v_id,
            track_id=track_id,
            vehicle_class=vehicle_class,
            frame_index=frame_index,
            timestamp=timestamp,
            estimated_speed_kmh=0.0,
            speed_limit_kmh=self.speed_limit_kmh,
            warning_tolerance_kmh=0.0,
            uncertainty_kmh=0.0,
            location_label=f"{location_label} (WRONG-WAY FLOW)",
            evidence_available=True,
            job_id=job_id,
            violation_type="WRONG_WAY_DRIVER"
        )

