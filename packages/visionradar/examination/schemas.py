from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class ExaminationEventType(str, Enum):
    POSSIBLE_PHONE_INTERACTION = "POSSIBLE_PHONE_INTERACTION"
    RESTRICTED_ZONE_ENTRY = "RESTRICTED_ZONE_ENTRY"
    LEAVING_SEATING_ZONE = "LEAVING_SEATING_ZONE"
    POSSIBLE_OBJECT_TRANSFER = "POSSIBLE_OBJECT_TRANSFER"
    REPEATED_ORIENTATION_PATTERN = "REPEATED_ORIENTATION_PATTERN"
    GAZE_DEVIATION = "GAZE_DEVIATION"
    PEER_PROXIMITY = "PEER_PROXIMITY"
    OTHER_OBSERVABLE_INTERACTION = "OTHER_OBSERVABLE_INTERACTION"

class ExaminationRole(str, Enum):
    STUDENT = "STUDENT"
    INVIGILATOR = "INVIGILATOR"
    EXAM_COORDINATOR = "EXAM_COORDINATOR"
    AUTHORIZED_STAFF = "AUTHORIZED_STAFF"
    UNKNOWN = "UNKNOWN"

class ZoneType(str, Enum):
    SEATING_ZONE = "SEATING_ZONE"
    RESTRICTED_ZONE = "RESTRICTED_ZONE"
    AISLE = "AISLE"

class SeatingZone(BaseModel):
    """Configured spatial zone within examination room."""
    zone_id: str = Field(..., description="Unique zone identifier (e.g., 'Seat-01')")
    name: str = Field(..., description="Descriptive zone label")
    zone_type: ZoneType = Field(default=ZoneType.SEATING_ZONE)
    polygon: List[List[float]] = Field(..., description="Normalized or pixel polygon [[x, y], ...]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "zone_type": self.zone_type.value,
            "polygon": self.polygon
        }

class PersonDetection(BaseModel):
    """Person detection & temporary track observation in examination scene."""
    track_id: int = Field(..., description="Temporary visual track identifier (not a biometric identity)")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0)
    assigned_role: str = Field(default="UNKNOWN", description="STUDENT | INVIGILATOR | EXAM_COORDINATOR | AUTHORIZED_STAFF | UNKNOWN")
    role_source: str = Field(default="DEFAULT", description="CONFIG | ZONE | HUMAN_REVIEW | DEFAULT")
    is_interpolated: bool = Field(default=False, description="Whether box was maintained through brief occlusion")
    size_category: Optional[str] = Field(default="medium", description="distant_small | medium | foreground")
    assigned_zone: Optional[str] = Field(default=None, description="Associated seating zone ID if within polygon")
    zone_confidence: Optional[float] = Field(default=None, description="Spatial membership confidence")
    posture_orientation: Optional[str] = Field(default=None, description="Coarse orientation: forward | lateral_left | lateral_right")
    associated_objects: List[Dict[str, Any]] = Field(default_factory=list, description="Associated object observations")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(self.confidence, 4),
            "assigned_role": self.assigned_role,
            "role_source": self.role_source,
            "is_interpolated": self.is_interpolated,
            "size_category": self.size_category,
            "assigned_zone": self.assigned_zone,
            "zone_confidence": round(self.zone_confidence, 4) if self.zone_confidence is not None else None,
            "posture_orientation": self.posture_orientation,
            "associated_objects": self.associated_objects
        }

class ExaminationObjectDetection(BaseModel):
    """Detected physical object relevant to examination observation."""
    object_id: int = Field(..., description="Object observation sequence ID")
    object_track_id: Optional[int] = Field(default=None, description="Persistent visual track ID for object")
    object_class: str = Field(..., description="Class: cell phone | paper | book | laptop | desk | candidate_electronic_device")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0)
    discrimination_status: str = Field(default="CONFIRMED", description="CONFIRMED | UNCERTAIN_ELECTRONIC_DEVICE")
    candidate_types: List[str] = Field(default_factory=list, description="Candidate categories e.g. ['cell phone', 'calculator', 'unknown']")
    near_person_track_id: Optional[int] = Field(default=None, description="Track ID of associated person")
    association_status: str = Field(default="UNASSOCIATED", description="INTERACTING | NEARBY | AMBIGUOUS | UNASSOCIATED")
    association_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    distance_to_person_px: Optional[float] = Field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_track_id": self.object_track_id,
            "object_class": self.object_class,
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(self.confidence, 4),
            "discrimination_status": self.discrimination_status,
            "candidate_types": self.candidate_types,
            "near_person_track_id": self.near_person_track_id,
            "association_status": self.association_status,
            "association_confidence": round(self.association_confidence, 4),
            "distance_to_person_px": round(self.distance_to_person_px, 1) if self.distance_to_person_px is not None else None
        }

class ExaminationEvent(BaseModel):
    """
    Observable candidate event detected across sequential multi-frame observations.
    All events represent neutral, verifiable visual candidate findings subject to human proctor review.
    Zero automated determinations of academic misconduct are ever emitted.
    """
    event_id: str = Field(..., description="Unique event identifier (UUID / slug)")
    job_id: int = Field(..., description="Associated processing job ID")
    video_id: int = Field(..., description="Associated video ID")
    event_type: ExaminationEventType = Field(...)
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds")
    person_track_id: Optional[int] = Field(default=None, description="Primary associated temporary track ID")
    assigned_role: str = Field(default="UNKNOWN", description="Assigned role of subject")
    associated_object_track_id: Optional[int] = Field(default=None, description="Associated object track ID")
    associated_object_class: Optional[str] = Field(default=None, description="Associated object category")
    supporting_objects: List[str] = Field(default_factory=list, description="Associated object classes detected")
    supporting_frames: List[int] = Field(default_factory=list, description="Keyframe indices containing evidence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Visual detection pattern confidence")
    uncertainty: float = Field(default=0.1, ge=0.0, le=1.0, description="Measurement / occlusion uncertainty")
    priority: str = Field(default="AMBER", description="Alert priority: RED (urgent review) | AMBER | INFO")
    evidence_strength: str = Field(default="MEDIUM", description="Evidence rating: HIGH | MEDIUM | LOW (heuristic score)")
    detection_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    association_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    event_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Explicit heuristic score")
    uncertainty_reasons: List[str] = Field(default_factory=list, description="Observed confounding factors or visibility limitations")
    primary_finding: bool = Field(default=False, description="Whether this event is the primary candidate finding for this session")
    supporting_observations: str = Field(default="", description="Empirical details (observation counts, peak conf, geometry)")
    visibility_warnings: List[str] = Field(default_factory=list, description="Visibility warnings (occlusion, blur, low light)")
    description: str = Field(..., description="Neutral factual description of observable visual activity")
    review_status: str = Field(default="PENDING", description="Review state: PENDING | ACCEPTED | REJECTED")
    reviewer_notes: Optional[str] = Field(default=None, description="Human proctor review notes")
    model_version: str = Field(default="exam-event-engine-v2.0")
    evidence_crop_path: Optional[str] = Field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "video_id": self.video_id,
            "event_type": self.event_type.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "person_track_id": self.person_track_id,
            "assigned_role": self.assigned_role,
            "associated_object_track_id": self.associated_object_track_id,
            "associated_object_class": self.associated_object_class,
            "supporting_objects": self.supporting_objects,
            "supporting_frames": self.supporting_frames,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "priority": self.priority,
            "evidence_strength": self.evidence_strength,
            "detection_confidence": round(self.detection_confidence, 4),
            "association_confidence": round(self.association_confidence, 4),
            "event_confidence": round(self.event_confidence, 4),
            "uncertainty_reasons": self.uncertainty_reasons,
            "primary_finding": self.primary_finding,
            "supporting_observations": self.supporting_observations,
            "visibility_warnings": self.visibility_warnings,
            "description": self.description,
            "review_status": self.review_status,
            "reviewer_notes": self.reviewer_notes,
            "model_version": self.model_version,
            "evidence_crop_path": self.evidence_crop_path
        }

class ExaminationFrameResult(BaseModel):
    """Full examination intelligence observations for a single video frame."""
    frame_index: int = Field(..., ge=0)
    timestamp_s: float = Field(..., ge=0.0)
    persons: List[PersonDetection] = Field(default_factory=list)
    objects: List[ExaminationObjectDetection] = Field(default_factory=list)
    active_events: List[ExaminationEvent] = Field(default_factory=list)
    zones: List[SeatingZone] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 2),
            "persons": [p.to_dict() for p in self.persons],
            "objects": [o.to_dict() for o in self.objects],
            "active_events": [e.to_dict() for e in self.active_events],
            "zones": [z.to_dict() for z in self.zones]
        }

