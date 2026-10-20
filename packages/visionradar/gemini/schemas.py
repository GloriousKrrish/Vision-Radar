"""
Pydantic schemas for the Gemini Multimodal Intelligence Service.
Defines structured contracts for candidate event interpretation, scene summaries,
and provenance tracking.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class GeminiCandidateEventInterpretation(BaseModel):
    """Structured multimodal interpretation of a candidate examination event."""
    event_id: str = Field(..., description="ID matching the local CV candidate event")
    event_type: str = Field(..., description="Event type name (e.g. POSSIBLE_PHONE_INTERACTION)")
    observed_behaviour: str = Field(..., description="Neutral factual description of observed physical action")
    contextual_assessment: str = Field(..., description="Contextual evaluation of person-object interaction")
    supporting_frame_references: List[int] = Field(default_factory=list, description="Frame indices supporting this interpretation")
    uncertainty_notes: List[str] = Field(default_factory=list, description="Visibility limitations or ambiguity reasons")
    visibility_warnings: List[str] = Field(default_factory=list, description="Partial occlusion, blur, or distance warnings")
    recommended_review_priority: str = Field("AMBER", description="RED for urgent review, AMBER for routine, INFO for informational")
    heuristic_relevance_score: float = Field(0.5, ge=0.0, le=1.0, description="Heuristic relevance score (not a probability)")

class GeminiExamAnalysisOutput(BaseModel):
    """Complete structured response for examination video analysis."""
    scene_summary: str = Field(..., description="High-level factual summary of the visible examination scene")
    environment_type: str = Field("CLASSROOM_EXAMINATION", description="Detected physical setting")
    candidate_findings: List[GeminiCandidateEventInterpretation] = Field(default_factory=list)
    contradictory_evidence_notes: List[str] = Field(default_factory=list, description="Observations that weaken or contradict misconduct hypotheses")
    primary_finding_event_id: Optional[str] = Field(None, description="Event ID of the highest priority candidate finding")
    proctor_action_recommendation: str = Field("HUMAN_REVIEW_RECOMMENDED", description="Suggested proctor review action")
    model_identifier: str = Field(..., description="Name of the Gemini model used for inference")

class GeminiTrafficSummaryOutput(BaseModel):
    """Structured response for multimodal traffic scene analysis."""
    scene_summary: str = Field(..., description="Overview of road environment, lanes, and weather conditions")
    traffic_flow_condition: str = Field("MODERATE", description="FREE_FLOW, MODERATE, CONGESTED, or STOP_AND_GO")
    incident_or_congestion_observations: List[str] = Field(default_factory=list)
    speed_disclaimer: str = Field(
        "Vehicle speed metrics are produced exclusively by VisionRadar's local homography calibration and ByteTrack pipeline. Gemini does not calculate or estimate speed.",
        description="Mandatory preservation disclaimer"
    )
    uncertainty_notes: List[str] = Field(default_factory=list)
    model_identifier: str = Field(..., description="Name of the Gemini model used for inference")

class GeminiStatusResponse(BaseModel):
    """Safe status response for frontend consumption."""
    enabled: bool
    configured: bool
    status: str  # READY, KEY_MISSING, DISABLED, ERROR
    model: str
    max_selected_frames: int
    timeout_sec: float
    detail: str

class GeminiAnalysisRequestPayload(BaseModel):
    """Payload to trigger Gemini contextual analysis."""
    model_override: Optional[str] = None
    force_refresh: bool = False
