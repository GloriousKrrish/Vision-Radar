from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class SceneType(str, Enum):
    TRAFFIC = "TRAFFIC"
    CLASSROOM_EXAMINATION = "CLASSROOM_EXAMINATION"
    OTHER_UNKNOWN = "OTHER_UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"

class AnalysisModule(str, Enum):
    TRAFFIC = "traffic"
    EXAMINATION = "examination"
    NONE = "none"

class SceneClassificationResult(BaseModel):
    """
    Standard structured output for automatic video scene classification.
    Returned by the Scene Router to recommend the appropriate processing pipeline.
    """
    scene_type: SceneType = Field(..., description="Estimated category: TRAFFIC | CLASSROOM_EXAMINATION | OTHER_UNKNOWN | AMBIGUOUS")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated classification confidence (0.0 to 1.0)")
    recommended_module: AnalysisModule = Field(..., description="Recommended downstream pipeline: traffic | examination | none")
    requires_user_confirmation: bool = Field(default=False, description="True if classification is ambiguous or confidence is low")
    reason: str = Field(..., description="Neutral factual explanation of visual cues detected")
    model_version: str = Field(default="scene-router-v1.0", description="Identifier of classifier model and heuristic rules")
    sampled_frames_count: int = Field(default=0, description="Number of representative sampled frames analyzed")
    cue_scores: Dict[str, float] = Field(default_factory=dict, description="Normalized visual cue scores (roadway, classroom, vehicles, desks)")
    video_path: Optional[str] = Field(default=None, description="Path to analyzed video file")
    sha256_hash: Optional[str] = Field(default=None, description="SHA256 hash of the video file")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_type": self.scene_type.value,
            "confidence": round(self.confidence, 4),
            "recommended_module": self.recommended_module.value,
            "requires_user_confirmation": self.requires_user_confirmation,
            "reason": self.reason,
            "model_version": self.model_version,
            "sampled_frames_count": self.sampled_frames_count,
            "cue_scores": {k: round(v, 4) for k, v in self.cue_scores.items()},
            "video_path": self.video_path,
            "sha256_hash": self.sha256_hash
        }
