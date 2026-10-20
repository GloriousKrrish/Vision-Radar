from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class CommonEnvelope(BaseModel):
    """
    Standardized common envelope for VisionRadar multi-domain results (Layer D).
    Encapsulates results from any analysis module (traffic, examination, etc.)
    under a stable, versioned contract.
    """
    schema_version: str = Field(default="2.0", description="Envelope contract version")
    job_id: int = Field(..., description="ID of the processing job")
    video_id: int = Field(..., description="ID of the source video")
    analysis_module: str = Field(..., description="Analysis domain: traffic | examination")
    source_timestamp: float = Field(default=0.0, description="Video playback timestamp in seconds")
    frame_index: int = Field(default=0, description="Source frame index")
    result_type: str = Field(..., description="Type of result: frame_observation | job_summary | event_alert")
    processing_status: str = Field(default="SUCCESS", description="Status: QUEUED | RUNNING | SUCCESS | FAILED")
    confidence: Optional[float] = Field(default=None, description="Aggregate confidence or uncertainty indicator")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Module-specific structured result payload")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "video_id": self.video_id,
            "analysis_module": self.analysis_module,
            "source_timestamp": self.source_timestamp,
            "frame_index": self.frame_index,
            "result_type": self.result_type,
            "processing_status": self.processing_status,
            "confidence": self.confidence,
            "payload": self.payload
        }
