from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, Optional
import datetime

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True

class VideoResponse(BaseModel):
    id: int
    project_id: int
    filename: str
    storage_path: str
    sha256_hash: str
    duration_sec: float
    fps: float
    width: int
    height: int
    codec: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True

class CalibrationCreate(BaseModel):
    image_points: List[Tuple[float, float]]
    world_points: List[Tuple[float, float]]
    camera_height_m: float = 9.0
    pitch_deg: float = 14.0

class CalibrationResponse(BaseModel):
    id: int
    video_id: int
    version: int
    h_matrix_json: List[List[float]]
    image_points_json: List[Tuple[float, float]]
    world_points_json: List[Tuple[float, float]]
    reprojection_rmse_m: float
    camera_height_m: float
    pitch_deg: float
    created_at: datetime.datetime

    class Config:
        from_attributes = True

class JobCreate(BaseModel):
    calibration_id: Optional[int] = None
    config_json: Optional[Dict[str, Any]] = None

class JobResponse(BaseModel):
    id: int
    video_id: int
    calibration_id: Optional[int]
    status: str
    stage: str
    progress_pct: float
    error_message: Optional[str]
    telemetry: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True

class ViolationReviewUpdate(BaseModel):
    review_status: str  # ACCEPTED or REJECTED
    reviewer_notes: Optional[str] = None

class ViolationResponse(BaseModel):
    id: int
    job_id: int
    track_id: int
    vehicle_class: str
    frame_index: int
    timestamp: float
    estimated_speed_kmh: float
    speed_limit_kmh: float
    uncertainty_kmh: float
    location_label: str
    review_status: str
    reviewer_notes: Optional[str]
    full_frame_url: Optional[str] = None
    crop_frame_url: Optional[str] = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True
