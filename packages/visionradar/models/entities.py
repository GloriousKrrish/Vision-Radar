import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from visionradar.models.database import Base

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    videos = relationship("Video", back_populates="project", cascade="all, delete-orphan")

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)
    sha256_hash = Column(String(64), nullable=False)
    duration_sec = Column(Float, nullable=False, default=0.0)
    fps = Column(Float, nullable=False, default=30.0)
    width = Column(Integer, nullable=False, default=1280)
    height = Column(Integer, nullable=False, default=720)
    codec = Column(String(32), nullable=True, default="h264")
    created_at = Column(DateTime, default=utcnow)

    project = relationship("Project", back_populates="videos")
    calibrations = relationship("Calibration", back_populates="video", cascade="all, delete-orphan")
    jobs = relationship("ProcessingJob", back_populates="video", cascade="all, delete-orphan")

class Calibration(Base):
    __tablename__ = "calibrations"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    h_matrix_json = Column(JSON, nullable=False)
    image_points_json = Column(JSON, nullable=False)
    world_points_json = Column(JSON, nullable=False)
    reprojection_rmse_m = Column(Float, nullable=False, default=0.0)
    camera_height_m = Column(Float, nullable=False, default=9.0)
    pitch_deg = Column(Float, nullable=False, default=14.0)
    created_at = Column(DateTime, default=utcnow)

    video = relationship("Video", back_populates="calibrations")
    jobs = relationship("ProcessingJob", back_populates="calibration")

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    calibration_id = Column(Integer, ForeignKey("calibrations.id"), nullable=True)
    status = Column(String(32), nullable=False, default="QUEUED")  # QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED
    stage = Column(String(64), nullable=False, default="Decoder")
    progress_pct = Column(Float, nullable=False, default=0.0)
    config_json = Column(JSON, nullable=True)
    telemetry_json = Column(JSON, nullable=True)
    git_hash = Column(String(40), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    video = relationship("Video", back_populates="jobs")
    calibration = relationship("Calibration", back_populates="jobs")
    tracks = relationship("Track", back_populates="job", cascade="all, delete-orphan")
    violations = relationship("Violation", back_populates="job", cascade="all, delete-orphan")

class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("processing_jobs.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    vehicle_class = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    first_frame = Column(Integer, nullable=False)
    last_frame = Column(Integer, nullable=False)
    trajectory_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    job = relationship("ProcessingJob", back_populates="tracks")
    speed_measurements = relationship("SpeedMeasurement", back_populates="track", cascade="all, delete-orphan")

class SpeedMeasurement(Base):
    __tablename__ = "speed_measurements"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    frame_index = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    instantaneous_kmh = Column(Float, nullable=True)
    smoothed_kmh = Column(Float, nullable=True)
    uncertainty_kmh = Column(Float, nullable=True)
    confidence_low_kmh = Column(Float, nullable=True)
    confidence_high_kmh = Column(Float, nullable=True)
    error_components_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    track = relationship("Track", back_populates="speed_measurements")

class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("processing_jobs.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    vehicle_class = Column(String(64), nullable=False)
    frame_index = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    estimated_speed_kmh = Column(Float, nullable=False)
    speed_limit_kmh = Column(Float, nullable=False)
    uncertainty_kmh = Column(Float, nullable=False)
    location_label = Column(String(128), default="Lane 1")
    review_status = Column(String(32), default="PENDING")  # PENDING, ACCEPTED, REJECTED
    reviewer_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    job = relationship("ProcessingJob", back_populates="violations")
    evidence = relationship("Evidence", back_populates="violation", uselist=False, cascade="all, delete-orphan")

class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, index=True)
    violation_id = Column(Integer, ForeignKey("violations.id"), nullable=False)
    full_frame_path = Column(String(512), nullable=False)
    crop_frame_path = Column(String(512), nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    violation = relationship("Violation", back_populates="evidence")

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    dataset_name = Column(String(128), nullable=False)
    detector_name = Column(String(128), nullable=False)
    tracker_name = Column(String(128), nullable=False)
    speed_method = Column(String(128), nullable=False)
    parameters_json = Column(JSON, nullable=True)
    mae_kmh = Column(Float, nullable=True)
    rmse_kmh = Column(Float, nullable=True)
    r2_score = Column(Float, nullable=True)
    status = Column(String(32), default="COMPLETED")
    created_at = Column(DateTime, default=utcnow)

class GroundTruthSequenceDB(Base):
    __tablename__ = "gt_sequences"

    id = Column(Integer, primary_key=True, index=True)
    sequence_id = Column(String(128), unique=True, nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=True)
    name = Column(String(255), nullable=False)
    source = Column(String(128), nullable=False)
    camera_calibration_json = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    sha256_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utcnow)

class GroundTruthObservationDB(Base):
    __tablename__ = "gt_observations"

    id = Column(Integer, primary_key=True, index=True)
    sequence_id = Column(String(128), ForeignKey("gt_sequences.sequence_id"), nullable=False)
    vehicle_id = Column(String(128), nullable=False)
    timestamp = Column(Float, nullable=False)
    frame_index = Column(Integer, nullable=False, default=0)
    speed_mps = Column(Float, nullable=False)
    speed_kmh = Column(Float, nullable=False)
    position_x = Column(Float, nullable=False)
    position_y = Column(Float, nullable=False)
    lane_id = Column(String(64), default="Lane 1")
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=utcnow)

class BenchmarkExperimentDB(Base):
    __tablename__ = "benchmark_experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String(128), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=True)
    gt_sequence_id = Column(String(128), ForeignKey("gt_sequences.sequence_id"), nullable=True)
    config_json = Column(JSON, nullable=True)
    detector_model = Column(String(128), nullable=False)
    tracker_model = Column(String(128), nullable=False)
    speed_method = Column(String(128), nullable=False)
    smoothing_method = Column(String(128), nullable=False)
    metrics_json = Column(JSON, nullable=True)
    telemetry_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)

class BenchmarkMatchDB(Base):
    __tablename__ = "benchmark_matches"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String(128), ForeignKey("benchmark_experiments.experiment_id"), nullable=False)
    gt_observation_id = Column(Integer, ForeignKey("gt_observations.id"), nullable=True)
    track_id = Column(Integer, nullable=False)
    frame_index = Column(Integer, nullable=False, default=0)
    gt_speed_kmh = Column(Float, nullable=False)
    est_speed_kmh = Column(Float, nullable=False)
    error_kmh = Column(Float, nullable=False)
    abs_error_kmh = Column(Float, nullable=False)
    uncertainty_kmh = Column(Float, nullable=False)
    matched_at = Column(DateTime, default=utcnow)
