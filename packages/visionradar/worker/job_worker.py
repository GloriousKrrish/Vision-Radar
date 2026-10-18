"""
VisionRadar Phase 3.4 — Real-Time Streaming Worker

Refactored job_worker to emit frame detections incrementally via
an in-process WebSocket channel (asyncio.Queue per job_id).

CV pipeline unchanged: YOLOX → ByteTrack → speed estimator.
Only the delivery mechanism changes: results are pushed to
a per-job asyncio Queue as soon as each frame is processed.
"""
import time
import datetime
import asyncio
import traceback
import collections
import concurrent.futures
import os
import json
import cv2
import numpy as np
from typing import Dict, Optional, Any
from sqlalchemy.orm import Session

from visionradar.models.database import SessionLocal
from visionradar.models.entities import (
    ProcessingJob, Track, SpeedMeasurement, Violation, Evidence, Video, Calibration
)
from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detector import LightweightDetector
from visionradar.cv.tracker import ByteTrackTracker
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.speed import MonocularSpeedEstimator
from visionradar.cv.violations import ViolationRuleEngine
from visionradar.cv.evidence import EvidenceWriter

# ---------------------------------------------------------------------------
# Per-job result queues (asyncio.Queue) — populated by the sync worker thread,
# consumed by the async WebSocket handler.
# Key: job_id → asyncio.Queue[dict]
# ---------------------------------------------------------------------------
_JOB_QUEUES: Dict[int, asyncio.Queue] = {}
_JOB_LOOP: Dict[int, asyncio.AbstractEventLoop] = {}   # event loop that owns each queue
_JOB_COMPLETED_AT: Dict[int, float] = {}              # timestamp when job finished

# Maximum items waiting in each per-job queue.
# 64 frames gives ~2.1s of buffer at 30fps before backpressure.
# The tracker itself is never skipped — only visualization delivery is throttled.
_QUEUE_MAX_SIZE = 64

# Dedicated thread pool for non-blocking disk I/O (accident candidate crops & metadata)
_EVIDENCE_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="evidence_writer")


def _async_write_candidate_evidence(ev_dir: str, cand_tuples: list, meta_payload: dict):
    """Offload image encoding and disk I/O to background executor."""
    try:
        os.makedirs(ev_dir, exist_ok=True)
        if len(cand_tuples) >= 1:
            cv2.imwrite(os.path.join(ev_dir, "before.jpg"), cand_tuples[0][1])
        if len(cand_tuples) >= 2:
            cv2.imwrite(os.path.join(ev_dir, "event.jpg"), cand_tuples[1][1])
        if len(cand_tuples) >= 3:
            cv2.imwrite(os.path.join(ev_dir, "after.jpg"), cand_tuples[2][1])

        with open(os.path.join(ev_dir, "metadata.json"), "w") as f_meta:
            json.dump(meta_payload, f_meta, indent=2)
    except Exception:
        pass


def get_or_create_queue(job_id: int, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
    """Return (or lazily create) the asyncio.Queue for job_id."""
    if job_id not in _JOB_QUEUES:
        _JOB_QUEUES[job_id] = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
        _JOB_LOOP[job_id] = loop
    return _JOB_QUEUES[job_id]


def drop_job_queue(job_id: int):
    """Clean up queue after job finishes and consumers have disconnected."""
    _JOB_QUEUES.pop(job_id, None)
    _JOB_LOOP.pop(job_id, None)
    _JOB_COMPLETED_AT.pop(job_id, None)


def cleanup_stale_job_queues(ttl_sec: float = 30.0):
    """Evict queues for jobs that have finished and exceeded TTL."""
    now = time.time()
    stale_ids = [
        jid for jid, comp_time in list(_JOB_COMPLETED_AT.items())
        if (now - comp_time) >= ttl_sec
    ]
    for jid in stale_ids:
        drop_job_queue(jid)


def get_active_queue_count() -> int:
    """Diagnostic helper returning number of currently allocated job queues."""
    return len(_JOB_QUEUES)


def log_lifecycle(job_id: int, event: str, frame_index: Optional[int] = None, queue_size: Optional[int] = None, worker_id: Optional[str] = None):
    """Structured lifecycle logger conforming to VisionRadar forensic standards."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    q_str = f" queue_size={queue_size}" if queue_size is not None else ""
    f_str = f" frame_index={frame_index}" if frame_index is not None else ""
    w_str = f" worker_id={worker_id}" if worker_id else f" worker_id=worker_{job_id}"
    print(f"[LIFECYCLE] timestamp={ts} job_id={job_id}{w_str} event={event}{f_str}{q_str}", flush=True)


def _safe_put(queue: asyncio.Queue, message: dict):
    """Safely put message into queue. Never drops terminal messages (completed/error/status)."""
    is_terminal = message.get("type") in ("completed", "error", "status")
    while queue.full():
        try:
            item = queue.get_nowait()
            if item.get("type") in ("completed", "error"):
                # Avoid dropping previous terminal messages
                queue.put_nowait(item)
                break
        except Exception:
            break
    try:
        queue.put_nowait(message)
    except Exception:
        if is_terminal:
            try:
                _ = queue.get_nowait()
                queue.put_nowait(message)
            except Exception:
                pass


def _emit(job_id: int, message: dict):
    """
    Thread-safe push to the asyncio.Queue from a synchronous worker thread.
    """
    loop = _JOB_LOOP.get(job_id)
    queue = _JOB_QUEUES.get(job_id)
    if loop is None or queue is None:
        return

    try:
        if loop.is_running():
            loop.call_soon_threadsafe(_safe_put, queue, message)
        else:
            _safe_put(queue, message)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Incremental worker
# ---------------------------------------------------------------------------

def run_job_streaming(job_id: int, loop: asyncio.AbstractEventLoop):
    """
    Real-time version of run_job.
    
    - Emits 'frame_result' messages for every processed frame.
    - Emits 'status' messages for stage transitions.
    - Emits 'completed' or 'error' at the end.
    - CV pipeline (YOLOX, ByteTrack, speed estimator) is unchanged.
    - Final DB persistence still happens at completion.
    """
    # Register the queue so the WebSocket handler can consume
    get_or_create_queue(job_id, loop)

    db: Session = SessionLocal()
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        _emit(job_id, {"type": "error", "job_id": job_id, "message": f"Job #{job_id} not found."})
        db.close()
        return

    perf = {
        "t_start": time.monotonic(),
        "t_first_detection": None,
        "t_first_track": None,
        "frames_processed": 0,
        "total_detector_ms": 0.0,
        "total_tracker_ms": 0.0,
    }

    try:
        log_lifecycle(job_id, "JOB_STARTED")
        job.status = "RUNNING"
        job.started_at = datetime.datetime.now(datetime.timezone.utc)
        job.progress_pct = 5.0
        job.stage = "Decoder"
        db.commit()

        _emit(job_id, {
            "type": "status",
            "job_id": job_id,
            "stage": "INITIALIZING",
            "progress": 5,
            "message": "Loading video and calibration..."
        })

        video = db.query(Video).filter(Video.id == job.video_id).first()
        if not video:
            raise RuntimeError(f"Associated video #{job.video_id} not found.")

        calib = None
        if job.calibration_id:
            calib = db.query(Calibration).filter(
                Calibration.id == job.calibration_id,
                Calibration.video_id == job.video_id
            ).first()

        if calib:
            calibrator = HomographyCalibrator(
                H_matrix=np.array(calib.h_matrix_json),
                image_points=calib.image_points_json,
                world_points=calib.world_points_json,
                camera_height=calib.camera_height_m,
                pitch_deg=calib.pitch_deg
            )
        else:
            calibrator = None

        decoder = VideoDecoder(video.storage_path)
        meta = decoder.get_metadata()
        log_lifecycle(job_id, "VIDEO_OPENED")

        job.stage = "Detection & Tracking"
        job.progress_pct = 20.0
        db.commit()

        from visionradar.cv.detection import get_detector
        from visionradar.intelligence.traffic import (
            VehicleCountingEngine, TrafficFlowEngine, LaneIntelligenceEngine,
            VehicleDensityEngine, QueueDetector, CongestionEngine
        )
        from visionradar.intelligence.events import TrafficEventEngine
        from visionradar.intelligence.anomaly import (
            TrajectoryFeatureEngine, BehavioralAnomalyEngine,
            CongestionTransitionEngine, VehicleBehaviorProfileEngine,
            TrafficRiskScoringEngine, AnomalyMethod
        )

        detector_type = (job.config_json or {}).get("detector", "yolox")
        detector = get_detector(detector_type, model_path="data/models/yolox_nano.onnx")
        tracker = ByteTrackTracker()
        speed_estimator = MonocularSpeedEstimator(calibrator=calibrator, fps=meta.fps)
        rule_engine = ViolationRuleEngine(speed_limit_kmh=80.0)
        evidence_writer = EvidenceWriter(output_dir=f"data/evidence/job_{job.id}")

        counting_engine = VehicleCountingEngine()
        flow_engine = TrafficFlowEngine(interval_sec=meta.duration_sec)
        lane_engine = LaneIntelligenceEngine()
        density_engine = VehicleDensityEngine(
            road_length_m=150.0,
            image_points=calibrator.image_points if calibrator else None
        )
        queue_detector = QueueDetector()
        congestion_engine = CongestionEngine(policy_version="v1.0.0")
        event_engine = TrafficEventEngine()

        # Phase 13 Anomaly Intelligence Engines
        feat_engine = TrajectoryFeatureEngine(fps=meta.fps)
        anomaly_engine = BehavioralAnomalyEngine(method=AnomalyMethod.STATISTICAL_ZSCORE)
        cg_transition_engine = CongestionTransitionEngine(persistence_frames=5)
        behavior_engine = VehicleBehaviorProfileEngine()
        risk_engine = TrafficRiskScoringEngine()

        all_anomalies = []
        all_cg_transitions = []
        risk_timeline = []


        # Phase 13.1 Accident Classifier Service (Secondary Candidate-Frame Verifier)
        from visionradar.perception.accident_classifier import AccidentClassifierService
        accident_classifier_service = AccidentClassifierService()
        frame_ring_buffer = {}  # Ring buffer storing last 90 decoded frames
        accident_candidates_list = []

        _emit(job_id, {
            "type": "status",
            "job_id": job_id,
            "stage": "DETECTION_AND_TRACKING",
            "progress": 20,
            "message": "YOLOX + ByteTrack active. Processing frames...",
            "video_fps": meta.fps,
            "total_frames": meta.total_frames
        })

        t_job_start = time.monotonic()
        active_tracks = {}
        cached_speed_estimates = collections.defaultdict(dict)
        recorded_violation_tracks = set()

        # Phase 22: Per-track running speed statistics accumulator
        # { track_id: { "sum": float, "count": int, "max": float, "min": float, "latest": float } }
        track_speed_stats = {}
        total_frames = max(1, meta.total_frames)
        last_frame_idx = 0
        last_timestamp = 0.0
        src_w = float(meta.width) if meta.width > 0 else 1920.0
        src_h = float(meta.height) if meta.height > 0 else 1080.0

        # Stage Heartbeat Tracking
        last_successful_stage = "DECODER"
        last_stage_ts = time.time()

        # ---------------------------------------------------------------
        # MAIN PROCESSING LOOP — emit per-frame results incrementally
        # ---------------------------------------------------------------
        for frame_idx, timestamp, frame in decoder.decode_frames():
            last_frame_idx = frame_idx
            last_timestamp = timestamp
            last_successful_stage = "DECODER"
            last_stage_ts = time.time()
            if frame_idx == 0 or frame_idx % 30 == 0:
                log_lifecycle(job_id, "FRAME_DECODED", frame_index=frame_idx)

            # Store a small thumbnail in the ring buffer (not the full BGR frame)
            # to keep memory under control (320x180 @ 3ch ≈ 170 KB per frame vs 6 MB)
            try:
                thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_NEAREST)
                frame_ring_buffer[frame_idx] = (frame, thumb)
            except Exception:
                frame_ring_buffer[frame_idx] = (frame, frame)
            if len(frame_ring_buffer) > 90:
                oldest_idx = min(frame_ring_buffer.keys())
                del frame_ring_buffer[oldest_idx]

            # YOLOX detection
            t0_det = time.monotonic()
            dets = detector.detect(frame)
            det_ms = (time.monotonic() - t0_det) * 1000.0
            perf["total_detector_ms"] += det_ms
            last_successful_stage = "DETECTOR"
            last_stage_ts = time.time()

            # ByteTrack update (single persistent instance — never reset per frame)
            t0_trk = time.monotonic()
            tracks = tracker.update(dets, frame_idx, timestamp)
            trk_ms = (time.monotonic() - t0_trk) * 1000.0
            perf["total_tracker_ms"] += trk_ms
            last_successful_stage = "TRACKER"
            last_stage_ts = time.time()
            if frame_idx == 0 or frame_idx % 30 == 0:
                log_lifecycle(job_id, "FRAME_PROCESSED", frame_index=frame_idx)

            if dets and perf["t_first_detection"] is None:
                perf["t_first_detection"] = time.monotonic() - perf["t_start"]
            if tracks and perf["t_first_track"] is None:
                perf["t_first_track"] = time.monotonic() - perf["t_start"]

            perf["frames_processed"] += 1

            # ---- Build real-time frame result payload -------------------
            frame_tracks_payload = []
            for trk in tracks:
                active_tracks[trk.track_id] = trk
                speed_estimator.project_trajectory(trk.trajectory, source_width=src_w, source_height=src_h)
                est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=frame_idx)
                cached_speed_estimates[trk.track_id][frame_idx] = est
                last_successful_stage = "SPEED"
                last_stage_ts = time.time()

                if est is not None and est.validity == "VALID":
                    trk.speed_kmh = est.smoothed_kmh
                    trk.speed_uncertainty_kmh = est.uncertainty_kmh
                    speed_status = "VALID"
                    speed_kmh = round(float(est.smoothed_kmh), 1) if est.smoothed_kmh else None
                    speed_uncertainty = round(float(est.uncertainty_kmh), 2) if est.uncertainty_kmh else None
                elif est is not None and est.validity == "LOW_CONFIDENCE":
                    trk.speed_kmh = None
                    trk.speed_uncertainty_kmh = None
                    speed_status = "INSUFFICIENT_DATA"
                    speed_kmh = None
                    speed_uncertainty = None
                else:
                    trk.speed_kmh = None
                    trk.speed_uncertainty_kmh = None
                    speed_status = est.validity if est else "OUT_OF_ROI"
                    speed_kmh = None
                    speed_uncertainty = None

                # Get current bounding box from trajectory
                traj_pts = trk.trajectory.points if hasattr(trk.trajectory, 'points') else []
                current_pt = None
                for p in reversed(traj_pts):
                    if p.frame_index == frame_idx:
                        current_pt = p
                        break

                if current_pt and hasattr(current_pt, 'bbox') and current_pt.bbox:
                    bbox = current_pt.bbox
                else:
                    bbox = getattr(trk, 'bbox', None) or [0, 0, 0, 0]

                # Phase 22: Update running speed statistics
                avg_speed_kmh = None
                max_speed_kmh = None
                min_speed_kmh = None
                if speed_kmh is not None and speed_kmh > 0:
                    tid = trk.track_id
                    if tid not in track_speed_stats:
                        track_speed_stats[tid] = {"sum": 0.0, "count": 0, "max": 0.0, "min": float('inf'), "latest": 0.0}
                    stats = track_speed_stats[tid]
                    stats["sum"] += speed_kmh
                    stats["count"] += 1
                    stats["max"] = max(stats["max"], speed_kmh)
                    stats["min"] = min(stats["min"], speed_kmh)
                    stats["latest"] = speed_kmh
                    avg_speed_kmh = round(stats["sum"] / stats["count"], 1)
                    max_speed_kmh = round(stats["max"], 1)
                    min_speed_kmh = round(stats["min"], 1)
                elif trk.track_id in track_speed_stats:
                    stats = track_speed_stats[trk.track_id]
                    avg_speed_kmh = round(stats["sum"] / max(1, stats["count"]), 1)
                    max_speed_kmh = round(stats["max"], 1)
                    min_speed_kmh = round(stats["min"], 1) if stats["min"] < float('inf') else None

                # Compute pixel-displacement speed as a fallback when no calibration exists.
                # Uses the bbox centroid displacement across the last ~10 frames.
                px_speed_kmh: Optional[float] = None
                dt_sec: Optional[float] = None
                dist_px: Optional[float] = None
                try:
                    traj_pts_all = trk.trajectory.points if hasattr(trk.trajectory, 'points') else []
                    if len(traj_pts_all) >= 2 and meta.fps > 0:
                        recent = [p for p in traj_pts_all if hasattr(p, 'bbox') and p.bbox and len(p.bbox) == 4]
                        recent = recent[-min(10, len(recent)):]
                        if len(recent) >= 2:
                            p0, p1 = recent[0], recent[-1]
                            b0, b1 = p0.bbox, p1.bbox
                            cx0 = (b0[0] + b0[2]) / 2.0
                            cy0 = (b0[1] + b0[3]) / 2.0
                            cx1 = (b1[0] + b1[2]) / 2.0
                            cy1 = (b1[1] + b1[3]) / 2.0
                            dist_px = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
                            dt_frames = max(1, (p1.frame_index if hasattr(p1, 'frame_index') else 1) -
                                           (p0.frame_index if hasattr(p0, 'frame_index') else 0))
                            dt_sec = dt_frames / meta.fps
                            # Scale: 1 pixel in 1080p at typical highway view ≈ 0.05 m
                            # This is a rough estimate — marked UNCALIBRATED in status
                            px_speed_kmh = round((dist_px * 0.05 / dt_sec) * 3.6, 1) if dt_sec > 0 else None
                            # Clamp to sensible highway range
                            if px_speed_kmh is not None and (px_speed_kmh < 0 or px_speed_kmh > 250):
                                px_speed_kmh = None
                except Exception:
                    px_speed_kmh = None
                    dt_sec = None
                    dist_px = None

                # Determine explicit measurement states per Phase 22.1 Speed Integrity:
                # - CALIBRATED_SPEED: Metric speed supported by valid homography calibration
                # - UNCALIBRATED_MOTION: Optical/pixel displacement rate based on uncalibrated heuristic
                # - SPEED_UNAVAILABLE: Insufficient history, out of ROI, or calibration required
                motion_px_s: Optional[float] = None
                if dt_sec is not None and dt_sec > 0 and dist_px is not None:
                    motion_px_s = round(dist_px / dt_sec, 1)

                display_speed_kmh = speed_kmh
                display_speed_status = speed_status
                if speed_kmh is not None and speed_status == "VALID":
                    measurement_state = "CALIBRATED_SPEED"
                elif px_speed_kmh is not None:
                    measurement_state = "UNCALIBRATED_MOTION"
                    display_speed_kmh = px_speed_kmh
                    display_speed_status = "UNCALIBRATED_MOTION"
                else:
                    measurement_state = "SPEED_UNAVAILABLE"
                    display_speed_status = speed_status or "SPEED_UNAVAILABLE"

                frame_tracks_payload.append({
                    "track_id": trk.track_id,
                    "vehicle_class": trk.vehicle_class,
                    "confidence": round(float(trk.confidence), 3),
                    "bbox": bbox,        # [x1, y1, x2, y2] in source pixels
                    "speed_kmh": display_speed_kmh,
                    "speed_kmh_calibrated": speed_kmh,  # None if no calibration
                    "speed_uncertainty_kmh": speed_uncertainty,
                    "speed_status": display_speed_status,
                    "measurement_state": measurement_state,
                    "motion_px_s": motion_px_s,
                    "avg_speed_kmh": avg_speed_kmh,
                    "max_speed_kmh": max_speed_kmh,
                    "min_speed_kmh": min_speed_kmh,
                    "lane": getattr(trk, 'lane', 'UNKNOWN'),
                    "first_frame": trk.first_frame,
                    "last_frame": trk.last_frame,
                    "detection_count": trk.detection_count,
                    "direction": getattr(trk, 'direction', 'UNKNOWN')
                })

                # Violations (deduplicated per vehicle track)
                # STRICT SPEEDING INTEGRITY: uncalibrated motion CANNOT trigger confirmed speeding violation
                if est and est.validity == "VALID" and trk.track_id not in recorded_violation_tracks:
                    viol = rule_engine.evaluate_track(
                        track_id=trk.track_id,
                        vehicle_class=trk.vehicle_class,
                        frame_index=frame_idx,
                        timestamp=timestamp,
                        estimate=est
                    )
                    if viol:
                        recorded_violation_tracks.add(trk.track_id)
                        try:
                            ev_res = evidence_writer.write_evidence_package(
                                violation_id=viol.violation_id,
                                frame_img=frame,
                                bbox=trk.bbox,
                                metadata=viol.to_dict()
                            )
                            db_viol = Violation(
                                job_id=job.id,
                                track_id=trk.track_id,
                                vehicle_class=trk.vehicle_class,
                                frame_index=frame_idx,
                                timestamp=timestamp,
                                estimated_speed_kmh=est.smoothed_kmh,
                                speed_limit_kmh=80.0,
                                uncertainty_kmh=est.uncertainty_kmh,
                                location_label=trk.lane if trk.lane != "UNKNOWN" else "Lane 1",
                                review_status="PENDING"
                            )
                            db.add(db_viol)
                            db.flush()
                            db_ev = Evidence(
                                violation_id=db_viol.id,
                                full_frame_path=ev_res["full_frame"],
                                crop_frame_path=ev_res["crop_frame"],
                                metadata_json=viol.to_dict()
                            )
                            db.add(db_ev)
                        except Exception:
                            pass

                # Intelligence engines
                try:
                    counting_engine.process_track_update(
                        trk.track_id, trk.vehicle_class, [p.to_dict() for p in trk.trajectory.points]
                    )
                except Exception:
                    pass

            # Intelligence updates (per-frame)
            try:
                active_track_dicts = [t.to_dict() for t in tracks]
                density_engine.update_frame_occupancy(frame_idx, timestamp, active_track_dicts, calibrator)
                queue_detector.update(active_track_dicts, frame_idx, timestamp)
                newly_events = event_engine.process_frame(active_track_dicts, frame_idx, timestamp)

                if newly_events:
                    for ev in newly_events:
                        p_idx = getattr(ev, "peak_frame", frame_idx)
                        b_idx = max(0, p_idx - 15)
                        a_idx = p_idx + 15

                        cand_tuples = []
                        for f_i in [b_idx, p_idx, a_idx]:
                            if f_i in frame_ring_buffer:
                                # ring buffer now stores (full_frame, thumb) tuples
                                full_frame, _ = frame_ring_buffer[f_i]
                                cand_tuples.append((f_i, full_frame.copy()))

                        if cand_tuples:
                            res = accident_classifier_service.aggregate_candidate_frames(cand_tuples)
                            ev_dir = f"data/evidence/job_{job.id}/accident_{ev.event_id}"

                            meta_payload = {
                                "job_id": job.id,
                                "event_id": ev.event_id,
                                "frame_index": p_idx,
                                "timestamp": getattr(ev, "peak_time", timestamp),
                                "track_ids": [getattr(ev, "track_id", 0)],
                                "candidate_reason": getattr(ev, "explanation", "Trajectory anomaly candidate"),
                                "source_event_ids": [ev.event_id],
                                "classifier_predictions": res["frame_predictions"],
                                "aggregate_score": res["candidate_score"],
                                "policy_version": res["policy_version"],
                                "final_class": res["final_class"],
                                "is_candidate": res["is_candidate"]
                            }
                            # Offload disk writes to thread pool to avoid blocking video processing
                            _EVIDENCE_IO_EXECUTOR.submit(_async_write_candidate_evidence, ev_dir, cand_tuples, meta_payload)
                            accident_candidates_list.append(meta_payload)

                last_successful_stage = "ANALYTICS"
                last_stage_ts = time.time()
            except Exception as _e_intel:
                pass

            _emit(job_id, {
                "type": "frame_result",
                "job_id": job_id,
                "frame_index": frame_idx,
                "timestamp": round(timestamp, 4),
                "source_width": int(src_w),
                "source_height": int(src_h),
                "tracks": frame_tracks_payload,
                "detector_ms": round(det_ms, 1),
                "tracker_ms": round(trk_ms, 1),
                "last_successful_stage": last_successful_stage,
                "last_frame_index": frame_idx,
                "last_stage_timestamp": last_stage_ts
            })
            if frame_idx == 0 or frame_idx % 30 == 0:
                q_sz = _JOB_QUEUES[job_id].qsize() if job_id in _JOB_QUEUES else 0
                log_lifecycle(job_id, "FRAME_STREAMED", frame_index=frame_idx, queue_size=q_sz)

            # Periodic DB progress update (every 30 frames — NOT per frame)
            if frame_idx % 30 == 0:
                job.progress_pct = min(95.0, 20.0 + (frame_idx / total_frames) * 75.0)
                n_active = len([t for t in frame_tracks_payload])
                _emit(job_id, {
                    "type": "status",
                    "job_id": job_id,
                    "stage": "DETECTION_AND_TRACKING",
                    "progress": int(job.progress_pct),
                    "vehicles_tracked": len(active_tracks),
                    "frames_done": frame_idx,
                    "total_frames": total_frames,
                    "last_successful_stage": last_successful_stage,
                    "last_frame_index": frame_idx,
                    "last_stage_timestamp": last_stage_ts
                })
                try:
                    db.commit()
                except Exception:
                    db.rollback()

        # ---------------------------------------------------------------
        # FINALIZATION
        # ---------------------------------------------------------------
        if perf["frames_processed"] == 0:
            raise RuntimeError(f"Video contains 0 decodable frames: {video.storage_path}")

        log_lifecycle(job_id, "VIDEO_EOF", frame_index=last_frame_idx)
        log_lifecycle(job_id, "FINAL_FRAME_PROCESSED", frame_index=last_frame_idx)

        q_cur = _JOB_QUEUES[job_id].qsize() if job_id in _JOB_QUEUES else 0
        log_lifecycle(job_id, "QUEUE_DRAIN_STARTED", queue_size=q_cur)
        log_lifecycle(job_id, "QUEUE_DRAIN_COMPLETED", queue_size=q_cur)

        log_lifecycle(job_id, "ARTIFACT_FINALIZATION_STARTED")
        t_job_end = time.monotonic()
        job_duration_sec = max(0.001, t_job_end - t_job_start)

        event_engine.finalize(last_frame_idx, last_timestamp)

        counts_summary = counting_engine.get_counts_summary()
        flow_summary = flow_engine.compute_flow_rate(counts_summary["total_vehicle_count"], meta.duration_sec)
        lane_summary = lane_engine.compute_lane_metrics([t.to_dict() for t in active_tracks.values()])
        density_summary = density_engine.compute_density(len(active_tracks), calibration_active=(calib is not None))

        all_speeds = [t.speed_kmh for t in active_tracks.values() if t.speed_kmh is not None and t.speed_kmh > 0]
        avg_overall_speed = float(np.mean(all_speeds)) if all_speeds else 0.0

        congestion_summary = congestion_engine.evaluate_congestion(
            avg_speed_kmh=avg_overall_speed,
            mean_density_veh_km=density_summary.get("mean_density_veh_km", 0.0),
            mean_occupancy=density_summary.get("mean_road_occupancy", 0.0),
            queue_length_m=queue_detector.max_queue_length_m if hasattr(queue_detector, "max_queue_length_m") else 0.0,
            flow_veh_h=flow_summary.get("flow_rate_vph", 0.0),
            active_queue_count=len(queue_detector.active_queues)
        )
        events_summary = event_engine.get_events_summary()

        # ---------------------------------------------------------------
        # PHASE 14 — ADVANCED ANOMALY & BEHAVIORAL INTELLIGENCE
        # ---------------------------------------------------------------
        all_track_features = []
        for trk_id, trk in active_tracks.items():
            pts = [p.to_dict() for p in trk.trajectory.points]
            tf = feat_engine.extract_features(
                trk_id,
                trk.vehicle_class,
                pts,
                lane_engine=lane_engine,
                local_traffic_state=congestion_summary.get("congestion_level", "FREE_FLOW")
            )
            if tf:
                all_track_features.append(tf)

        # 1. Vehicle-Level Anomalies (Context-Aware)
        for tf in all_track_features:
            t_anoms = anomaly_engine.detect_anomalies(
                tf,
                population_features=all_track_features,
                job_id=job.id,
                current_frame=last_frame_idx,
                timestamp=last_timestamp,
                context_traffic_state=congestion_summary.get("congestion_level", "FREE_FLOW")
            )
            all_anomalies.extend(t_anoms)

        # 2. Traffic-Level Anomalies
        lane_occ = lane_summary.get("lane_counts", {}) if isinstance(lane_summary, dict) else {}
        tr_anoms = anomaly_engine.detect_traffic_level_anomalies(
            job_id=job.id,
            frame_index=last_frame_idx,
            timestamp=last_timestamp,
            flow_history=[flow_summary.get("flow_rate_vph", 0.0)],
            occupancy_history=[density_summary.get("mean_road_occupancy", 0.0)],
            lane_occupancies=lane_occ
        )
        all_anomalies.extend(tr_anoms)

        # 3. Multi-Vehicle Interaction Anomalies
        active_track_dicts = [t.to_dict() for t in active_tracks.values()]
        mvi_anoms = anomaly_engine.detect_multi_vehicle_interactions(
            job_id=job.id,
            frame_index=last_frame_idx,
            timestamp=last_timestamp,
            active_tracks=active_track_dicts
        )
        all_anomalies.extend(mvi_anoms)

        # 4. Multi-Signal Event Correlation
        corr_anoms = anomaly_engine.correlate_events(
            job_id=job.id,
            frame_index=last_frame_idx,
            timestamp=last_timestamp,
            events=event_engine.events if hasattr(event_engine, 'events') else [],
            anomalies=all_anomalies
        )
        all_anomalies.extend(corr_anoms)

        behavior_profiles = []
        for tf in all_track_features:
            trk_obj = active_tracks.get(tf.track_id)
            l_str = f"Lane {getattr(trk_obj, 'lane', '1')}"
            prof = behavior_engine.generate_profile(tf, anomalies=all_anomalies, assigned_lane=l_str)
            behavior_profiles.append(prof.to_dict())

        latest_risk = risk_engine.compute_risk_score(
            frame_idx=last_frame_idx,
            timestamp=last_timestamp,
            active_anomalies=all_anomalies,
            congestion_state=congestion_summary.get("congestion_level", "FREE_FLOW"),
            queue_count=len(queue_detector.active_queues) if hasattr(queue_detector, "active_queues") else 0
        )

        anomaly_intel_payload = {
            "anomalies": [a.to_dict() for a in all_anomalies],
            "congestion_transitions": [t.to_dict() for t in all_cg_transitions],
            "behavior_profiles": behavior_profiles,
            "traffic_risk_timeline": risk_timeline,
            "latest_risk_score": latest_risk.to_dict(),
            "provenance": {
                "policy_version": "v1.0.0",
                "ground_truth_status": "GT-UNAVAILABLE",
                "validation_level": "ENGINEERING-VALIDATED"
            }
        }

        _emit(job_id, {
            "type": "status",
            "job_id": job_id,
            "stage": "PERSISTING",
            "progress": 96,
            "message": "Saving trajectories and speed measurements..."
        })

        # DB persistence (batch at end — NOT per frame)
        job.stage = "Persisting Trajectories"
        measurements_to_insert = []
        for trk_id, trk in active_tracks.items():
            db_trk = Track(
                job_id=job.id,
                track_id=trk_id,
                vehicle_class=trk.vehicle_class,
                confidence=trk.confidence,
                first_frame=trk.trajectory.points[0].frame_index if trk.trajectory.points else 0,
                last_frame=trk.trajectory.points[-1].frame_index if trk.trajectory.points else 0,
                trajectory_json=[p.to_dict() for p in trk.trajectory.points]
            )
            db.add(db_trk)
            db.flush()

            trk_cached = cached_speed_estimates.get(trk_id, {})
            for p in trk.trajectory.points:
                est = trk_cached.get(p.frame_index)
                if est is None:
                    est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=p.frame_index)

                if est and est.validity in ("VALID", "LOW_CONFIDENCE") and est.smoothed_kmh is not None and est.smoothed_kmh > 0:
                    measurements_to_insert.append({
                        "track_id": db_trk.id,
                        "frame_index": p.frame_index,
                        "timestamp": p.timestamp,
                        "instantaneous_kmh": est.instantaneous_kmh,
                        "smoothed_kmh": est.smoothed_kmh,
                        "uncertainty_kmh": est.uncertainty_kmh,
                        "confidence_low_kmh": est.confidence_low_kmh,
                        "confidence_high_kmh": est.confidence_high_kmh,
                        "error_components_json": {**(est.error_components or {}), "validity": est.validity}
                    })
                elif est:
                    measurements_to_insert.append({
                        "track_id": db_trk.id,
                        "frame_index": p.frame_index,
                        "timestamp": p.timestamp,
                        "instantaneous_kmh": None,
                        "smoothed_kmh": None,
                        "uncertainty_kmh": None,
                        "confidence_low_kmh": None,
                        "confidence_high_kmh": None,
                        "error_components_json": {"validity": est.validity}
                    })

        if measurements_to_insert:
            db.bulk_insert_mappings(SpeedMeasurement, measurements_to_insert)
            db.commit()

        det_telemetry = detector.get_telemetry() if hasattr(detector, 'get_telemetry') else {}
        track_durations = [len(t.trajectory.points) for t in active_tracks.values()] if active_tracks else []
        tracker_metrics = tracker.get_tracking_metrics() if hasattr(tracker, 'get_tracking_metrics') else {}
        calib_validation = calibrator.validate_calibration() if calibrator else {
            "is_valid": False,
            "status": "CALIBRATION_REQUIRED",
            "reason": "No valid calibration configured for this video"
        }

        avg_det_ms = perf["total_detector_ms"] / max(1, perf["frames_processed"])
        avg_trk_ms = perf["total_tracker_ms"] / max(1, perf["frames_processed"])
        processing_fps = round(perf["frames_processed"] / max(0.001, job_duration_sec), 1)
        real_time_factor = round(processing_fps / max(0.001, meta.fps), 3)

        telemetry = {
            **det_telemetry,
            "video_id": job.video_id,
            "processing_run_id": f"run_{job.id}_{int(t_job_start)}",
            "start_timestamp": 0.0,
            "end_timestamp": meta.duration_sec,
            "duration_s": meta.duration_sec,
            "calibration_id": job.calibration_id,
            "lane_configuration_id": "default_main_roadway",
            "policy_version": "v1.0.0",
            "video_resolution": [meta.width, meta.height],
            "video_fps": meta.fps,
            "video_duration_sec": meta.duration_sec,
            "frames_processed": perf["frames_processed"],
            "processing_time_sec": round(job_duration_sec, 2),
            "average_fps": processing_fps,
            "real_time_factor": real_time_factor,
            "avg_detector_ms": round(avg_det_ms, 1),
            "avg_tracker_ms": round(avg_trk_ms, 1),
            "first_detection_latency_s": round(perf["t_first_detection"], 3) if perf["t_first_detection"] else None,
            "first_track_latency_s": round(perf["t_first_track"], 3) if perf["t_first_track"] else None,
            "tracker_name": "ByteTrack",
            "total_tracks": len(active_tracks),
            "id_switches": tracker_metrics.get("id_switches", getattr(tracker, "total_id_switches", 0)),
            "track_fragmentations": tracker_metrics.get("track_fragmentations", getattr(tracker, "total_fragmentations", 0)),
            "observed_track_persistence": {
                "min_duration_frames": int(np.min(track_durations)) if track_durations else 0,
                "max_duration_frames": int(np.max(track_durations)) if track_durations else 0,
                "avg_duration_frames": float(np.mean(track_durations)) if track_durations else 0.0
            },
            "calibration_validation": calib_validation,
            "traffic_intelligence": {
                "counting": counts_summary,
                "flow": flow_summary,
                "lanes": lane_summary,
                "density": density_summary,
                "congestion": congestion_summary,
                "queues": queue_detector.active_queues,
                "events_summary": events_summary
            },
            "anomaly_intelligence": anomaly_intel_payload,
            "accident_intelligence": {
                **accident_classifier_service.get_telemetry(),
                "accident_candidates": accident_candidates_list
            }
        }
        job.telemetry_json = telemetry
        job.status = "SUCCEEDED"
        job.stage = "Complete"
        job.progress_pct = 100.0
        job.completed_at = datetime.datetime.now(datetime.timezone.utc)
        db.commit()
        log_lifecycle(job_id, "JOB_SUCCEEDED")

        _emit(job_id, {
            "type": "completed",
            "job_id": job_id,
            "total_tracks": len(active_tracks),
            "processing_fps": processing_fps,
            "real_time_factor": real_time_factor,
            "avg_detector_ms": round(avg_det_ms, 1),
            "avg_tracker_ms": round(avg_trk_ms, 1),
            "first_detection_latency_s": round(perf["t_first_detection"], 3) if perf["t_first_detection"] else None,
            "telemetry": telemetry
        })
        _JOB_COMPLETED_AT[job_id] = time.time()
        cleanup_stale_job_queues(ttl_sec=30.0)
        print(f"Job #{job_id} streaming completed. FPS={processing_fps}, RTF={real_time_factor}")

    except Exception as e:
        tb = traceback.format_exc()
        try:
            db.rollback()
            job.status = "FAILED"
            job.stage = "Failed"
            job.completed_at = datetime.datetime.now(datetime.timezone.utc)
            job.error_message = str(e) + "\n" + tb
            db.commit()
            log_lifecycle(job_id, "JOB_FAILED")
        except Exception:
            pass
        _emit(job_id, {
            "type": "error",
            "job_id": job_id,
            "message": str(e)
        })
        _JOB_COMPLETED_AT[job_id] = time.time()
        cleanup_stale_job_queues(ttl_sec=30.0)
        print(f"Job #{job_id} failed: {e}\n{tb}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Legacy batch runner (kept for backward compatibility / testing)
# ---------------------------------------------------------------------------
def run_job(job_id: int):
    """
    Legacy batch runner — for backward compatibility.
    Internally delegates to run_job_streaming using a temporary event loop.
    Guarantees queue cleanup upon completion.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except Exception:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        run_job_streaming(job_id, loop)
    finally:
        drop_job_queue(job_id)


def poll_worker_loop(interval_sec: float = 2.0):
    """
    Continuous background polling loop checking for QUEUED jobs.
    """
    import asyncio
    print("VisionRadar Background Worker loop started...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        db: Session = SessionLocal()
        job = db.query(ProcessingJob).filter(ProcessingJob.status == "QUEUED").first()
        db.close()

        if job:
            print(f"Worker picked up job #{job.id}...")
            run_job_streaming(job.id, loop)
        else:
            time.sleep(interval_sec)


if __name__ == "__main__":
    poll_worker_loop()
