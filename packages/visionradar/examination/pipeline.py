import time
import datetime
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from visionradar.models.database import SessionLocal
from visionradar.models.entities import (
    ProcessingJob,
    Video,
    ExaminationTrackRecord,
    ExaminationEventRecord,
    ExaminationZoneRecord
)
from visionradar.models.envelope import CommonEnvelope
from visionradar.examination.schemas import (
    SeatingZone,
    ZoneType,
    ExaminationFrameResult
)
from visionradar.examination.detector import ExaminationDetector
from visionradar.examination.tracker import PersonTracker, ExaminationObjectTracker, associate_persons_and_objects
from visionradar.examination.seating import SeatingZoneManager
from visionradar.examination.events import ExaminationEventEngine
from visionradar.worker.job_worker import _emit, is_job_cancelled

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)

def run_examination_job(job_id: int):
    """
    Independent worker pipeline for Examination Intelligence (Layer C).
    Executes person detection, object tracking, spatial person-object association,
    seating zone analysis, and evidence-based temporal candidate event tracking.
    Never mutates or accesses vehicle ByteTrack or traffic speed logic.
    """
    db: Session = SessionLocal()
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        print(f"[ExaminationPipeline] Error: Job #{job_id} not found.")
        db.close()
        return

    video = db.query(Video).filter(Video.id == job.video_id).first()
    if not video or not os.path.exists(video.storage_path):
        job.status = "FAILED"
        job.stage = "Error"
        job.error_message = f"Video file not found for Job #{job_id}"
        db.commit()
        db.close()
        _emit(job_id, {"type": "error", "job_id": job_id, "message": job.error_message})
        return

    # Extract user-configured seating zones from job config if provided
    config = job.config_json or {}
    configured_zones_raw = config.get("zones", [])
    configured_zones = []
    for z in configured_zones_raw:
        try:
            configured_zones.append(
                SeatingZone(
                    zone_id=z["zone_id"],
                    name=z["name"],
                    zone_type=ZoneType(z.get("zone_type", "SEATING_ZONE")),
                    polygon=z["polygon"]
                )
            )
        except Exception:
            pass

    # Save initial zones to database
    for z in configured_zones:
        existing_z = db.query(ExaminationZoneRecord).filter(
            ExaminationZoneRecord.video_id == video.id,
            ExaminationZoneRecord.zone_id == z.zone_id
        ).first()
        if not existing_z:
            db.add(
                ExaminationZoneRecord(
                    video_id=video.id,
                    zone_id=z.zone_id,
                    name=z.name,
                    zone_type=z.zone_type.value,
                    polygon_json=z.polygon
                )
            )
    db.commit()

    # Open video capture
    cap = cv2.VideoCapture(video.storage_path)
    if not cap.isOpened():
        job.status = "FAILED"
        job.stage = "Error"
        job.error_message = f"Failed to open video capture for: {video.storage_path}"
        db.commit()
        db.close()
        _emit(job_id, {"type": "error", "job_id": job_id, "message": job.error_message})
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    # Transition job to RUNNING
    job.status = "RUNNING"
    job.stage = "ExaminationInference"
    job.started_at = utcnow()
    job.progress_pct = 0.0
    db.commit()

    _emit(job_id, {
        "type": "status",
        "job_id": job_id,
        "stage": "ExaminationInference",
        "progress_pct": 0.0,
        "analysis_module": "examination"
    })

    # Initialize examination components
    detector = ExaminationDetector()
    tracker = PersonTracker()
    object_tracker = ExaminationObjectTracker()
    seating_manager = SeatingZoneManager(configured_zones)
    event_engine = ExaminationEventEngine(job_id=job_id, video_id=video.id, fps=fps)

    frame_index = 0
    t_start = time.perf_counter()
    processed_count = 0
    inference_durations = []

    # Processing loop
    while cap.isOpened():
        if is_job_cancelled(job_id):
            job.status = "CANCELLED"
            job.stage = "Cancelled"
            job.error_message = "Cancelled by user"
            db.commit()
            _emit(job_id, {"type": "status", "job_id": job_id, "stage": "Cancelled", "progress_pct": job.progress_pct})
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_s = (msec / 1000.0) if msec > 0 else (frame_index / fps)

        processing_mode = config.get("processing_mode", "balanced")
        if processing_mode == "high_speed":
            stride = 3
        elif processing_mode == "forensic":
            stride = 1
        else:
            stride = 2

        is_det_frame = (frame_index % stride == 0)
        t_inf0 = time.perf_counter()
        if is_det_frame:
            raw_persons, raw_objects = detector.detect(frame)
            active_persons = tracker.update(raw_persons, frame_index)
            tracked_objects = object_tracker.update(raw_objects, frame_index)
        else:
            active_persons = tracker.predict_step(frame_index)
            tracked_objects = object_tracker.predict_step(frame_index)

        associate_persons_and_objects(active_persons, tracked_objects, object_tracker.tracks)
        t_inf = time.perf_counter() - t_inf0
        inference_durations.append(t_inf)

        # Associate seating zones
        for p in active_persons:
            seating_manager.associate_person(p)

        # Evaluate sequential observable events
        active_events = event_engine.process_frame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            persons=active_persons,
            objects=tracked_objects,
            zones=seating_manager.get_zones()
        )

        frame_result = ExaminationFrameResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            persons=active_persons,
            objects=tracked_objects,
            active_events=active_events,
            zones=seating_manager.get_zones()
        )

        # Calculate live throughput FPS
        processed_count += 1
        elapsed_now = max(0.001, time.perf_counter() - t_start)
        live_fps = round(processed_count / elapsed_now, 1)

        # Emit Common Envelope to real-time WebSocket queue
        envelope = CommonEnvelope(
            schema_version="2.0",
            job_id=job_id,
            video_id=video.id,
            analysis_module="examination",
            source_timestamp=timestamp_s,
            frame_index=frame_index,
            result_type="frame_observation",
            processing_status="RUNNING",
            payload={
                **frame_result.to_dict(),
                "throughput_fps": live_fps,
                "processing_mode": processing_mode
            }
        )
        _emit(job_id, envelope.to_dict())

        frame_index += 1

        # Periodic progress update to DB
        if frame_index % 15 == 0 or frame_index == total_frames:
            prog = min(99.0, round((frame_index / total_frames) * 100.0, 1))
            job.progress_pct = prog
            db.commit()

    cap.release()
    total_elapsed = max(0.001, time.perf_counter() - t_start)
    avg_fps = round(processed_count / total_elapsed, 2)
    avg_inf_ms = round(float(np.mean(inference_durations)) * 1000.0, 2) if inference_durations else 0.0

    if job.status != "CANCELLED":
        # Finalize and persist examination results
        job.status = "SUCCEEDED"
        job.stage = "Completed"
        job.progress_pct = 100.0
        job.completed_at = utcnow()

        # Persist person tracks
        for t in tracker.tracks:
            traj_data = [{"frame": h["frame"], "bbox": h["bbox"]} for h in t.history]
            trk_rec = ExaminationTrackRecord(
                job_id=job_id,
                track_id=t.track_id,
                entity_type="person",
                first_frame=t.first_frame,
                last_frame=t.last_frame,
                assigned_zone=t.assigned_zone,
                trajectory_json=traj_data
            )
            db.add(trk_rec)

        # Persist object tracks
        for ot in object_tracker.tracks:
            ot_traj = [{"frame": h["frame"], "bbox": h["bbox"], "conf": h["conf"]} for h in ot.history]
            ot_rec = ExaminationTrackRecord(
                job_id=job_id,
                track_id=ot.object_track_id,
                entity_type=f"object_{ot.object_class}",
                first_frame=ot.first_frame,
                last_frame=ot.last_frame,
                assigned_zone=f"person_{ot.associated_person_track_id}" if ot.associated_person_track_id else None,
                trajectory_json=ot_traj
            )
            db.add(ot_rec)

        # Persist all detected candidate events with full evidence details
        all_events = event_engine.get_all_events()
        for evt in all_events:
            evidence_details = {
                "supporting_objects": evt.supporting_objects,
                "priority": evt.priority,
                "evidence_strength": evt.evidence_strength,
                "detection_confidence": evt.detection_confidence,
                "association_confidence": evt.association_confidence,
                "event_confidence": evt.event_confidence,
                "uncertainty_reasons": evt.uncertainty_reasons,
                "primary_finding": evt.primary_finding,
                "supporting_observations": evt.supporting_observations,
                "visibility_warnings": evt.visibility_warnings,
                "associated_object_track_id": evt.associated_object_track_id,
                "associated_object_class": evt.associated_object_class,
                "supporting_frames": evt.supporting_frames
            }
            evt_rec = ExaminationEventRecord(
                event_id=evt.event_id,
                job_id=job_id,
                video_id=video.id,
                event_type=evt.event_type.value,
                start_frame=evt.start_frame,
                end_frame=evt.end_frame,
                start_time=evt.start_time,
                end_time=evt.end_time,
                person_track_id=evt.person_track_id,
                supporting_objects_json=evidence_details,
                confidence=evt.confidence,
                uncertainty=evt.uncertainty,
                description=evt.description,
                review_status=evt.review_status,
                reviewer_notes=evt.reviewer_notes
            )
            db.add(evt_rec)

        primary_evt = next((e for e in all_events if e.primary_finding), None)

        telemetry = {
            "analysis_module": "examination",
            "model_metadata": detector.get_model_metadata(),
            "processed_frames": processed_count,
            "total_frames": total_frames,
            "average_fps": avg_fps,
            "average_inference_ms": avg_inf_ms,
            "total_processing_time_s": round(total_elapsed, 2),
            "person_tracks_count": len(tracker.tracks),
            "object_tracks_count": len(object_tracker.tracks),
            "candidate_events_count": len(all_events),
            "primary_finding": primary_evt.to_dict() if primary_evt else None,
            "completed_at": job.completed_at.isoformat()
        }
        job.telemetry_json = telemetry
        db.commit()

        # Emit terminal completion payload
        completion_envelope = CommonEnvelope(
            schema_version="2.0",
            job_id=job_id,
            video_id=video.id,
            analysis_module="examination",
            source_timestamp=video.duration_sec,
            frame_index=processed_count,
            result_type="job_summary",
            processing_status="SUCCESS",
            confidence=0.92,
            payload={
                "type": "completed",
                "telemetry": telemetry,
                "primary_finding": primary_evt.to_dict() if primary_evt else None,
                "candidate_events": [e.to_dict() for e in all_events],
                "tracks_count": len(tracker.tracks)
            }
        )
        _emit(job_id, completion_envelope.to_dict())

    db.close()

