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
from visionradar.worker.job_worker import (
    _emit, is_job_cancelled, _ACTIVE_JOBS, _CANCELLED_JOBS, _JOB_LOCK,
    _JOB_COMPLETED_AT, cleanup_stale_job_queues
)

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)

def run_examination_job(job_id: int):
    """
    Independent worker pipeline for Examination Intelligence (Layer C).
    Executes person detection, object tracking, spatial person-object association,
    seating zone analysis, and evidence-based temporal candidate event tracking.
    Never mutates or accesses vehicle ByteTrack or traffic speed logic.
    """
    with _JOB_LOCK:
        _ACTIVE_JOBS.add(job_id)

    db: Session = SessionLocal()
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        print(f"[ExaminationPipeline] Error: Job #{job_id} not found.")
        _emit(job_id, {"type": "error", "job_id": job_id, "message": f"Job #{job_id} not found."})
        with _JOB_LOCK:
            _ACTIVE_JOBS.discard(job_id)
        db.close()
        return

    # Check for early cancellation before allocating pipeline resources
    if job.status == "CANCELLED" or is_job_cancelled(job_id):
        job.status = "CANCELLED"
        job.stage = "Cancelled"
        job.completed_at = utcnow()
        job.error_message = job.error_message or "Cancelled by user or superseded."
        db.commit()
        _emit(job_id, {
            "type": "error",
            "job_id": job_id,
            "message": job.error_message
        })
        with _JOB_LOCK:
            _ACTIVE_JOBS.discard(job_id)
            _CANCELLED_JOBS.discard(job_id)
        _JOB_COMPLETED_AT[job_id] = time.time()
        db.close()
        return

    cap = None
    try:
        video = db.query(Video).filter(Video.id == job.video_id).first()
        if not video or not os.path.exists(video.storage_path):
            job.status = "FAILED"
            job.stage = "Failed"
            job.completed_at = utcnow()
            job.error_message = f"Video file not found for Job #{job_id}"
            db.commit()
            _emit(job_id, {"type": "error", "job_id": job_id, "message": job.error_message})
            return

        # Extract user-configured seating zones and roles from job config
        config = job.config_json or {}
        configured_zones_raw = config.get("zones", [])
        configured_roles = config.get("roles", {})
        default_role = config.get("default_role", "UNKNOWN")
        podium_is_staff_zone = config.get("podium_is_staff_zone", True)

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
            job.stage = "Failed"
            job.completed_at = utcnow()
            job.error_message = f"Failed to open video capture for: {video.storage_path}"
            db.commit()
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
        first_detection_latency_s = None

        processing_mode = config.get("processing_mode", "balanced")
        if processing_mode == "high_speed":
            stride = 4
        elif processing_mode == "forensic":
            stride = 1
        else:
            stride = 3

        # Processing loop
        while cap.isOpened():
            if is_job_cancelled(job_id):
                job.status = "CANCELLED"
                job.stage = "Cancelled"
                job.completed_at = utcnow()
                job.error_message = "Cancelled by user or superseded."
                db.commit()
                _emit(job_id, {
                    "type": "error",
                    "job_id": job_id,
                    "message": "Cancelled by user or superseded."
                })
                break

            is_det_frame = (frame_index % stride == 0)

            # Phase E Performance: Use cap.grab() on non-detection frames to skip full RGB matrix decoding
            if is_det_frame:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            else:
                ret = cap.grab()
                if not ret:
                    break
                frame = None

            timestamp_s = frame_index / fps if fps > 0 else (frame_index / 30.0)

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

            if first_detection_latency_s is None and (len(active_persons) > 0 or len(tracked_objects) > 0):
                first_detection_latency_s = time.perf_counter() - t_start

            # Associate seating zones
            for p in active_persons:
                seating_manager.associate_person(p)

            # Apply configurable role mapping (configured roles > staff zones > default role > UNKNOWN)
            for p in active_persons:
                trk_id_str = str(p.track_id)
                if trk_id_str in configured_roles:
                    r_entry = configured_roles[trk_id_str]
                    if isinstance(r_entry, dict):
                        p.assigned_role = r_entry.get("role", "UNKNOWN")
                        p.role_source = r_entry.get("source", "CONFIG")
                    else:
                        p.assigned_role = str(r_entry)
                        p.role_source = "CONFIG"
                elif p.assigned_zone and podium_is_staff_zone:
                    matching_zone = next((z for z in seating_manager.get_zones() if z.zone_id == p.assigned_zone), None)
                    if matching_zone and matching_zone.zone_type == ZoneType.RESTRICTED_PODIUM:
                        p.assigned_role = "INVIGILATOR"
                        p.role_source = "ZONE_MAPPING"
                elif default_role != "UNKNOWN" and p.assigned_role == "UNKNOWN":
                    p.assigned_role = default_role
                    p.role_source = "DEFAULT"

                # Mirror role to underlying tracker track
                underlying_t = next((t for t in tracker.tracks if t.track_id == p.track_id), None)
                if underlying_t:
                    underlying_t.assigned_role = p.assigned_role
                    underlying_t.role_source = p.role_source

            # Evaluate sequential observable events (strictly excludes authorized staff from student cheating candidates)
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

            # Periodic progress update to DB (throttled to every 60 frames to eliminate SQLite sync latency)
            if frame_index % 60 == 0 or frame_index == total_frames:
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

            # Persist person tracks with verified role metadata across the entire video duration
            person_tracks_to_save = list(tracker.all_tracks.values()) if hasattr(tracker, 'all_tracks') and tracker.all_tracks else tracker.tracks
            for t in person_tracks_to_save:
                if not t.history:
                    continue
                t.history.sort(key=lambda x: x["frame"])
                traj_data = [{"frame": h["frame"], "bbox": h["bbox"]} for h in t.history]
                trk_rec = ExaminationTrackRecord(
                    job_id=job_id,
                    track_id=t.track_id,
                    entity_type="person",
                    first_frame=t.first_frame,
                    last_frame=t.last_frame,
                    assigned_zone=t.assigned_zone,
                    assigned_role=t.assigned_role,
                    role_source=t.role_source,
                    trajectory_json=traj_data
                )
                db.add(trk_rec)

            # Persist object tracks across the entire video duration
            object_tracks_to_save = list(object_tracker.all_tracks.values()) if hasattr(object_tracker, 'all_tracks') and object_tracker.all_tracks else object_tracker.tracks
            for ot in object_tracks_to_save:
                if not ot.history:
                    continue
                ot.history.sort(key=lambda x: x["frame"])
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
                    assigned_role=evt.assigned_role,
                    supporting_objects_json=evidence_details,
                    confidence=evt.confidence,
                    uncertainty=evt.uncertainty,
                    description=evt.description,
                    review_status=evt.review_status,
                    reviewer_notes=evt.reviewer_notes
                )
                db.add(evt_rec)

            primary_evt = next((e for e in all_events if e.primary_finding), None)

            detected_count = sum(1 for i in range(processed_count) if i % stride == 0)
            skipped_count = processed_count - detected_count

            telemetry = {
                "analysis_module": "examination",
                "model_metadata": detector.get_model_metadata(),
                "source_video_fps": round(fps, 2),
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "detected_frames": detected_count,
                "skipped_frames": skipped_count,
                "stride": stride,
                "processing_mode": processing_mode,
                "average_fps": avg_fps,
                "end_to_end_fps": avg_fps,
                "average_inference_ms": avg_inf_ms,
                "detector_inference_ms": avg_inf_ms,
                "first_detection_latency_s": round(first_detection_latency_s, 3) if first_detection_latency_s is not None else None,
                "total_processing_time_s": round(total_elapsed, 2),
                "person_tracks_count": len(person_tracks_to_save),
                "object_tracks_count": len(object_tracks_to_save),
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
                    "status": "SUCCEEDED",
                    "telemetry": telemetry,
                    "processing_fps": avg_fps,
                    "throughput_fps": avg_fps,
                    "first_detection_latency_s": round(first_detection_latency_s, 3) if first_detection_latency_s is not None else None,
                    "primary_finding": primary_evt.to_dict() if primary_evt else None,
                    "candidate_events": [e.to_dict() for e in all_events],
                    "tracks_count": len(person_tracks_to_save)
                }
            )
            _emit(job_id, completion_envelope.to_dict())

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            db.rollback()
            job.status = "FAILED"
            job.stage = "Failed"
            job.completed_at = utcnow()
            job.error_message = f"{str(e)}\n{tb}"
            db.commit()
        except Exception:
            pass
        _emit(job_id, {"type": "error", "job_id": job_id, "message": str(e)})
        print(f"[ExaminationPipeline] Job #{job_id} failed with error: {e}", flush=True)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        with _JOB_LOCK:
            _ACTIVE_JOBS.discard(job_id)
            _CANCELLED_JOBS.discard(job_id)
        _JOB_COMPLETED_AT[job_id] = time.time()
        cleanup_stale_job_queues(ttl_sec=30.0)
        try:
            db.close()
        except Exception:
            pass

