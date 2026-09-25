import time
import datetime
import traceback
import numpy as np
from sqlalchemy.orm import Session
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Track, SpeedMeasurement, Violation, Evidence, Video, Calibration
from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detector import LightweightDetector
from visionradar.cv.tracker import ByteTrackTracker
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.speed import MonocularSpeedEstimator
from visionradar.cv.violations import ViolationRuleEngine
from visionradar.cv.evidence import EvidenceWriter

def run_job(job_id: int):
    """
    Executes a VisionRadar video processing job.
    Stage A: Decode video, detect vehicles, track identities, build raw trajectories.
    Stage B: Project via calibration, compute timestamp-aware speed & uncertainty, evaluate violations.
    """
    db: Session = SessionLocal()
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        print(f"Job #{job_id} not found.")
        db.close()
        return

    try:
        job.status = "RUNNING"
        job.started_at = datetime.datetime.now(datetime.timezone.utc)
        job.progress_pct = 5.0
        job.stage = "Decoder"
        db.commit()

        video = db.query(Video).filter(Video.id == job.video_id).first()
        if not video:
            raise RuntimeError(f"Associated video #{job.video_id} not found.")

        # Load Calibration
        calib = None
        if job.calibration_id:
            calib = db.query(Calibration).filter(Calibration.id == job.calibration_id).first()

        if calib:
            calibrator = HomographyCalibrator(
                H_matrix=np.array(calib.h_matrix_json),
                image_points=calib.image_points_json,
                world_points=calib.world_points_json,
                camera_height=calib.camera_height_m,
                pitch_deg=calib.pitch_deg
            )
        else:
            # Default fallback calibration
            img_pts = [(330.0, 160.0), (470.0, 160.0), (748.0, 435.0), (51.0, 435.0)]
            world_pts = [(-1.0, 150.0), (13.0, 150.0), (13.0, 0.5), (-1.0, 0.5)]
            calibrator = HomographyCalibrator(image_points=img_pts, world_points=world_pts)

        decoder = VideoDecoder(video.storage_path)
        meta = decoder.get_metadata()

        job.stage = "Detection & Tracking"
        job.progress_pct = 20.0
        db.commit()

        from visionradar.cv.detection import get_detector
        from visionradar.intelligence.traffic import (
            VehicleCountingEngine, TrafficFlowEngine, LaneIntelligenceEngine,
            VehicleDensityEngine, QueueDetector, CongestionEngine
        )
        from visionradar.intelligence.events import TrafficEventEngine

        detector_type = (job.config_json or {}).get("detector", "yolox")
        detector = get_detector(detector_type, model_path="data/models/yolox_nano.onnx")
        tracker = ByteTrackTracker()
        speed_estimator = MonocularSpeedEstimator(calibrator=calibrator, fps=meta.fps)
        rule_engine = ViolationRuleEngine(speed_limit_kmh=80.0)
        evidence_writer = EvidenceWriter(output_dir=f"data/evidence/job_{job.id}")

        # Instantiate Intelligence Engines
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

        t_job_start = time.time()
        active_tracks = {}
        total_frames = max(1, meta.total_frames)

        last_frame_idx = 0
        last_timestamp = 0.0

        # STAGE A & B Execution
        for frame_idx, timestamp, frame in decoder.decode_frames():
            last_frame_idx = frame_idx
            last_timestamp = timestamp
            dets = detector.detect(frame)
            tracks = tracker.update(dets, frame_idx, timestamp)

            active_track_dicts = []
            src_w = 1920.0 if meta.width < 1000 else float(meta.width)
            src_h = 1080.0 if meta.height < 600 else float(meta.height)
            for trk in tracks:
                active_tracks[trk.track_id] = trk
                speed_estimator.project_trajectory(trk.trajectory, source_width=src_w, source_height=src_h)
                est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=frame_idx)
                if est is not None and est.validity == "VALID":
                    trk.speed_kmh = est.smoothed_kmh
                    trk.speed_uncertainty_kmh = est.uncertainty_kmh
                else:
                    trk.speed_kmh = None
                    trk.speed_uncertainty_kmh = None

                    # Save candidate violations
                    viol = rule_engine.evaluate_track(
                        track_id=trk.track_id,
                        vehicle_class=trk.vehicle_class,
                        frame_index=frame_idx,
                        timestamp=timestamp,
                        estimate=est
                    )
                    if viol:
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

                # Update counting engine
                counting_engine.process_track_update(
                    trk.track_id, trk.vehicle_class, [p.to_dict() for p in trk.trajectory.points]
                )

                active_track_dicts.append(trk.to_dict())

            # Evaluate Instantaneous Density & Occupancy
            density_engine.update_frame_occupancy(frame_idx, timestamp, active_track_dicts, calibrator)

            # Evaluate Queues & Event State Machines per frame
            active_queues = queue_detector.update(active_track_dicts, frame_idx, timestamp)
            event_engine.process_frame(active_track_dicts, frame_idx, timestamp)

            if frame_idx % 15 == 0:
                job.progress_pct = min(95.0, 20.0 + (frame_idx / total_frames) * 75.0)
                db.commit()

        # Finalize remaining active temporal events at end of video
        event_engine.finalize(last_frame_idx, last_timestamp)

        # Compute Final Intelligence Metrics
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

        # Save tracks and speed measurements
        job.stage = "Persisting Trajectories"
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

            for p in trk.trajectory.points:
                est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=p.frame_index)
                if est and est.validity in ("VALID", "LOW_CONFIDENCE") and est.smoothed_kmh is not None and est.smoothed_kmh > 0:
                    db_sm = SpeedMeasurement(
                        track_id=db_trk.id,
                        frame_index=p.frame_index,
                        timestamp=p.timestamp,
                        instantaneous_kmh=est.instantaneous_kmh,
                        smoothed_kmh=est.smoothed_kmh,
                        uncertainty_kmh=est.uncertainty_kmh,
                        confidence_low_kmh=est.confidence_low_kmh,
                        confidence_high_kmh=est.confidence_high_kmh,
                        error_components_json={**(est.error_components or {}), "validity": est.validity}
                    )
                    db.add(db_sm)
                elif est:
                    db_sm = SpeedMeasurement(
                        track_id=db_trk.id,
                        frame_index=p.frame_index,
                        timestamp=p.timestamp,
                        instantaneous_kmh=None,
                        smoothed_kmh=None,
                        uncertainty_kmh=None,
                        confidence_low_kmh=None,
                        confidence_high_kmh=None,
                        error_components_json={"validity": est.validity}
                    )
                    db.add(db_sm)

        t_job_end = time.time()
        job_duration_sec = max(0.001, t_job_end - t_job_start)

        # Assemble Forensic Telemetry Package
        det_telemetry = detector.get_telemetry() if hasattr(detector, 'get_telemetry') else {}
        track_durations = [len(t.trajectory.points) for t in active_tracks.values()] if active_tracks else []
        tracker_metrics = tracker.get_tracking_metrics() if hasattr(tracker, 'get_tracking_metrics') else {}
        calib_validation = calibrator.validate_calibration() if calibrator else {"is_valid": False}
        
        telemetry = {
            **det_telemetry,
            "video_id": job.video_id,
            "processing_run_id": f"run_{job.id}_{int(t_job_start)}",
            "start_timestamp": 0.0,
            "end_timestamp": meta.duration_sec,
            "duration_s": meta.duration_sec,
            "calibration_id": job.calibration_id or 1,
            "lane_configuration_id": "default_main_roadway",
            "policy_version": "v1.0.0",
            "video_resolution": [meta.width, meta.height],
            "video_fps": meta.fps,
            "video_duration_sec": meta.duration_sec,
            "frames_processed": total_frames,
            "processing_time_sec": round(job_duration_sec, 2),
            "average_fps": round(total_frames / job_duration_sec, 1),
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
            }
        }
        job.telemetry_json = telemetry

        job.status = "SUCCEEDED"
        job.stage = "Complete"
        job.progress_pct = 100.0
        job.completed_at = datetime.datetime.now(datetime.timezone.utc)
        db.commit()
        print(f"Job #{job_id} completed successfully with detector '{telemetry.get('detector_name')}'.")

    except Exception as e:
        db.rollback()
        job.status = "FAILED"
        job.error_message = str(e) + "\n" + traceback.format_exc()
        db.commit()
        print(f"Job #{job_id} failed: {e}")
    finally:
        db.close()

def poll_worker_loop(interval_sec: float = 2.0):
    """
    Continuous background polling loop checking for QUEUED jobs.
    """
    print("VisionRadar Background Worker loop started...")
    while True:
        db: Session = SessionLocal()
        job = db.query(ProcessingJob).filter(ProcessingJob.status == "QUEUED").first()
        db.close()

        if job:
            print(f"Worker picked up job #{job.id}...")
            run_job(job.id)
        else:
            time.sleep(interval_sec)

if __name__ == "__main__":
    poll_worker_loop()
