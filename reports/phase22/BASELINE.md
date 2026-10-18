# Phase 22 Baseline Audit: Individual Vehicle Speed Intelligence

**Date:** 2026-10-15  
**Environment:** Python 3.14.4, Windows, FastAPI, React + Vite + TypeScript  
**Status:** Pre-Implementation Baseline Established  

---

## 1. Baseline Test Execution & Results

### Suite A: Speed & Tracking Reliability
- **Command:** python -m pytest tests/test_speed.py tests/test_speed_phase7.py tests/test_phase20_tracking_reliability.py
- **Result:** 23 passed in 4.77s
- **Pass Rate:** 100% (23/23)
- **Coverage:** MonocularSpeedEstimator, SpeedProvenance, Homography calibration safety, ByteTrack trajectory continuity, Multi-Estimator calculations.

### Suite B: Core API & Pipeline Infrastructure
- **Command:** python -m pytest tests/test_api.py tests/test_cv_pipeline.py tests/test_detector_registry.py
- **Result:** 7 passed in 14.39s (8 deprecation warnings from Pydantic v2 and FastAPI on_event)
- **Pass Rate:** 100% (7/7)
- **Coverage:** FastAPI REST job endpoints, YOLOX-Nano ONNX detector registry, End-to-end CV pipeline execution.

### Suite C: Frontend Production Build
- **Command:** cd apps/web && npm run build
- **Result:** 	sc && vite build completed in 8.68s.
- **Assets:** dist/assets/index-Bw2RxVV4.js (217.22 kB), 37 modules transformed with 0 errors.

---

## 2. Architecture & File Responsibility Mapping

1. **Video Upload & Job Scheduling:**
   - pps/api/main.py: POST /api/v1/projects/{project_id}/videos, POST /api/v1/videos/{video_id}/jobs
   - packages/visionradar/models/entities.py: Video, ProcessingJob, Track, SpeedMeasurement
2. **Video Decoding & Detection:**
   - packages/visionradar/cv/decoder.py: VideoDecoder OpenCV frame iterator
   - packages/visionradar/cv/detector.py: LightweightDetector / YOLOXDetector (ONNX Runtime)
3. **Tracking & Trajectory Modeling:**
   - packages/visionradar/cv/tracker.py: ByteTrackTracker persistent track state and Kalman filtering
   - packages/visionradar/cv/trajectory.py: VehicleTrajectory, TrajectoryPoint
4. **Calibration & Homography:**
   - packages/visionradar/cv/calibration.py: HomographyCalibrator with numerical stability checks and ROI bounding
5. **Speed Estimation Engine:**
   - packages/visionradar/cv/speed.py: MonocularSpeedEstimator, SpeedProvenance, SpeedEstimate
6. **Worker Loop & Real-Time Delivery:**
   - packages/visionradar/worker/job_worker.py: Async worker emitting rame_result packets over WebSocket
   - Real-time payload emits 	rack_id, ehicle_class, box, speed_kmh, speed_status, vg_speed_kmh, max_speed_kmh, min_speed_kmh
7. **Frontend Visualization & Workbench:**
   - pps/web/src/components/WorkbenchView.tsx: Canvas bounding box rendering, telemetry HUD, Track Inspector
   - Target for Phase 22: Add live Synchronized Vehicle Table and refined speed overlay badges.

---

## 3. Existing User Git Modifications

Git inspection revealed active, working changes on:
- pps/api/main.py
- pps/web/src/components/WorkbenchView.tsx
- packages/visionradar/cv/tracker.py
- packages/visionradar/worker/job_worker.py
- scripts/start_server.py

**Preservation Mandate:** None of these existing working features will be discarded or overwritten. Phase 22 will add strictly non-breaking, minimal extensions to WorkbenchView.tsx and ensure data flows seamlessly.

---

## 4. Current Limitations & Known State
- When no calibration is provided for an uploaded video, metric speed is correctly withheld with status CALIBRATION_REQUIRED or OUT_OF_ROI.
- In demo/synthetic mode, synthetic vehicles V are rendered; in real video mode, actual YOLOX + ByteTrack detections are rendered.
