# VisionRadar — Phase 7A Speed Pipeline Forensic Audit Report

**Date:** 2026-09-26  
**Auditor:** Antigravity AI Perception Team  
**Scope:** Complete End-to-End Speed Pipeline Architecture Inspection  

---

## 1. Executive Summary

This forensic audit traces VisionRadar's end-to-end vehicle speed estimation pipeline from video input frames down to database telemetry and frontend UI rendering.

```
Video Input (MP4/RTSP)
  ↓
Frame & Monotonic Timestamp (time.perf_counter())
  ↓
YOLOX-Nano ONNX Bounding Box Detection (640x640 Letterbox)
  ↓
ByteTrack Multi-Object Tracker (Track ID assignment)
  ↓
Track Anchor / Contact Point (Bottom-center (cx, y2) pixel coord)
  ↓
Calibration Polygon ROI Membership (pointPolygonTest)
  ↓
DLT Homography Transformation H (Image (u,v) → World (X,Y) meters)
  ↓
Contiguous Segment Filtering (OUT → VALID → OUT Isolation)
  ↓
Multi-Estimator Speed Engine (Instantaneous, Path Average, Regression, Kalman)
  ↓
Estimator Disagreement & Heuristic Uncertainty Calculation
  ↓
Perception Quality Engine Track & Trajectory Validation
  ↓
API Serialization & Frontend Track Inspector Rendering
  ↓
Database Telemetry Persistence
```

---

## 2. Detailed Pipeline Component Audit

| Pipeline Stage | Implementation Module | Source File | Technical Details & Execution Logic |
| :--- | :--- | :--- | :--- |
| **1. Video Capture & Timestamps** | `VideoSource` | `packages/visionradar/perception/capture.py` | Reads OpenCV video frames, attaches high-resolution monotonic timestamps (`time.perf_counter()`). |
| **2. Bounding Box Detection** | `InferenceEngine` & `postprocess` | `packages/visionradar/perception/postprocess.py` | Production YOLOX-Nano ONNX ($640\times 640$ letterbox, conf=0.25, NMS=0.45). |
| **3. Tracking & Anchor Selection** | `PerceptionTracker` & `VehicleTrajectory` | `packages/visionradar/cv/trajectory.py` | ByteTrack assigns track IDs; bottom-center box midpoint $(cx, y2)$ selected as road contact anchor. |
| **4. Calibration & Homography** | `HomographyCalibrator` | `packages/visionradar/cv/calibration.py` | 4-point DLT matrix $H$ maps $(u,v) \to (X,Y)$ meters. Checks polygon ROI and denominator stability ($|denom| \ge 0.05$). |
| **5. Segment Reset & Contamination Removal** | `MonocularSpeedEstimator` | `packages/visionradar/cv/speed.py` | Filters `OUT_OF_ROI` points. Resets Kalman/regression state across contiguous valid segments. |
| **6. Speed Engine & Uncertainty** | `MultiEstimatorSpeedEngine` | `packages/visionradar/cv/speed.py` | Evaluates Instantaneous, Path-Average, Regression, and Kalman methods; calculates estimator agreement and heuristic uncertainty bounds. |
| **7. Perception Quality Integration** | `PerceptionQualityEngine` | `packages/visionradar/perception/quality_engine.py` | Flags spatial position jumps (>150 px/frame) and low-confidence dropouts. |
| **8. API & Database Telemetry** | `JobWorker` & FastAPI | `packages/visionradar/worker/job_worker.py` | Serializes speed provenance, agreement, and validity to SQLite DB and WebSocket stream. |
| **9. Frontend UI Transparency** | `WorkbenchView` | `apps/web/src/components/WorkbenchView.tsx` | Renders speed provenance modal, method badge, uncertainty bounds, and trajectory quality. |

---

## 3. Audited Baseline Strengths & Identified Limitations

### Audited Strengths:
1. **Timestamp Precision**: Uses true frame-level monotonic timestamps rather than assuming fixed 30 FPS.
2. **Bottom-Center Contact Point**: Anchor pixel $(cx, y2)$ properly reflects vehicle-ground plane contact.
3. **Homography Stability Check**: Checks condition number and denominator magnitude to prevent projection explosions.

### Identified Limitations for Phase 7 Fixes:
1. **Estimator Provenance**: Legacy code selected a single speed method without exposing estimator disagreement.
2. **Contamination across ROI Boundaries**: Transitioning `OUT_OF_ROI -> VALID -> OUT_OF_ROI` required explicit segment resets to avoid stale state poisoning.
3. **Uncertainty Nomenclature**: Legacy bounds were labeled as statistical confidence intervals despite being heuristic approximations.
4. **Ground-Truth Availability**: Raw radar GT files in `BrnoCompSpeed-master` are **NOT LOCALLY AVAILABLE**. Speed accuracy validation is explicitly marked as `BLOCKED BY GROUND TRUTH`.
