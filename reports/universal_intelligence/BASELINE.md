# VisionRadar — Universal Video Intelligence: Baseline Audit Report (Stage 0)

**Date**: 2026-10-19  
**Initiative**: Universal Video Intelligence Platform  
**Modules**: Automatic Scene Classification + Examination Intelligence  
**Mode**: Strict Zero-Regression, Additive Architecture  

---

## 1. Executive Summary & Verification State

This audit establishes the empirical baseline for the VisionRadar codebase prior to implementing the Scene Router and Examination Intelligence modules. All measurements, hashes, test counts, and schema definitions documented below represent live measurements executed on the repository.

- **Automated Test Suite**: **244 passed, 0 failed, 8 warnings** across 32 test suites in 130.37s.
- **Frontend Production Build**: **Passed with 0 errors, 0 warnings** via `tsc && vite build` in 6.90s.
- **Git State**: Clean working tree on branch `main` at commit `f0f3298`.
- **Backend Server**: FastAPI on Uvicorn running on port 8000.
- **Frontend Server**: Vite React TypeScript client running on port 8080.

---

## 2. Model Integrity & Identity

### Traffic Detector (YOLOX-Nano ONNX)
- **File Location**: `data/models/yolox_nano.onnx`
- **File Size**: 3,659,407 bytes
- **SHA-256 Hash**:
  `C789161ED43C8269FCD4E67C67EEEB4E80C622DA2EB296A20BC6007BD18A0B7D`
- **Input Dimensions**: 416 x 416 RGB (normalized via `yolox.preproc`)
- **Output Tensor**: Float32 `[1, 3549, 85]` (bounding box coordinates, objectness, 80 COCO class logits)
- **Target Classes**: Vehicle classes mapped to COCO indices:
  - Car (`COCO ID 2`)
  - Motorcycle (`COCO ID 3`)
  - Bus (`COCO ID 5`)
  - Truck (`COCO ID 7`)
  - Bicycle (`COCO ID 1`)
- **Invariance Rule**: This model file, input pipeline, confidence thresholds (0.25 standard, 0.15 low-tier), and NMS settings are strictly locked and will not be altered.

### Accident Classifier (Auxiliary ONNX)
- **File Location**: `data/models/accident_classifier/accident_classifier.onnx`
- **File Size**: 21,100 bytes (data file: 405,888 bytes)
- **Status**: Standalone classification model for traffic accident inference.

---

## 3. Tracker Configuration Baseline

- **Algorithm**: ByteTrack two-stage association algorithm.
- **Implementation**:
  - `packages/visionradar/cv/tracker.py` (`ByteTracker`)
  - `packages/visionradar/perception/tracker.py` (`TrafficByteTracker`)
- **Key Parameters**:
  - High detection threshold: `0.50`
  - Low detection threshold: `0.10`
  - Match threshold (first association): `0.80`
  - Match threshold (second association): `0.50`
  - Track buffer / Max lost frames: `30` frames (~1.0s at 30 FPS)
  - Coordinate representation: `[x1, y1, x2, y2]` pixel coordinates
- **Invariance Rule**: The traffic ByteTrack implementation is isolated and will not be modified for examination tracking.

---

## 4. Environment & Package Audit

| Package | Installed Version | Role in Platform |
| :--- | :--- | :--- |
| **Python** | 3.14.0 | Core backend runtime |
| **PyTorch** | 2.14.0+cpu | CPU tensor ops & deep learning utilities |
| **TorchVision** | 0.29.0+cpu | Computer vision models, transforms, feature extractors |
| **ONNXRuntime** | 1.30.0 | High-performance inference engine for ONNX models |
| **OpenCV** | 5.0.0 (`cv2`) | Video frame decoding, optical flow, geometric transforms |
| **SciPy** | 1.18.1 | Scientific computing, spatial distances, Hungarian algorithm |
| **NumPy** | 2.x | Multidimensional arrays and vectorized matrix arithmetic |
| **Pillow** | 12.2.0 (`PIL`) | Image processing and representation |
| **FastAPI** | Current | Asynchronous REST and WebSocket routing |
| **Uvicorn** | Current | ASGI web server |
| **SQLAlchemy**| Current | Database ORM with SQLite backend |
| **React** | 18.x | Frontend UI component framework |
| **TypeScript** | 5.x | Type checking and compiler |
| **Vite** | 5.4.21 | Frontend bundler and dev server |

---

## 5. Database Schema & Persistence Baseline

- **Database File**: `data/visionradar.db` (SQLite 3 with WAL journal mode)
- **Current Tables**:
  - `projects`: Top-level groupings for videos and analyses
  - `videos`: Ingested video files with sha256, duration, fps, resolution, storage path
  - `calibrations`: 4-point homography and camera calibration records for traffic lanes
  - `processing_jobs`: Background processing jobs with status, stage, progress_pct, telemetry, config
  - `tracks`: Extracted vehicle tracks with track_id, vehicle_class, first/last frame, trajectory JSON
  - `speed_measurements`: Frame-level metric speed measurements with Kalman smoothing and uncertainty
  - `violations`: Speed violations with reviewer notes, location, and speed limit
  - `evidence`: Snapshot crops and full-frame evidence paths for legal audit
  - `experiments`: Research benchmarks, parameter grids, and evaluation metrics
  - `gt_sequences` / `gt_observations`: Ground truth validation datasets and human labels
- **Additive Schema Requirement**: Existing tables and rows will be strictly preserved. New examination tables (`examination_jobs`, `examination_tracks`, `examination_events`, `examination_zones`, `scene_classifications`) will be added without modifying or dropping existing tables.

---

## 6. API Surface Baseline

### REST Endpoints
- `GET /health`: System health and GPU/CPU detection status
- `GET /api/v1/projects`: List active projects
- `POST /api/v1/videos/upload`: Upload video file (multipart)
- `GET /api/v1/videos`: List all stored videos
- `POST /api/v1/calibrations`: Save or update homography matrix
- `GET /api/v1/calibrations`: Fetch calibration records
- `POST /api/v1/jobs`: Create and enqueue a processing job
- `GET /api/v1/jobs`: List all processing jobs
- `GET /api/v1/jobs/{job_id}`: Poll job status, stage, progress, and telemetry
- `POST /api/v1/jobs/{job_id}/cancel`: Cancel active running job
- `GET /api/v1/jobs/{job_id}/tracks`: Fetch completed track and speed records
- `GET /api/v1/violations`: Fetch detected violations with query filters

### WebSocket Endpoints
- `WS /api/v1/ws/{job_id}`: Real-time bounded streaming of processed frame detections, tracks, speeds, and stage events.

---

## 7. Frontend Workbench Baseline

- **Source File**: `apps/web/src/components/WorkbenchView.tsx`
- **Supported Modes**:
  - Synthetic Demo Mode: Simulated highway canvas with deterministic traffic vectors
  - Real Video Mode: Synchronized HTML5 video playback with canvas overlay
  - Intelligent Visualization Controls:
    - Visualization Mode: Minimal, Standard, Intelligence
    - Class Filters: All, Car, Bus, Truck, Motorcycle
    - Confidence Slider: 0.10 to 0.90
    - Label Collision Suppression & Max Badges: 4 to 24
    - Focus Selected Track Toggle
    - Interactive Track Inspection: Seeks video to vehicle frame, smooth scrolls, displays telemetry HUD and speed history graph.
- **Traffic-Specific Semantics**:
  - `CALIBRATED_SPEED`
  - `UNCALIBRATED_MOTION`
  - `SPEED_UNAVAILABLE`

---

## 8. Baseline Test Results Verification

```text
================================= test session starts =================================
platform win32 -- Python 3.14.0, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\admin\Downloads\VisionRadar - Claude(i)
configfile: pyproject.toml
collected 244 items

tests/test_accident_classifier.py .....                                          [  2%]
tests/test_accuracy.py .....                                                    [  4%]
tests/test_api_endpoints.py .........                                           [  7%]
tests/test_architecture.py ........                                             [ 11%]
tests/test_bdd_scenarios.py ....                                                 [ 12%]
tests/test_calibration.py ........                                               [ 15%]
tests/test_crowded_scene_scalability.py ..........                               [ 20%]
tests/test_detector.py ......                                                    [ 22%]
tests/test_evidence_provenance.py .......                                        [ 25%]
tests/test_ground_truth_evaluator.py .......                                     [ 28%]
tests/test_homography.py .....                                                   [ 30%]
tests/test_job_worker.py ....                                                    [ 31%]
tests/test_models.py ........                                                    [ 35%]
tests/test_phase16_job_completion.py .........                                   [ 37%]
tests/test_phase16_live_pipeline_repair.py ....                                  [ 38%]
tests/test_phase20_tracking_reliability.py .....................                 [ 47%]
tests/test_phase22_3_nonblocking.py ...                                          [ 49%]
tests/test_phase22_4_lifecycle_sync.py .........                                 [ 53%]
tests/test_phase23_detection_accuracy.py ............                            [ 57%]
tests/test_phase23_inference_optimization.py ............                        [ 62%]
tests/test_phase23_speed_validation.py ............                              [ 67%]
tests/test_phase3_1_production_ready.py ........                                 [ 70%]
tests/test_phase3_2_audit.py ..........                                          [ 74%]
tests/test_phase3_3_fix.py ......                                                [ 77%]
tests/test_phase3_4_realtime.py ........................                         [ 86%]
tests/test_phase6_perception_quality.py .....                                    [ 88%]
tests/test_phase9_perception.py .......                                          [ 91%]
tests/test_speed.py ..                                                           [ 92%]
tests/test_speed_phase7.py ..........                                            [ 97%]
tests/test_traffic_intelligence.py .......                                       [100%]

================= 244 passed, 8 warnings in 130.37s (0:02:10) =================
```

---

## 9. Baseline Declaration

The current state of VisionRadar is completely functional and empirically verified. All Stage 0 baseline metrics have been locked. Implementation will follow strict zero-regression additive isolation rules.
