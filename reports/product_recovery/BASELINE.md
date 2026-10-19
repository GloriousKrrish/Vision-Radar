# VisionRadar Product Recovery — Baseline Audit

## 1. Executive Summary & Objective

VisionRadar is undergoing a comprehensive product recovery and engineering audit to transform the prototype into a reliable, production-oriented universal vision intelligence platform.

**Core Philosophy**:
`RELIABILITY > ACCURACY > PERFORMANCE > NEW FEATURES`

Before any code modification or system expansion is undertaken, this document establishes the empirical baseline of the repository, accounting for all existing user modifications, test metrics, active pipelines, and architectural boundaries.

---

## 2. Repository Architecture & Component Map

| Layer | Component | Tech Stack | Location | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Edge / Model Ingestion** | Object Detection | YOLOX-Nano ONNX Runtime (`yolox_nano.onnx`) | `packages/visionradar/cv/detection/yolox.py` | Active, CPU-optimized |
| **Object Tracking** | Multi-Object Association | ByteTrack Algorithm (`ByteTrack`) | `packages/visionradar/cv/tracker.py` | Active, frame-by-frame ID persistence |
| **Geometry & Scale** | Planar Calibration | 4-point DLT Homography (`cv2.findHomography`) | `packages/visionradar/cv/calibration.py` | Enforces condition number & RMSE |
| **Metric Engine** | Speed & Trajectory | Monocular perspective projection + rolling window | `packages/visionradar/cv/speed.py`, `trajectory.py` | Tri-state: `CALIBRATED_SPEED`, `UNCALIBRATED_MOTION`, `SPEED_UNAVAILABLE` |
| **Worker / Queue** | Job Worker | Threading + Asyncio Queue + SQLite WAL | `packages/visionradar/worker/job_worker.py` | Active-job cancellation, safe WebSocket backpressure |
| **Backend API** | REST + WebSocket | FastAPI, Pydantic, SQLAlchemy 2.0 | `apps/api/main.py`, `schemas.py` | Endpoints for videos, jobs, calibrations, tracks, violations |
| **Frontend UI** | Interactive Workbench | React 18, Vite, TypeScript, Canvas 2D | `apps/web/src/components/WorkbenchView.tsx` | Decoupled 60 FPS playback, minimal/standard/intelligence modes |
| **Database** | Persistent Storage | SQLite with Write-Ahead Logging (`WAL`) | `packages/visionradar/models/entities.py`, `data/visionradar.db` | Normalized schema with foreign keys and cascade deletions |

---

## 3. Git Status & Preservation Integrity

- **Branch**: `main`
- **Tracked Modifications**:
  - `apps/api/main.py`: Cancellation endpoint `/jobs/{id}/cancel` and server lifecycle cleanup.
  - `apps/web/src/components/WorkbenchView.tsx`: Stale box elimination on seek, atomic buffer transition, crowd controls, minimal/standard/intelligence modes.
  - `packages/visionradar/cv/calibration.py`: Homography condition number checks and road plane sanity validation.
  - `packages/visionradar/cv/speed.py`: Calibration requirement checks and speed error boundaries.
  - `packages/visionradar/models/entities.py`: Database schema definitions.
  - `packages/visionradar/worker/job_worker.py`: Non-blocking cancellation check, lifecycle events, safe WebSocket queue backpressure.
  - `requirements.txt`: Locked dependencies.
- **Preservation Policy**: Zero deletions of working functionality; all existing user modifications and data are preserved.

---

## 4. Empirical Baseline Metrics

### 4.1 Automated Test Suite
- **Command**: `python -m pytest tests/`
- **Total Tests**: **244 passed, 0 failed** (100% pass rate across 32 test files)
- **Execution Time**: **101.91 seconds**
- **Test Modules**: Covers API, CV pipeline, detector registry, real video E2E, homography, perception engine, ground-truth speed, traffic intelligence, anomaly intelligence, worker lifecycle (scenarios A–H), and track persistence.

### 4.2 Frontend Production Build
- **Command**: `npm run build` (in `apps/web`)
- **Typecheck & Bundler**: TypeScript (`tsc`) + Vite production bundler
- **Result**: **0 errors, 0 warnings**
- **Build Time**: **5.30 seconds**
- **Bundle Size**: `index.html` (0.78 kB), CSS (9.89 kB), JS (244.90 kB gzip: 74.37 kB).

### 4.3 Runtime Throughput & Latency (Empirically Measured)
- **Video Playback Rate**: **58 – 60 FPS** (decoupled via `requestAnimationFrame`).
- **AI Inference Latency**: **~135 – 150 ms per frame** (~7.1 FPS throughput on CPU).
- **Post-Job Seek Latency**: **< 16 ms** (direct lookup from indexed database frame map).
- **Process Memory**: Steady-state process RSS ~210 MB; frame buffer memory < 45 MB for standard clip.
- **Queue Bounds**: WebSocket queue strictly capped at 50 items with terminal event preservation.

---

## 5. Existing Capabilities vs. Proposals

| Category | Capability | Current Code Status | Operational Verdict |
| :--- | :--- | :--- | :--- |
| **Detection** | Vehicle bounding box detection | Fully implemented (YOLOX-Nano ONNX) | Operational |
| **Tracking** | ByteTrack multi-object tracking | Fully implemented (ByteTrack) | Operational |
| **Calibration** | 4-point homography calibration | Fully implemented (`HomographyCalibrator`) | Operational |
| **Speed** | Monocular speed estimation | Fully implemented (`SpeedEstimator`) | Requires valid calibration |
| **Workbench** | Minimal, Standard, Intelligence | Fully implemented (`WorkbenchView.tsx`) | Operational |
| **Reporting** | PDF report generation | Implemented (`pdf_report.py`) | Operational |
| **Hardware** | Edge NPU / Coral Edge TPU acceleration | Architecture & reference ready | Edge firmware required |
| **General Vision**| Animals, emergency vehicles, hazards | Proposed expansion | Not implemented (future) |
