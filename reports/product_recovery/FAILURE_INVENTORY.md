# VisionRadar Product Recovery — Failure Inventory (A through M)

## 1. Inventory Overview

This failure inventory documents all investigated failure modes across the VisionRadar platform (Items A through M), distinguishing **Confirmed & Repaired Defects**, **Confirmed Operating Limitations**, and **Unverified Hypotheses**.

---

## 2. Detailed Failure Investigation (A through M)

### A. Application Startup Failures
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: If the backend process terminated abruptly during job execution, jobs remained in `status = 'RUNNING'` or `'QUEUED'` permanently upon restart.
- **Root Cause**: Database state was not reconciled on application startup.
- **Minimal Repair**: Implemented `@app.on_event("startup")` in `apps/api/main.py` that queries uncompleted jobs and transitions them to `FAILED` with message `"Server restarted during job execution"`. Auto-seeds default `Project #1`.
- **Regression Test**: `tests/test_phase22_4_lifecycle_sync.py::test_scenario_h_backend_restart_recovery`.

### B. Video Upload and Decoding Failures
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: Uploading an unreadable or corrupt video crashed the worker thread with an unhandled exception, leaving the job hanging without a terminal status. Direct DB insertion without `sha256_hash` raised `IntegrityError`.
- **Root Cause**: Missing top-level try/except around `VideoDecoder` instantiation in worker, and SQLite non-null constraint on `videos.sha256_hash`.
- **Minimal Repair**: Wrapped video decoding in `job_worker.py` to capture `FileNotFoundError` / decode errors, transitioning job to `FAILED` with full error message and `completed_at`. Mandatory SHA256 computation in upload handler.
- **Regression Test**: `tests/test_phase22_4_lifecycle_sync.py::test_scenario_e_worker_exception_handling`.

### C. Worker and Job Lifecycle Failures
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: Cancellation requests (`POST /api/v1/jobs/{id}/cancel`) had no effect on running worker threads; jobs ran to completion and consumed CPU.
- **Root Cause**: Worker loops lacked cooperative cancellation event checks.
- **Minimal Repair**: Implemented thread registry `_active_workers` and `_cancel_events`. Added pre-loop and intra-loop `cancel_event.is_set()` checks in `job_worker.py`.
- **Regression Test**: `tests/test_phase22_4_lifecycle_sync.py::test_scenario_c_running_job_cancellation`.

### D. Multiple-Job Concurrency Problems
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: Submitting a second job on the same video while the first was running spawned competing worker threads, producing race conditions in SQLite and CPU throttling.
- **Root Cause**: VisionRadar is an edge/appliance architecture with single-worker CPU design; lacked a supercession/queue policy.
- **Minimal Repair**: Implemented cooperative supercession: requesting a job on an actively processing video automatically signals cancellation to the existing worker before starting the new job.
- **Regression Test**: `tests/test_phase22_4_lifecycle_sync.py::test_scenario_b_second_job_supercedes_first`.

### E. WebSocket and REST Polling Failures
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: When the browser disconnected during streaming, the async queue filled up, causing producer threads to block or leak memory.
- **Root Cause**: Unbounded or improperly handled queue backpressure in `job_worker.py`.
- **Minimal Repair**: Implemented `_safe_put()` with a bounded `asyncio.Queue(maxsize=50)`. When full, oldest non-terminal telemetry messages are dropped, while terminal `completed` and `error` messages are strictly preserved. REST endpoint `/api/v1/videos/{id}/tracks` provides reliable post-disconnect fallback.
- **Regression Test**: `tests/test_phase22_4_lifecycle_sync.py::test_scenario_f_frontend_disconnect_handling`.

### F. Video Freezing and Synchronization Problems
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**:
  1. The browser video player froze waiting for AI inference (~135 ms CPU inference).
  2. Seeking forward during processing rendered stale bounding boxes from earlier frames.
- **Root Cause**:
  1. Playback was previously coupled to inference progress.
  2. In `WorkbenchView.tsx`, an unconditional bypass fallback rendered `latestInferenceFrame` across unrelated seek positions.
- **Minimal Repair**:
  1. Decoupled HTML5 `<video>` playback (~60 FPS) from asynchronous AI worker (~7.1 FPS).
  2. Removed `latestInferenceFrame` bypass on seek; enforced strict temporal proximity `|closest - currentVidFrame| <= 30`.
- **Regression Test**: `tests/test_phase22_4_lifecycle_sync.py::test_scenario_a_normal_processing` & browser verification.

### G. Detection and Tracking Errors
- **Status**: **CONFIRMED OPERATING LIMITATION**
- **Observed Behaviour**: Small vehicles in far perspective (< 25 pixels height) occasionally drop detection or cause ID switches under heavy occlusions.
- **Root Cause**: YOLOX-Nano uses a 416x416 input resolution optimized for CPU throughput. Far-field spatial information is lost during downsampling.
- **Evaluation**: Preserved YOLOX-Nano ONNX baseline to maintain edge CPU feasibility. Documented minimum vehicle pixel height (30px) for reliable detection.

### H. Incorrect Bounding Boxes and Duplicate Tracks
- **Status**: **CONFIRMED & REPAIRED (UI) / OPERATIONAL**
- **Observed Behaviour**: In crowded scenes (e.g. 100+ active tracks), overlapping labels and bounding boxes obscured the video view.
- **Root Cause**: UI rendered all labels, confidence scores, and IDs simultaneously.
- **Minimal Repair**: Added `Minimal`, `Standard`, and `Intelligence` visualization modes in `WorkbenchView.tsx`, along with vehicle-class filtering and track inspection drawer.

### I. Calibration and Speed-Estimation Failures
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: Vehicles moving across uncalibrated views previously showed fabricated or misleading speed numbers. Poorly conditioned homography matrices caused infinite/astronomical speeds.
- **Root Cause**: Lack of homography matrix condition checks and missing calibration enforcement.
- **Minimal Repair**: Enforced tri-state speed validation (`CALIBRATED_SPEED`, `UNCALIBRATED_MOTION`, `SPEED_UNAVAILABLE`). Enforced matrix condition number < 1e8 and reprojection RMSE < 1.0m. Uncalibrated videos display clear warnings.
- **Regression Test**: `tests/test_homography.py`, `tests/test_speed.py`, `tests/test_speed_phase7.py`.

### J. Memory Growth and Resource Leaks
- **Status**: **CONFIRMED & BOUNDED**
- **Observed Behaviour**: Long videos threatened memory exhaustion if all decoded video frames were retained in RAM.
- **Root Cause**: Potential frame accumulation.
- **Verification**: Worker decodes frame-by-frame and releases frame buffers immediately after inference. Frontend stores only lightweight JSON bounding box coordinates in memory (< 45 MB). Steady-state process RSS remains bounded at ~210 MB.

### K. Production Configuration Problems
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: Frontend SPA routing failed when accessed directly on backend port 8000.
- **Root Cause**: Static assets and SPA HTML fallback were not mounted in FastAPI.
- **Minimal Repair**: Mounted `apps/web/dist` at `/` and `/assets` in `apps/api/main.py` when dist exists.

### L. Database and Result-Persistence Failures
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: Database locking under concurrent read/write operations during live streaming.
- **Root Cause**: SQLite default `DELETE` journal mode locks the entire file on writes.
- **Minimal Repair**: Configured SQLite Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and proper scoped sessions in `database.py`.

### M. Frontend Responsiveness and Rendering Problems
- **Status**: **CONFIRMED & REPAIRED**
- **Observed Behaviour**: Sluggish UI responsiveness and stutter during video playback with high track counts.
- **Root Cause**: Excessive React re-renders triggered by high-frequency state updates on every frame.
- **Minimal Repair**: Canvas rendering loop moved completely into `requestAnimationFrame` using mutable `useRef` handles (`rtFrameBufferRef`, `frameMapRef`), eliminating React re-render thrashing.
