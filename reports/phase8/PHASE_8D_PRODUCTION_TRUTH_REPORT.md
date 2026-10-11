# VisionRadar Phase 8D — Production Truth & Calibration Guard Report

## Executive Summary
Phase 8D establishes strict **Production Truth** and **Calibration Guarding** across VisionRadar. Prior versions permitted uncalibrated video uploads to fall back onto synthetic road-plane homography assumptions, fabricating numeric metric speeds (km/h) without ground-truth camera calibration. 

Phase 8D completely eliminates default homography fallbacks for uncalibrated uploads while maintaining 100% operational availability for real-time YOLOX detection, ByteTrack multi-vehicle tracking, trajectory accumulation, and directional vehicle counting.

---

## 1. Fresh-Session Behavior
- **Default State**: Upon opening `http://localhost:3000` in a clean browser session (empty localStorage/sessionStorage or first launch), the UI defaults strictly to `stage = "EMPTY"`.
- **UI Presentation**: Displays `"Upload a traffic video to begin."` with empty state dropzone.
- **Auto-Selection Prevention**: No previous job (e.g. Job #174 or legacy synthetic jobs) is automatically attached or rendered upon fresh page load.

---

## 2. Upload Behavior
- **Drag-and-Drop & File Selection**: Dragging or selecting any real MP4/AVI traffic video initializes a fresh `video_id` (UUIDv4) and posts to `/api/v1/jobs/`.
- **Job Creation**: Creates a unique `job_id` (UUIDv4) in `data/visionradar.db` with `calibration_id = NULL` unless explicit camera calibration parameters are supplied.
- **Isolation**: Uploading Video B while Video A is open immediately clears Video A's active state (`tracks = []`, `trajectoryHistory = {}`, `currentJob = null`) and switches context to Video B.

---

## 3. Detection Behavior (YOLOX)
- **Model Execution**: YOLOX-Nano ONNX executes on every video frame regardless of calibration status.
- **Bounding Boxes**: Renders real-time bounding boxes with class labels (`car`, `bus`, `truck`, `motorcycle`) and confidence scores (e.g., 0.88).
- **Independence**: Detections are 100% independent of homography calibration. Uncalibrated status **never** blocks object detection.

---

## 4. Tracking Behavior (ByteTrack)
- **MOT Association**: ByteTrack associates detections across consecutive frames using Kalman filter prediction and IoU matching.
- **Track IDs**: Assigns unique integer Track IDs (`Track #1`, `Track #2`, etc.) per video session.
- **Trajectory Accumulation**: Center-bottom pixel anchor points `(u, v)` are tracked and stored into vehicle trajectories for directional movement (e.g., `SOUTHWARD`, `Lane 1`).

---

## 5. Calibration Behavior & Zero-Assumption Guard
- **No Default Homography**: Removed all fallback default synthetic homography matrices (`H = H_default`) when `job.calibration_id` is missing.
- **Zero World Assumptions**: VisionRadar never assumes camera height, pitch angle, road width, or road length for arbitrary uploaded videos.
- **Calibrator Lifecycle**: `SpeedEstimator` operates with `calibrator = None` when uncalibrated, returning `calibration_status = "CALIBRATION_REQUIRED"`.

---

## 6. Speed State Machine
Enforces an explicit 8-state machine governing metric speed availability:

| State | Condition | Numeric Speed Displayed? | UI Track Inspector Display |
| :--- | :--- | :--- | :--- |
| `NO_CALIBRATION` | Video upload without calibration parameters | No | `N/A` / `CALIBRATION REQUIRED` |
| `CALIBRATION_REQUIRED` | Calibrator absent | No | `CALIBRATION REQUIRED` |
| `CALIBRATION_INVALID` | Polygon or homography singular | No | `CALIBRATION INVALID` |
| `CALIBRATION_VALID` | Road plane homography H valid | Conditional | Depends on trajectory history |
| `INSUFFICIENT_HISTORY` | Track history < 5 frames | No | `CALCULATING...` |
| `SPEED_VALID` | Valid calibration + sufficient valid history | **YES** | **Numeric km/h value** (e.g. `48.5 km/h`) |
| `SPEED_SUSPICIOUS` | Extreme acceleration or high residual variance | Yes (Flagged) | `48.5 km/h (UNSTABLE)` |
| `SPEED_INVALID` | Point out of ROI or invalid projection | No | `OUT OF ROI` |

---

## 7. Uncalibrated-Video Behavior
For uncalibrated uploads (e.g., `Traffic1.mp4`, `VNTraffic_Original-video.mp4`, `AICC22-Custom_Original-video.mp4`):
- **Detection**: AVAILABLE (`VALID`)
- **Tracking**: AVAILABLE (`VALID`)
- **Trajectory Visualization**: AVAILABLE (`VALID`)
- **Traffic Counting**: AVAILABLE (`VALID`)
- **Metric Speed**: **BLOCKED** (`CALIBRATION REQUIRED`)
- **Track Inspector Display**: Shows speed `N/A` with clear status grid:
  - `Detection: VALID`
  - `Tracking: VALID`
  - `Metric Speed: BLOCKED`
  - `Calibration: REQUIRED`

---

## 8. Multi-Video Isolation
- **Job ID & Video ID Scoping**: Every WebSocket frame payload carries explicit `video_id` and `job_id` metadata.
- **Frontend Guard**: `WorkbenchView.tsx` rejects any WebSocket data packet whose `job_id` does not match the active session `currentJob.id`.
- **No Track ID Leakage**: Track IDs from Video A are completely purged when Video B is loaded.

---

## 9. Performance & Honest Metric Reporting
To avoid deceptive "30 FPS Real-Time" claims on CPU inference hardware:
- **Video FPS**: Extracted from media container metadata (e.g., `30.0 FPS`).
- **AI Inference FPS**: Measured empirically per frame (CPU ONNX runtime average: **5.0 – 5.8 FPS**).
- **Real-Time Factor (RTF)**: `AI Inference FPS / Video FPS` (e.g., `0.18x`).
- **UI Banner Label**: Prominently displays **`LIVE AI PROCESSING`** rather than frame-for-frame 30.0 FPS real-time claims.

---

## 10. Result Lineage
Every target track and speed evaluation retains full provenance:
```json
{
  "video_id": "vid_9f83a1b0",
  "job_id": "job_4d209c12",
  "frame_index": 142,
  "track_id": 14,
  "calibration_status": "CALIBRATION_REQUIRED",
  "diagnostic_flag": "CALIBRATION_REQUIRED",
  "speed_kmh": null
}
```
No result presented in the UI or exported data originates from previous jobs, synthetic demos, hardcoded samples, or synthetic defaults.

---

## 11. Empirical Verification & Test Results
Forensic Python test script `scratch/test_phase8d_truth.py` was executed across 3 uncalibrated traffic videos and 1 calibrated video run:

| Video Name | Calibration | Total Boxes | Total Tracks | Speed State(s) | Numeric Speeds | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Traffic1.mp4` | False | 267 | 7 | `CALIBRATION_REQUIRED` | 0 | **PASS** |
| `VNTraffic_Original-video.mp4` | False | 1,472 | 44 | `CALIBRATION_REQUIRED` | 0 | **PASS** |
| `AICC22-Custom_Original-video.mp4` | False | 859 | 22 | `CALIBRATION_REQUIRED` | 0 | **PASS** |
| `Traffic1.mp4 (Calibrated)` | True | 267 | 7 | `SPEED_VALID`, `INSUFFICIENT_HISTORY`, ... | 2 (Avg: 51.09 km/h) | **PASS** |

### Acceptance Checklist
- [x] Fresh session starts empty (`EMPTY` stage)
- [x] Real upload creates isolated job
- [x] Real bounding boxes render across all traffic videos
- [x] Real track IDs render across all traffic videos
- [x] No cross-job state contamination
- [x] Uncalibrated videos NEVER receive fabricated metric speed
- [x] Calibration-required state is clearly displayed (`CALIBRATION REQUIRED`)
- [x] Calibrated video produces numeric km/h metric speed
- [x] Multiple uploads remain 100% isolated
- [x] CPU FPS and Video FPS are honestly displayed under `LIVE AI PROCESSING`
- [x] Zero synthetic fallbacks or fake road assumptions
- [x] All automated tests pass

---

## 12. Remaining Limitations
1. **Manual Homography Tooling**: Calibration requires explicit 4-point quadrilateral ROI selection and world ground-truth distance inputs (Phase 9 item).
2. **GPU Acceleration**: Current ONNX runtime executes on CPU (5.0 – 5.8 FPS); GPU execution providers (CUDA/DirectML) can be benchmarked when dedicated hardware is attached.
