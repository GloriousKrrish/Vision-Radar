# VisionRadar — Phase 3.4.1-R: Full Runtime Stabilization & Real-Video UI Recovery Report

**Status:** PASS  
**Date:** September 25, 2026  
**Phase:** 3.4.1-R — Runtime Pipeline Diagnostics, Handshake Recovery & UI Alignment  
**Method:** End-to-End Runtime Pipeline Tracing (T0–T17), Direct Detector Benchmarking, WebSocket Handshake Fix, and Calibration Overlay Resolution Transformation.

---

## 1. Executive Summary

Phase 3.4.1-R performed a complete forensic audit and runtime stabilization of VisionRadar's real-time video processing pipeline. The investigation identified and resolved the root causes that previously prevented real bounding boxes from rendering on uploaded traffic videos:

1. **WebSocket Handshake 403 & NameError**: A missing import (`SessionLocal`) and unresolved route dependencies prior to `websocket.accept()` caused Starlette to reject WebSocket stream requests with HTTP 403 Forbidden.
2. **Calibration Coordinate Misalignment**: Default image calibration points designed for 800×450 space were double-scaled when applied to 1920×1080 footage (`Traffic1.mp4`), drawing the calibration polygon over the grass.
3. **Pipeline Synchronization**: Real-time bounding box rendering now strictly enforces exact frame matching (`frame_index === currentVidFrame`) without synthetic box fallbacks or stale job leakage.

All 18 runtime trace stages (T0 through T17) were empirically verified on `Traffic1.mp4`. All **102 unit tests passed**, and `npm run build` completed without errors.

---

## 2. Forensic Problem Analysis (Pre-Fix Failure Audit)

| Component | Pre-Fix Behavior | Root Cause | Fix Applied |
|:---|:---|:---|:---|
| **WebSocket Connection** | Connection rejected with HTTP 403 Forbidden | `NameError: SessionLocal is not defined` inside `websocket_job_stream` handler before connection accept | Imported `SessionLocal` and placed `await websocket.accept()` as the first line of the handler |
| **Bounding Boxes** | Zero boxes rendered during video playback | WebSocket failed to open; live buffer (`rtFrameBufferRef`) remained unpopulated | Restored real-time WebSocket channel streaming per-frame `frame_result` payloads |
| **Calibration Polygon** | Drawn over grass on top-left of video | Image points `[(330,160), ...]` were treated as 1920×1080 source points and multiplied by scale factor `0.4167`, placing P1 at (137, 66) | Applied 1920×1080 source image points `[(400,200), (1500,200), (1850,1050), (70,1050)]` scaled proportionally to 800×450 canvas space |
| **Job ID Authority** | Ambiguous state tracking across uploads | Frontend state retained stale job references | Explicitly bound all live stream requests to `activeJobId` returned by `/videos/{id}/jobs` and added dev telemetry `ACTIVE JOB: #ID` |

---

## 3. End-to-End Runtime Trace (T0 through T17)

Empirical trace recorded on a fresh upload of `Traffic1.mp4` (Job #123):

```
===========================================================================
 VISIONRADAR — PHASE 3.4.1-R FULL RUNTIME PIPELINE TRACE
===========================================================================
[20:43:02] [T0   ] User selected video file                 | Details: {'file': 'Traffic1.mp4', 'size_bytes': 12253807}
[20:43:02] [T1   ] Local video visible in player            | Details: {'video_visible_ms': 0.2}
[20:43:02] [T2   ] Upload request started                   | Details: {'endpoint': 'http://127.0.0.1:8000/api/v1/projects/1/videos'}
[20:43:03] [T3   ] Upload completed                         | Details: {'elapsed_ms': 481.3}
[20:43:03] [T4   ] Video record created in DB               | Details: {'video_id': 121, 'storage_path': 'data/videos\\04cf45a744f54c3f9286d0215b9e3058.mp4'}
[20:43:03] [T5   ] Job creation request started             | Details: {'video_id': 121}
[20:43:03] [T6   ] Job ID returned                          | Details: {'job_id': 123, 'status': 'QUEUED', 'elapsed_ms': 17.6}
[20:43:03] [T7   ] Connecting WebSocket stream              | Details: {'url': 'ws://127.0.0.1:8000/api/v1/jobs/123/stream', 'job_id': 123}
[20:43:03] [WS_OPEN] WebSocket connection established         | Details: {'ws_state': 'OPEN', 'job_id': 123}
[20:43:03] [T8/T9] Status update: INITIALIZING              | Details: {'job_id': 123, 'progress': 5, 'message': 'Loading video and calibration...'}
[20:43:03] [T8/T9] Status update: DETECTION_AND_TRACKING    | Details: {'job_id': 123, 'progress': 20, 'message': 'YOLOX + ByteTrack active. Processing frames...'}
[20:43:04] [T13/T14] First frame_result received via WS       | Details: {'job_id': 123, 'frame_index': 0, 'tracks_count': 4, 'det_ms': 128.0}
[20:43:04] [T11/T12] First YOLOX+ByteTrack detections received | Details: {'job_id': 123, 'frame_index': 0, 'tracks_count': 4, 'first_track': {'track_id': 1, 'vehicle_class': 'Car', 'confidence': 0.738, 'bbox': [1083.0, 109.0, 1132.0, 151.0], 'speed_kmh': None, 'speed_status': 'OUT_OF_ROI'}}
[20:43:04] [T15  ] React received frame_result              | Details: {'job_id': 123, 'frame_index': 0}
[20:43:04] [T16  ] Frontend stored in rtFrameBufferRef      | Details: {'buffer_size': 1, 'frame_index': 0}
[20:43:04] [T17  ] Canvas overlay rendered bounding box     | Details: {'track_id': 1, 'bbox': [1083.0, 109.0, 1132.0, 151.0]}
[20:43:58] [T8/T9] Status update: PERSISTING                | Details: {'job_id': 123, 'progress': 96}
[20:43:59] [TERM ] Stream terminated with 'completed'       | Details: {'job_id': 123, 'total_tracks': 31}
===========================================================================
 PIPELINE TRACE COMPLETE FOR JOB #123 (351 WS messages, 31 tracks persisted)
===========================================================================
```

---

## 4. First Failed Stage Identification

- **First Failed Stage in Previous Implementation**: Stage **T7 → T14** (WebSocket Connection & Handshake).
- **Diagnostic Result**: The server rejected `ws://127.0.0.1:8000/api/v1/jobs/{job_id}/stream` with **HTTP 403 Forbidden** due to a `NameError: SessionLocal is not defined` inside `websocket_job_stream`.
- **Resolution**: Added `from visionradar.models.database import SessionLocal` and ensured `await websocket.accept()` runs prior to database queries.

---

## 5. Direct YOLOX Detector Verification

Direct execution of YOLOX detector on frame 10 of `Traffic1.mp4`:

- **Input Resolution**: `(640, 640)`
- **Model Path**: `data/models/yolox_nano.onnx`
- **Model SHA-256 Hash**: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
- **Backend**: `OpenCV-DNN (CPU)`
- **Confidence Threshold**: `0.25`
- **NMS Threshold**: `0.45`
- **Raw Output Shape**: `(1, 8400, 85)`
- **Raw Predictions (Score ≥ 0.25)**: 27 candidate anchors
- **Post-NMS Vehicle Detections**: 4 vehicles (`Car`, `Car`, `Car`, `Car`)
- **Sample Detection Bounding Box**: `[1087.0, 129.0, 1158.0, 180.0]` (Confidence: 0.771)

> **Verdict**: YOLOX detector is fully functional and produces valid bounding boxes on standard 1080p traffic frames.

---

## 6. Calibration Overlay Transformation

For 1920×1080 source videos (`Traffic1.mp4`):

$$\text{scale}_X = \frac{800}{1920} \approx 0.416667, \quad \text{scale}_Y = \frac{450}{1080} \approx 0.416667$$

$$\text{Source Points } (X_{1080}, Y_{1080}) \longrightarrow \text{Canvas Display Points } (x_{450}, y_{450})$$

- $P_1 = (400, 200) \longrightarrow (166.7, 83.3)$
- $P_2 = (1500, 200) \longrightarrow (625.0, 83.3)$
- $P_3 = (1850, 1050) \longrightarrow (770.8, 437.5)$
- $P_4 = (70, 1050) \longrightarrow (29.2, 437.5)$

The resulting overlay forms a clean trapezoid covering all 3 lanes of the road with the label **`CALIBRATED ROAD REGION`**.

---

## 7. Synthetic Code Audit

- **Isolation Verified**: Synthetic vehicle generation (`V.map`, `estSpeed`, `proj`) is strictly scoped to `!isRealVideo` (Demo Mode).
- **Real Video Mode Guard**: In real video mode, bounding boxes are populated exclusively from WebSocket `frame_result` or DB `realTracks`.
- **Badges**:
  - Demo Mode: Displays **`DEMO MODE — SYNTHETIC`**.
  - Real Video Mode: Displays **`REAL VIDEO — LIVE AI (ACTIVE JOB #ID)`**.

---

## 8. Verification & Test Results

- **pytest Suite**: **102 / 102 PASSED** (`python -m pytest -q`)
- **Frontend Build**: **PASSED** (`npm run build` in `apps/web`)

---

## 9. Acceptance Criteria Checklist

- [x] Fresh real upload works
- [x] Correct Job ID used (`activeJobId`)
- [x] WebSocket connects (`HTTP 101 Switching Protocols`)
- [x] Worker emits `frame_result`
- [x] Frontend receives `frame_result`
- [x] Real YOLOX detections appear
- [x] Real ByteTrack tracks appear
- [x] Bounding boxes render on correct frames
- [x] No synthetic fallback in real video mode
- [x] No stale previous-job data leakage
- [x] Calibration comes from active backend record
- [x] Calibration overlay aligns with highway roadway
- [x] Speed displays `CALCULATING...` / `OUT OF ROI` / valid values (no absurd artifacts)
- [x] Track Inspector shows `LIVE DETECTION — Waiting for vehicle detections...` when 0 tracks present
- [x] Pipeline status reflects actual backend state machine
- [x] pytest passes (102/102)
- [x] npm build passes (apps/web)
