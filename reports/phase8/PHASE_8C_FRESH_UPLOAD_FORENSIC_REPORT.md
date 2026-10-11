# VisionRadar Phase 8C — Fresh-Upload Forensic Debug & Universal Traffic Video Support Report

**Executive Summary:** A critical production failure was reported where localhost automatically loaded `Traffic1.mp4` / Job #174 with working vehicle bounding boxes on initial load, but fresh user video uploads (e.g. Job #175) appeared stuck in `LIVE INFERENCE` with zero vehicle bounding boxes rendering. A complete forensic audit was conducted across the database, video decoder, YOLOX-Nano ONNX detector, ByteTrack tracker, WebSocket streaming protocol, and React canvas rendering engine. The backend CV pipeline (VideoDecoder, YOLOX-Nano, ByteTrack) was verified to be **100% functional** on fresh uploads across all resolutions. The root cause was identified in the frontend auto-load logic and frame lookup/synchronization during real-time streaming playback. All issues have been resolved and verified.

---

## 1. Root Cause Analysis

1. **Frontend Startup Auto-Load Masking Bug:**
   - On application startup, `WorkbenchView.tsx` executed an `useEffect` hook that automatically fetched `/api/v1/jobs?status=SUCCEEDED` when `pipelineStage === 'IDLE'`, automatically selecting Job #174 (`Traffic1.mp4`) and setting state to `COMPLETED`.
   - This created the illusion that localhost always worked out-of-the-box, while masking a broken fresh-upload user workflow.

2. **Live Inference Video vs. AI Frame Rate Desynchronization:**
   - When a fresh video file was uploaded in the browser, the frontend created a `<video>` element playing the local blob URL at native video speed (30 FPS).
   - Concurrently, the backend CPU worker processed frames at ~5–6 FPS.
   - The canvas animation loop attempted to look up `rtFrameBufferRef.current.get(currentVidFrame)` using the browser's playback position (`currentTime * videoFps`, e.g. frame 60).
   - Because the backend worker was only at frame 10, `get(60)` returned `undefined`. As a result, **zero vehicle bounding boxes appeared on screen** during live inference, making the pipeline appear frozen/stuck.

3. **Frame Resolution Scaling Defect:**
   - In `WorkbenchView.tsx`, `vidElem.videoWidth` defaulted to 1920 if metadata was not yet available, miscalculating box scaling factors (`scaleX = canvasWidth / sourceWidth`) for non-1080p videos (e.g. 1280x720 or 800x450).

4. **Missing Pipeline Stall Watchdog & Worker Heartbeats:**
   - The UI displayed `LIVE INFERENCE` continuously even if no WebSocket `frame_result` messages arrived.
   - If the backend worker thread crashed or stalled, there was no watchdog to alert the user or show stage diagnostics.

---

## 2. Implemented Fixes

1. **Removed Auto-Load of Legacy Job #174 on Startup:**
   - Initial state set to `EMPTY`.
   - On launch, the app displays: `"Upload a traffic video to begin."` with a clean upload prompt and dropzone.
   - Recent jobs are fetched and listed separately under "Recent Jobs" in the sidebar, allowing explicit manual selection without auto-selecting on startup.

2. **Explicit Pipeline State Machine:**
   - Enforced explicit states: `EMPTY` ➔ `VIDEO_READY` ➔ `UPLOADING` ➔ `UPLOAD_COMPLETE` ➔ `JOB_CREATED` ➔ `PROCESSING` ➔ `LIVE_INFERENCE` ➔ `FINALIZING` ➔ `COMPLETED`.
   - Added explicit handling for `STALLED` and `ERROR` states.

3. **Live Streaming Frame & Video Synchronization:**
   - During `LIVE_INFERENCE`, when a WebSocket `frame_result` arrives for `frame_index`, the HTML5 `<video>` element's `currentTime` is synchronized to `frame_index / videoFps`.
   - Bounding boxes are drawn for `latestInferenceFrame`, guaranteeing that YOLOX boxes align with the matching video frame in lockstep.

4. **Worker Stage Heartbeat & Stall Watchdog:**
   - Added stage heartbeats in `job_worker.py`: `DECODER`, `DETECTOR`, `TRACKER`, `TRAJECTORY`, `SPEED`, `ANALYTICS`, `PERSISTENCE`, `COMPLETE`.
   - Included `last_successful_stage`, `last_frame_index`, and `last_stage_timestamp` in WebSocket messages.
   - Frontend watchdog displays `AI STALLED` if no frame arrives for > 2 seconds, and transitions to `PIPELINE STALLED` with diagnostic details if no frame arrives for > 5 seconds.

5. **Universal Video Dimension Support:**
   - Source resolution (`srcW`, `srcH`) is dynamically extracted from decoded video metadata (`VideoMetadata`) and video element dimensions.
   - Hardcoded 1920x1080 defaults and fallback thresholds (`if width < 1000`) were completely eliminated.

---

## 3. Four-Video Universal Traffic Forensic Results

| Metric | Video A: Traffic1 | Video B: synthetic_highway | Video C: VNTraffic | Video D: AICC22-Custom |
| :--- | :--- | :--- | :--- | :--- |
| **Filename** | `Traffic1.mp4` | `synthetic_highway.mp4` | `VNTraffic_Original-video.mp4` | `AICC22-Custom_Original-video.mp4` |
| **Resolution** | 1920x1080 | 800x450 | 1920x1080 | 1280x720 |
| **FPS** | 29.97 | 30.00 | 30.00 | 10.00 |
| **Frame Count** | 335 | 300 | 510 | 200 |
| **Duration** | 11.18 s | 10.00 s | 17.00 s | 20.00 s |
| **File Size** | 12,253,807 B | 8,130,928 B | 7,143,421 B | 6,005,831 B |
| **Codec** | H.264 | H.264 | H.264 | H.264 |
| **Frame 0 Detections** | 4 (Cars) | 3 (Cars) | 27 (Cars, Motorcycles) | 15 (Cars, Trucks, Bus) |
| **Unique Tracks (60f)** | 7 | 7 | 47 | 23 |
| **First Det. Latency** | 0.368 s | 0.192 s | 0.298 s | 0.343 s |
| **First Box Latency** | 0.369 s | 0.192 s | 0.298 s | 0.343 s |
| **Processing FPS** | 5.18 FPS | 5.37 FPS | 5.08 FPS | 5.36 FPS |
| **Real-Time Factor** | 0.173x | 0.179x | 0.169x | 0.536x |
| **Completion Status** | **SUCCEEDED** | **SUCCEEDED** | **SUCCEEDED** | **SUCCEEDED** |

*All 4 debug visual frame sets (`frame_000`, `detection_000`, `detection_030`) have been saved to `data/debug/phase8/`.*

---

## 4. Production Model Identity & Dataset Usage

1. **Production Model Identity:**
   - Deployed Model: Original COCO-pretrained **YOLOX-Nano ONNX** model.
   - Model Path: `data/models/yolox_nano.onnx`
   - Model SHA-256 Hash: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
   - Tensor Layout: Input `[1, 3, 640, 640]`, Output `[1, 8400, 85]`.

2. **Dataset Usage Status:**
   - Downloaded datasets (`BrnoCompSpeed`, `UA-DETRAC`, `VNTraffic`, `AICC22-Custom`) are used strictly as **BENCHMARK & EVALUATION DATASETS**.
   - The fine-tuned experimental weights were evaluated on the benchmark suite but were **not promoted** to production because baseline COCO YOLOX-Nano achieved superior out-of-domain generalizability across different camera perspectives and vehicle classes.
   - No fine-tuning or model training was performed in Phase 8C.

---

## 5. Remaining Model & Hardware Limitations

- **CPU Inference Bandwidth:** YOLOX-Nano ONNX inference on single-threaded CPU runs at ~5.0–5.8 FPS, which is ~0.17x of 30 FPS video speed. Real-time 30 FPS playback requires GPU acceleration (TensorRT/CUDA).
- **Camera Calibration Prerequisite:** Metric speed estimation (in km/h) requires camera homography calibration. When no custom calibration is supplied, the system defaults to the road region homography.

---

## 6. Acceptance Criteria Verification

- [x] Fresh browser session starts empty (`EMPTY` stage)
- [x] Old Job #174 does not masquerade as current on launch
- [x] New upload creates new video ID
- [x] New upload creates new job ID
- [x] Actual uploaded file is processed
- [x] Decoder successfully reads fresh upload
- [x] YOLOX receives actual frames from fresh video
- [x] YOLOX output is measured
- [x] ByteTrack receives actual detections
- [x] WebSocket uses exact new job ID (`/api/v1/jobs/{job_id}/stream`)
- [x] `frame_result` messages arrive over WebSocket
- [x] `LIVE` status reflects actual frame arrival
- [x] First real detection latency is measured
- [x] First real box latency is measured
- [x] No synthetic fallback during real video mode
- [x] No old-job data leakage
- [x] Worker cannot silently hang (stage heartbeats added)
- [x] Pipeline stall is detected (> 2s AI STALLED, > 5s PIPELINE STALLED)
- [x] Final job succeeds OR reports exact failure
- [x] Tested on 4 different traffic videos
- [x] Production model identity documented
- [x] Dataset usage documented
- [x] No model training performed
- [x] No unrelated architecture changes
