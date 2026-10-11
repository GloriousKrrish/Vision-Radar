# VisionRadar Upload & Render Stability Forensic Report

## 1. User-Reported Symptom
When users load the VisionRadar application directly (Case A), the default synthetic demo clip plays smoothly without visual glitches or flickering. However, when users upload their own traffic video MP4/WebM file (Case B), live processing begins, but the screen, HTML5 video player, and canvas overlay continuously flicker, glitch, and present an unstable visual display during live processing.

## 2. Reproduction Steps
1. Open VisionRadar web workbench.
2. Confirm default synthetic demo mode operates cleanly at 30 FPS without flickering.
3. Click "Upload Traffic Video MP4/WebM" and select a user video (e.g. `Traffic1.mp4`, `VNTraffic_Original-video.mp4`, `AICC22-Custom_Original-video.mp4`, `synthetic_highway.mp4`).
4. Observe frontend during `LIVE_INFERENCE` stage as WebSocket streams frame results from YOLOX + ByteTrack backend worker.
5. **Observed Failure (Before Fix)**: The HTML5 `<video>` element continuously sought forward/backward, flashing black frames, tearing bounding boxes, and triggering 30+ full React re-renders per second.

## 3. Affected Scenarios
- **CASE A (Initial Load / Synthetic Demo)**: NOT AFFECTED. No WebSocket messages were active, so `video.currentTime` was untouched.
- **CASE B (User Upload Live Processing)**: SEVERELY AFFECTED. HTML5 video playback seeking feedback loop actively destroyed decoder playback during live streaming.
- **CASE C (Completed Job Playback)**: MODERATELY AFFECTED due to high-frequency state churn and uncoordinated Canvas `useEffect` triggers.

## 4. Initial Demo Clip Behavior
In synthetic demo mode (`isRealVideo = false`), the application renders synthetic 3D projected road geometry and simulated vehicle tracks (`V`). The canvas is driven by a synthetic animation loop. `isLiveStreamingRef.current` remains `false`, so no WebSocket frame handler modified video state.

## 5. Uploaded Video Behavior
During real video upload, `isLiveStreamingRef.current` was set to `true`. As WebSocket `frame_result` messages arrived from backend worker (30 messages/sec):
1. `ws.onmessage` checked `Math.abs(videoRef.current.currentTime - targetTime) > 0.1` and explicitly called `videoRef.current.currentTime = targetTime`.
2. Hardware/software video decoders in the browser were forced to perform continuous seeking operations 30 times per second.
3. Seeking on playing HTML5 `<video>` elements causes hardware video decoders to flush internal reference buffers, leading to blank/black frame draws on `<canvas>`.
4. Concurrently, the synthetic demo animation loop was STILL running in the background, calling `setFrame` 30 times/sec and triggering duplicate React re-renders.

## 6. Root Cause
The root cause was a **Triple Render/Decoder Feedback Loop**:

1. **Decoder Seek Flushes**: `ws.onmessage` in `WorkbenchView.tsx` set `videoRef.current.currentTime = targetTime` on EVERY incoming WebSocket frame result (30 FPS). Setting `currentTime` on playing HTML5 video elements forces decoders to seek, causing blank/black frame draws (`cx.drawImage(videoElem)`).
2. **High-Frequency React State Churn**: `ws.onmessage` called 5 React state setters per frame (`setLatestInferenceFrame`, `setLastSuccessfulStage`, `setLastStageTimestamp`, `setServerPipelineLatencyMs`, `setLiveTrackCount`), forcing 30+ full component re-renders per second.
3. **Unpaused Synthetic Demo Loop**: The synthetic demo `requestAnimationFrame` loop was active even when `isRealVideo === true`, calling `setFrame` at 30 FPS and triggering duplicate canvas `useEffect` runs.

## 7. Evidence
- `WorkbenchView.tsx` line 371 (before fix): `videoRef.current.currentTime = targetTime;` inside high-frequency `ws.onmessage`.
- `WorkbenchView.tsx` line 507 (before fix): synthetic demo loop `setFrame(f => ...)` lacked `!isRealVideo` guard.
- Forensic Telemetry instrumentation confirmed:
  - `activeWebSocketCount`: 1
  - `activeRenderLoopCount`: 1 (after fix)
  - `outOfOrderFrameCount`: 0
  - Decoder seek calls during live streaming: Reduced from 30/sec to 0/sec.

## 8. Frontend Lifecycle Analysis
- `WorkbenchView` initial mount initializes `canvasRef` and `videoRef`.
- Single dedicated `requestAnimationFrame` render loop (`drawCanvas`) runs at 60 FPS display refresh rate.
- Refs (`telemetryRef`, `realCalibRef`, `speedLimitRef`, `selTrackRef`, `playingRef`, `frameRef`) maintain current state without causing canvas render loop recreation.

## 9. WebSocket Analysis
- `connectWebSocket(jobId)` ensures single active WebSocket per job (`1 upload → 1 job → 1 WebSocket`).
- Guard `if (activeJobIdRef.current !== jobId) return;` inside `ws.onmessage` guarantees stale messages from previous jobs are immediately discarded.
- Low-frequency state updates (latency, stage text, track counts) are throttled to 4Hz (every 250ms), reducing React re-render frequency by 87%.

## 10. Frame Synchronization Analysis
- Natural presentation model: HTML5 `<video>` plays forward naturally without forced `currentTime` mutations.
- Canvas lookup calculates current video frame index dynamically: `currentVidFrame = Math.floor(videoElem.currentTime * vidFps)`.
- Detections for `currentVidFrame` are retrieved from bounded real-time frame buffer `rtFrameBufferRef.current.get(currentVidFrame)`.
- Fallback to `latestInferenceFrameRef.current` if within 2 frames of playback head.

## 11. Frame Buffer Analysis
- Bounded sliding window: `rtFrameBufferRef.current` maintains maximum 120 frames (`RT_BUFFER_FRAMES`).
- Monotonic presentation: Frames strictly ordered by `frame_index`.
- Memory bounds: Buffer size capped, preventing unbounded growth during long video processing.

## 12. React Render Analysis
- Before Fix: 30+ React component re-renders per second during live streaming.
- After Fix: React re-renders reduced to 4Hz status update intervals (or stage transitions).
- Canvas rendering decoupled from React state lifecycle into a dedicated 60 FPS `requestAnimationFrame` loop.

## 13. Backend Pipeline Analysis
- Backend YOLOX + ByteTrack worker (`job_worker.py`) produces monotonic `frame_result` payloads containing `frame_index`, `tracks`, `capture_ts`, `result_ts`.
- WebSocket message ordering is strictly monotonic.
- Dropped frames during high load are handled gracefully by frontend frame lookup without resetting video playback.

## 14. Fix Implemented
1. **Eliminated Decoder Seeking Feedback Loop**: Removed `videoRef.current.currentTime = targetTime` from `ws.onmessage`. HTML5 video plays naturally.
2. **Dedicated Single rAF Render Loop**: Implemented a single `requestAnimationFrame` loop that draws `cx.drawImage(vidElem, 0, 0, W, H)` and overlay matching `currentVidFrame`.
3. **Paused Demo Loop during Real Video**: Added `!isRealVideoRef.current` guard to demo frame state updates.
4. **State Churn Reduction & Throttling**: High-frequency frame data saved in `rtFrameBufferRef` silently. Status updates throttled to 250ms intervals.
5. **Object URL Lifecycle Cleanup**: Implemented explicit `URL.revokeObjectURL(currentBlobUrlRef.current)` on new file uploads and component unmount.
6. **Strict Job Isolation**: Discarded WebSocket messages where `jobId !== activeJobIdRef.current`.
7. **Forensic Telemetry Panel**: Added toggleable `🔬 Forensic Telemetry` panel exposing live counts of active WebSockets, render loops, buffer size, received/presented frames, out-of-order frames, and blob URL allocations.

## 15. Regression Testing
- **Phase 16 Stability Suite**: `tests/test_phase16_upload_render_stability.py` (5/5 PASSED).
- **Full Repository Suite**: `python -m pytest tests/` (205/205 PASSED).
- **Frontend Production Build**: `npm run build --prefix apps/web` (PASSED).

## 16. Browser Verification
- **Verified**: Clean HTML5 video playback and overlay drawing without decoder seeking or canvas tearing.
- **Verified**: Monotonic frame matching for live streaming and completed job playback.
- **Verified**: Calibration guard, metric speed estimation, Phase 13 anomaly risk score, and Phase 13.1 accident candidate panels remain 100% functional.

## 17. Performance Before/After
| Metric | Before Fix | After Fix | Status |
| :--- | :--- | :--- | :--- |
| HTML5 Video Seeking | 30 calls/sec | 0 calls/sec | FIXED |
| Component Re-renders | 30+ /sec | ~4 /sec (throttled status) | OPTIMIZED |
| Canvas Render FPS | Jittery / Glitching | Smooth 60 FPS rAF | FIXED |
| Active Render Loops | 2 (Demo + Canvas) | 1 (Dedicated rAF) | FIXED |
| Active WebSockets | Unchecked | 1 per Job (Isolated) | FIXED |
| Object URL Leaks | Unrevoked | Revoked on Upload/Unmount | FIXED |
| Full Test Suite | 200/200 PASSED | 205/205 PASSED | VERIFIED |
| Web Build | PASSED | PASSED | VERIFIED |

## 18. Remaining Limitations
- HTML5 video playback speed depends on browser media decoder hardware acceleration support.
- If backend worker drops frames under extreme CPU load, overlay matches nearest buffered frame within ±2 frame threshold.

## 19. Final Status
**FIXED**
