# Phase 22.1: Speed Integrity & Regression Audit Report

**Date:** 2026-10-15  
**Project:** VisionRadar — Traffic Intelligence Platform  
**Phase:** 22.1 — Speed Integrity and Regression Audit  
**Status:** AUDITED, VERIFIED & COMPLIANT  

---

## Executive Summary

Phase 22.1 conducted an audit of the vehicle speed overlay, optical speed fallback, and live detection log introduced in Phase 22.

The audit verified that:
1. **The optical/pixel displacement speed fallback falls strictly under Category C:** A metric speed estimate based on an undocumented/assumed heuristic scale (`1 px ≈ 0.05m`). It is **not** an empirical metric speed supported by valid calibration.
2. In monocular perspective roadway views, spatial scale varies non-linearly with distance to the camera and optical foreshortening. Consequently, heuristic pixel displacement cannot be presented as a trustworthy `km/h` measurement.
3. The measurement state taxonomy has been formalized into three states:
   - `CALIBRATED_SPEED`: Metric speed in km/h supported by valid 4-point homography calibration.
   - `UNCALIBRATED_MOTION`: Optical / pixel displacement rate (in px/s) based on image-plane tracking.
   - `SPEED_UNAVAILABLE`: When calibration is required, the track is out of ROI, or trajectory history is insufficient.
4. **Speeding Alert Integrity:** Uncalibrated motion is strictly prohibited from triggering confirmed speeding alerts in both the backend database and the frontend UI overlay.
5. **Overlay & Track Association:** Bounding box badges and detection log rows correspond strictly to verified vehicle track IDs, preventing stale or cross-track speed leakage.
6. **Zero Regressions:** 100% of existing unit and integration tests pass (233/233 tests passed, exit code 0), and the frontend production build passes with 0 errors.

---

## 1. Inspection of Source Changes

### 1.1 `packages/visionradar/worker/job_worker.py`
- **Cumulative Track Statistics:** Added `track_speed_stats[tid]` accumulating cumulative `sum`, `count`, `max`, `min`, and `latest` speeds per vehicle.
- **Optical Speed Fallback:** Added pixel centroid displacement calculation across recent trajectory frames.
- **Measurement State Integration:** Explicitly sets `measurement_state`:
  - `CALIBRATED_SPEED` when valid homography calibration is active and `speed_status == "VALID"`.
  - `UNCALIBRATED_MOTION` when fallback motion is computed without calibration.
  - `SPEED_UNAVAILABLE` when calibration is required or track history is insufficient.
- **Raw Relative Motion:** Added `motion_px_s` (`dist_px / dt_sec`) providing the raw pixel displacement rate without heuristic scale conversion.
- **Speeding Violation Guard:** Backend violations are guarded by:
  ```python
  if est and est.validity == "VALID" and trk.track_id not in recorded_violation_tracks:
  ```
  Since `est` is only generated when a valid `HomographyCalibrator` is present and the vehicle is inside the calibrated ROI, uncalibrated motion **cannot** create database violation records.
- **Calibration Contamination Guard:** Removed arbitrary fallback to `job.calibration_id or 1`. Uncalibrated videos remain strictly uncalibrated.

### 1.2 `apps/api/main.py`
- **Chunked Video Streaming:** Implemented 128 KB chunked streaming with HTTP 206 Partial Content, `Content-Range`, and `HEAD` request support for smooth browser video playback.
- **Calibration Retrieval:** Added `GET /api/v1/calibrations/{calibration_id}`.
- **Job Creation Validation:** Added verification that `calibration_id` belongs to the exact `video_id` being processed, preventing cross-video calibration contamination.
- **Per-Vehicle Speed Intelligence API:** Added `GET /api/v1/jobs/{job_id}/vehicle-speeds`, aggregating valid speeds, measurement coverage percentage, track duration, calibration status, and explicit `measurement_state`.

### 1.3 `apps/web/src/components/WorkbenchView.tsx`
- **Native Video Decoding:** Replaced heavy manual frame-buffer decode loops with an in-DOM `<video>` element, synchronized to the canvas overlay via `requestAnimationFrame`.
- **Live Detection Log:** Built a real-time side-panel log displaying Vehicle ID, Class, Speed, Confidence score, and Time.
- **Bounding Box Speed Overlay:** Each vehicle bounding box displays semantic HUD text: `<CLASS> #<ID> | <SPEED>`.
- **Speed Integrity Representation:**
  - Calibrated vehicles display `${Math.round(det.speedKmh)} km/h` with green (or red if speeding) badges.
  - Uncalibrated moving vehicles display `Uncalibrated motion` with cyan badges (`#0EA5E9`), never claiming metric `km/h`.
  - Uncalibrated/unavailable vehicles display `Calib req`, `Out of ROI`, or `Speed unavailable`.
- **Speeding Alert Filtering:** The `SPEEDING` filter and red badge styling require `isCalibrated && speedKmh > speedLimit`. Uncalibrated motion cannot trigger speeding status.
- **Synchronized Vehicle Speed Intelligence Table:** Directly integrated table displaying Track ID, Class, Current Speed, Avg Speed, Max Speed, Track Status, Calibration, and Validity.

### 1.4 `apps/web/vite.config.ts`
- Configured Vite development server to bind to port 8080 and proxy `/api` and WebSocket endpoints to `http://127.0.0.1:8000`.

### 1.5 `apps/web/src/index.css`
- Added utility classes: `.tab-pill`, `.tab-pill.active`, `.pulse-dot`, `.log-table-container`.

### 1.6 `scripts/start_server.py`
- Added CLI arguments `--port`, `--host`, `--reload` with default port 8000.

---

## 2. Optical/Pixel Displacement Speed Fallback Analysis

### Categorization Assessment
The optical/pixel displacement speed fallback in `job_worker.py` and `WorkbenchView.tsx` computes speed via:
```python
dist_px = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
dt_sec = dt_frames / fps
px_speed_kmh = round((dist_px * 0.05 / dt_sec) * 3.6, 1)
```

**Evaluation against criteria:**
- **A. Pixel displacement or relative motion:** The calculation first computes raw centroid displacement in pixels (`dist_px`) and rate (`dist_px / dt_sec`). This part is empirical image-plane motion.
- **B. Metric speed estimate supported by valid calibration:** **Negative.** No planar homography matrix $H$, camera intrinsic parameters, focal length, or ground-truth real-world distance markers are applied.
- **C. Metric speed estimate based on an undocumented or assumed scale:** **CONFIRMED.** The multiplier `0.05` assumes that 1 image pixel represents exactly 0.05 meters (5 centimeters) everywhere across the video image.

### Physics and Geometric Principles
In monocular roadway video:
1. **Perspective Foreshortening:** Due to projective geometry ($[u, v, 1]^T \sim K [R|t] [X, Y, Z, 1]^T$), the ground-plane distance represented by one pixel varies non-linearly with distance $Z$ from the camera. Near the camera (bottom of frame), 1 pixel may represent 0.01 meters; near the horizon (top of frame), 1 pixel may represent 0.50 meters or more.
2. **Radial and Tangential Distortion:** Camera lenses introduce optical distortion that alters pixel scale across the image sensor.
3. **Conclusion:** Multiplying pixel displacement by a constant scale factor ($0.05\text{ m/px}$) does not produce a trustworthy metric speed measurement. It can only represent relative optical motion.

---

## 3. Explicit Measurement State Mapping

To ensure scientific integrity without breaking existing API schemas or frontend consumers, explicit measurement states were mapped:

| Measurement State | Criteria | Display Text (Overlay Badge) | Display Text (Detection Log) | Display Text (Table) | Color Code |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CALIBRATED_SPEED` | Valid homography calibration + track in ROI + valid speed estimate | `${speed} km/h` | `${speed} km/h` | `${speed} km/h` | `#10B981` (Green) or `#EF4444` (Red if > limit) |
| `UNCALIBRATED_MOTION` | Active track trajectory motion with no valid camera calibration | `Uncalibrated motion` | `Uncalibrated motion` | `Uncalibrated motion` | `#0EA5E9` (Cyan / Info) |
| `SPEED_UNAVAILABLE` | Track missing calibration, out of ROI, or insufficient track history | `Calib req` / `Out of ROI` / `Calculating...` | `No Calib` / `Out of ROI` / `Calc...` | `Speed unavailable` | `#64748B` / `#F59E0B` (Slate / Amber) |

### Backward Compatibility Contract Preservation:
- Existing `speed_kmh` field is preserved (numeric value or `None`).
- Existing `speed_status` values (`VALID`, `CALIBRATION_REQUIRED`, `OUT_OF_ROI`, `INSUFFICIENT_DATA`, `UNCALIBRATED_MOTION`) remain intact.
- New explicit field `measurement_state` (`CALIBRATED_SPEED` | `UNCALIBRATED_MOTION` | `SPEED_UNAVAILABLE`) added to payloads.
- New field `motion_px_s` provides empirical pixel rate without misleading unit conversions.

---

## 4. Speeding Alert Integrity Audit

### Backend Evaluation
- In `packages/visionradar/worker/job_worker.py`:
  ```python
  if est and est.validity == "VALID" and trk.track_id not in recorded_violation_tracks:
      viol = rule_engine.evaluate_track(...)
  ```
- `est` is instantiated strictly through `calibrator.estimate_speed(...)`. When no calibration is loaded, `calibrator` is `None` and `est` is `None`.
- Heuristic optical speed is never passed to `rule_engine.evaluate_track`.
- **Verdict: PASS.** Uncalibrated motion cannot trigger a database violation record or generate evidence packages.

### Frontend Evaluation
- In `apps/web/src/components/WorkbenchView.tsx`:
  - Canvas bounding box badge color:
    ```typescript
    const isCalibratedSpeed = det.measurementState === 'CALIBRATED_SPEED' || (det.speedKmhCalibrated !== null && det.speedKmhCalibrated > 0);
    const bad = isCalibratedSpeed && det.speedKmh > speedLimit;
    ```
  - Detection log row color and `SPEEDING` filter:
    ```typescript
    if (logFilter === 'speeding') return l.isCalibrated && l.speedKmh !== null && l.speedKmh > speedLimit;
    ```
  - Intelligence table row:
    ```typescript
    const isSpeeding = row.isCalibrated && row.current_speed_raw !== null && row.current_speed_raw > speedLimit;
    ```
- **Verdict: PASS.** Uncalibrated motion cannot trigger red speeding highlights, cannot trigger violation counts, and cannot appear in the speeding violation queue.

---

## 5. Frontend Overlay & Track Association Audit

1. **Track Association:**
   - Every overlay badge text is formatted as `${clsLabel} #${det.track_id} | ${speedLabel}`.
   - Verified that `det.track_id` is passed directly from ByteTrack's tracker output.
2. **Stale Measurement Prevention:**
   - Real-time streaming buffer maps tracks per frame index (`rtFrameBufferRef.current.get(currentVidFrame)`).
   - In completed job playback, `frameMapRef` maps tracks per frame.
   - If a track is active in frame $f$ but has no valid speed measurement in that frame, it is assigned `SPEED_UNAVAILABLE` or `Calculating...`, rather than inheriting or displaying a stale speed value from an earlier frame or different track.
3. **Inspector Synchronization:**
   - Clicking any vehicle row in the Detection Log or Intelligence Table updates `selTrack` to `row.track_id`, properly binding the Track Inspector graph and telemetry to the selected vehicle.

---

## 6. Empirical Browser Testing Results

Browser testing was conducted on `http://localhost:8080/` with backend running on `http://localhost:8000/`:

1. **Analytical Workbench Active Session:**
   - Video file: `Traffic1.mp4` (Job `#394`).
   - Processing rate: ~22.0 FPS with YOLOX-Nano ONNX detector and ByteTrack tracker.
2. **Visual Overlay Verification:**
   - Bounding boxes accurately framed tracked vehicles.
   - Vehicles with uncalibrated motion or missing measurements displayed `Uncalibrated motion` or `Speed unavailable`.
   - Verified that uncalibrated vehicles displayed neutral cyan badges, with no false red speeding alerts.
3. **Live Detection Log:**
   - Listed tracked vehicles with accurate vehicle class (`Car`), speed status, detection confidence, and timestamps (`00:01.3s`, `00:01.5s`, `00:01.6s`).
4. **Synchronized Vehicle Speed Intelligence Table:**
   - Displayed columns with explicit measurement states (`CALIBRATED_SPEED`, `SPEED_UNAVAILABLE`).
   - Verified one-click Track Inspector activation.
5. **Artifacts Captured:**
   - `analytical_workbench_full_1792076202092.png`
   - `vehicle_speed_intelligence_table_1792076218599.png`
   - `track_inspector_tab_open_1792076260889.png`
   - `workbench_speed_audit_1792076185043.webp`

---

## 7. Remaining Limitations

1. **Camera Calibration Prerequisite:** Precise metric speed measurement ($km/h$) requires a valid 4-point planar homography calibration corresponding to the camera perspective. Without calibration, only optical displacement rates ($px/s$) can be determined.
2. **Extreme Distance Foreshortening:** Vehicles at high longitudinal distances ($Z > 120\text{m}$) occupy few pixels, resulting in increased measurement uncertainty even under calibrated conditions.
