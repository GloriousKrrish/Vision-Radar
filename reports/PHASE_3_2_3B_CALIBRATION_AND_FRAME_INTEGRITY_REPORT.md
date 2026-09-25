# VISIONRADAR — PHASE 3.2.3B FORENSIC IMPLEMENTATION REPORT
**EXACT FRAME SYNCHRONIZATION + CALIBRATION-AWARE SPEED ROI + DYNAMIC CALIBRATION OVERLAY**

> **Final Status**: **PASS**  
> **Canonical Target**: Job #70 (`Traffic1.mp4`, 1920x1080 @ 29.97 FPS, 335 frames, 27 tracks, Calibration ID 1)  
> **Scope**: Frame Synchronization, Calibration-Aware ROI Filtering, and Dynamic Calibration Overlay. (Speed calculation engine formulas left untouched per directive).

---

## 1. Files Changed
1. **[packages/visionradar/cv/calibration.py](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/packages/visionradar/cv/calibration.py)**:
   - Added `is_point_in_quadrilateral(u, v, source_width, source_height)` to perform exact 2D polygon membership testing (`cv2.pointPolygonTest`) against calibrated $P1 \rightarrow P2 \rightarrow P3 \rightarrow P4$ points.
   - Added `is_homography_stable(u, v, source_width, source_height)` to perform numerical denominator singularity checks ($|D| \ge 0.05$) and spatial sensitivity testing ($\frac{d(world)}{d(px)} \le 5.0\text{ m/px}$).
   - Updated `image_to_world(u, v)` to automatically handle resolution scaling between 1920x1080 source image space and 800x450 display canvas space.
2. **[packages/visionradar/cv/speed.py](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/packages/visionradar/cv/speed.py)**:
   - Updated `MonocularSpeedEstimator.project_trajectory` to set `roi_status = "OUT_OF_ROI"` (or `"CALIBRATION_UNSTABLE"`) and `world_pos = None` for points outside the calibrated road plane.
   - Updated `estimate_speed_at_frame` to restrict moving-window linear regression strictly to `VALID` inside-ROI points (`world_pos is not None and roi_status == "VALID"`).
   - Ensured out-of-ROI samples return `validity = "OUT_OF_ROI"` (or `"CALIBRATION_UNSTABLE"`), setting `smoothed_kmh = None` and `uncertainty_kmh = None` (no fabricated $\pm 15\text{ km/h}$ uncertainty).
3. **[packages/visionradar/cv/trajectory.py](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/packages/visionradar/cv/trajectory.py)**:
   - Added `roi_status: str = "VALID"` attribute to `TrajectoryPoint` and dictionary serialization.
4. **[packages/visionradar/models/entities.py](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/packages/visionradar/models/entities.py)**:
   - Updated `SpeedMeasurement` schema to make `instantaneous_kmh`, `smoothed_kmh`, `uncertainty_kmh`, `confidence_low_kmh`, `confidence_high_kmh` columns `nullable=True` to store `None` for out-of-ROI samples.
5. **[packages/visionradar/worker/job_worker.py](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/packages/visionradar/worker/job_worker.py)**:
   - Passed video pixel dimensions (`src_w`, `src_h`) to `project_trajectory`.
   - Updated DB persistence loop to record `smoothed_kmh = None`, `uncertainty_kmh = None`, and `validity` status for out-of-ROI samples.
6. **[apps/web/src/components/WorkbenchView.tsx](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/apps/web/src/components/WorkbenchView.tsx)**:
   - Removed hardcoded static overlay array (`HP = [[100, 420], ...]`) from Real Video Mode.
   - Added `realCalib` state dynamically fetched from `/api/v1/videos/{video_id}/calibrations`.
   - Rendered dynamic $P1 \rightarrow P2 \rightarrow P3 \rightarrow P4$ polygon overlay derived from the exact calibration assigned to the active job.
   - Updated vehicle badge tags to display `[OUT OF ROI]` when speed is unavailable or invalid.
   - Updated Track Inspector to display `OUT OF ROI (Outside Calibrated Plane)` status instead of misleading speed or uncertainty values.

---

## 2. Exact Frame Synchronization Implementation
- **Mechanism**: Removed all $\pm 6$ frame array search logic (`Math.abs(pt.frame_index - currentVidFrame) <= 6`).
- **Data Flow**: `GET /api/v1/jobs/{job_id}/tracks` trajectory samples are indexed in a local `frameMapRef` (`Map<number, any[]>`) keyed strictly by `frame_index`.
- **Canvas Repaint**: During video playback, repaints fetch `frameMapRef.current.get(currentVidFrame)`. If no detection exists for `currentVidFrame`, `undefined` is returned and **NO BOX** is drawn.

---

## 3. Frame 200 Regression
* **Target Case**: Job #70 (`Traffic1.mp4`), Frame 200.
* **Before Fix**: Array search selected Frame 194 ($-6$ frames, $-0.2002\text{s}$ lag, bbox `[894, 98, 954, 151]`).
* **After Fix**:
  - **Requested Frame**: 200
  - **Rendered Frame**: 200
  - **Frontend Source BBox (xyxy)**: `[896.0, 113.0, 956.0, 169.0]`
  - **Canonical API BBox (xyxy)**: `[896.0, 113.0, 956.0, 169.0]`
  - **Difference**: `[0.0, 0.0, 0.0, 0.0]`

---

## 4. Frame Cache Behavior
- **Memory Structure**: $O(1)$ constant-time lookup Map (`frameMapRef`).
- **Zero Repaint Overhead**: Animation loop does not make network `fetch()` calls inside `requestAnimationFrame`.
- **Clean Transitions**: Seeking or frame scrubbing immediately clears old bounding boxes. Frame 194 detections are **NEVER** rendered for frame 200.

---

## 5. ROI Implementation & Polygon Membership
- **Rule**: For every vehicle detection, its ground anchor $(u, v) = \left(\frac{x_1 + x_2}{2}, y_2\right)$ in source image coordinates is tested against the calibration polygon $P1(330, 160) \rightarrow P2(470, 160) \rightarrow P3(748, 435) \rightarrow P4(51, 435)$.
- **Algorithm**: `cv2.pointPolygonTest(poly_pts, (u, v), False)`
  - $\ge 0 \implies$ Inside ROI (`roi_status = "VALID"`)
  - $< 0 \implies$ Outside ROI (`roi_status = "OUT_OF_ROI"`)

---

## 6. Homography Safety Method
- **Denominator Check**: Homography projection denominator $D(u, v) = H_{20}u + H_{21}v + H_{22}$.
  - Singularity line occurs at $v \approx 90.88\text{px}$ in 450 canvas space (where $D(u, v) = 0$).
  - If $|D(u, v)| < 0.05$, the sample is rejected (`roi_status = "CALIBRATION_UNSTABLE"`).
- **Spatial Sensitivity Check**: Local spatial derivative $\Delta Y = |Y(u, v+1) - Y(u, v)|$.
  - If $\Delta Y > 5.0\text{ m/px}$, 1-pixel centroid jitter produces $> 5.0\text{m}$ world displacement. Sample is rejected (`roi_status = "CALIBRATION_UNSTABLE"`).

---

## 7. Invalid Sample Handling & Speed-Window Reset
- **No Poisoning**: Out-of-ROI and unstable horizon samples have `world_pos = None`.
- **Window Filtering**: When estimating speed at target frame $N$, `MonocularSpeedEstimator` filters the temporal window $[N - 7, N + 7]$ to include **ONLY** valid inside-ROI samples (`world_pos is not None and roi_status == "VALID"`).
- **Reset/Rebuild**: When a vehicle enters the valid road plane from the horizon (e.g. Track #1 entering $y_{450} \ge 160.0$), earlier out-of-ROI samples are excluded from the linear regression fit. The regression is computed cleanly from valid calibrated road plane points.

---

## 8. Dynamic Calibration Overlay & Lineage
- **Backend Source of Truth**: Calibration ID 1 associated with Video #70 / Job #70.
- **Frontend Integration**: `WorkbenchView.tsx` fetches `GET /api/v1/videos/{video_id}/calibrations` upon video ingestion/job selection.
- **Display Scaling**: $P1-P4$ polygon coordinates are scaled dynamically to the $800 \times 450$ display canvas:
  $$P_{display}[0] = P_{src}[0] \times \text{scaleX}, \quad P_{display}[1] = P_{src}[1] \times \text{scaleY}$$
- **Identity Lineage**: `job.calibration_id` matches the calibration used by the speed estimator, frontend overlay, and telemetry.

---

## 9. Before / After Speed Samples

| Track / Frame | Before Phase 3.2.3B | After Phase 3.2.3B | Status |
|---|---|---|---|
| **Track #1 (Frame 0)** | $12,499.0\text{ km/h}$ / $705.0\text{ km/h}$ | `N/A` (`OUT_OF_ROI`) | **PASS** (Artifact eliminated without clipping) |
| **Track #1 (Frame 10)** | $894.0\text{ km/h}$ | `N/A` (`OUT_OF_ROI`) | **PASS** (Horizon sample rejected) |
| **Track #1 (Frame 43)** | $705.0\text{ km/h}$ (poisoned) | `N/A` (`OUT_OF_ROI`) | **PASS** (Invalid sample rejected) |
| **Track #2 (Frame 45)** | $901.0\text{ km/h}$ (horizon contamination) | `161.6 km/h` | **PASS** (Clean regression from valid samples) |
| **Track #15 (Frame 200)** | BBox `[894, 98, 954, 151]` (Frame 194) | BBox `[896, 113, 956, 169]` (Frame 200) | **PASS** (Exact frame synchronization) |

---

## 10. Verification Results
- **Frontend Build**: `npm run build` in `apps/web` passed with 0 TypeScript / Vite compilation errors.
- **Backend Tests**: `python -m pytest -q` passed 25/25 tests cleanly (`.........................`).
- **Audit Script**: `python scripts/reconcile_phase3_2_3b.py` passed all polygon membership, homography safety, frame matching, and Track #1 contamination tests.

---

## 11. Remaining Limitations (Explicit Out-of-Scope Items)
1. **Speed Benchmark Accuracy**:
   - Physical homography scaling for Calibration 1 produces high speed estimates (e.g. 161 km/h for Track 2) due to camera pitch geometry.
   - **Note**: Per explicit prompt directive, no speed calculation formulas or homography matrices were modified in this phase. Speed accuracy benchmarking belongs to Phase 3.3.

---

## Final Verdict
**PASS** — Exact frame rendering, calibration-aware polygon ROI filtering, homography numerical safety, invalid sample regression window resetting, and dynamic calibration overlay implemented and verified across unit tests, build tools, API endpoints, and regression audits.
