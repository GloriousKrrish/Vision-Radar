# VISIONRADAR — PHASE 3.2.3A FORENSIC IMPLEMENTATION REPORT
**EXACT REAL-VIDEO FRAME SYNCHRONIZATION**

> **Final Status**: **PASS**  
> **Target Video / Job**: Job #64 (`Traffic1.mp4`, 1920x1080 @ 29.97 FPS, 335 frames, 27 tracks)  
> **Scope**: Real-Video Frame Synchronization Fix Only (Speed Engine / Homography untouched)

---

## 1. Original Synchronization Bug
In the previous implementation of [WorkbenchView.tsx](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/apps/web/src/components/WorkbenchView.tsx), real video bounding boxes were selected using an arbitrary frame tolerance window:
```typescript
const point = trk.trajectory.find(
    (pt: any) => Math.abs(pt.frame_index - currentVidFrame) <= 6
);
```
Because `Array.prototype.find()` returns the *first* matching trajectory point in array order, when `currentVidFrame = 200`, the search matched frame `194` (the earliest point within $200 \pm 6$).

At 29.97 FPS, a 6-frame error equals $-0.2002\text{ seconds}$, causing bounding boxes to lag visually by $15\text{px}$ to $18\text{px}$ vertically above fast-moving highway vehicles:
* **Requested Frame 200 BBox (Track #15)**: `[896.0, 113.0, 956.0, 169.0]`
* **Selected Frame 194 Stale BBox**: `[894.0, 98.0, 954.0, 151.0]`

---

## 2. Files Changed
1. **[apps/api/main.py](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/apps/api/main.py)**:
   - Modified `GET /api/v1/jobs/{job_id}/frames/{frame_index}` endpoint (`get_job_frame_result`).
   - Removed backend $\pm 3$ frame fallback logic. Enforced strict 1:1 frame matching (`if p.get("frame_index") == frame_index`).
2. **[apps/web/src/components/WorkbenchView.tsx](file:///c:/Users/admin/Downloads/VisionRadar%20-%20Claude%28i%29/apps/web/src/components/WorkbenchView.tsx)**:
   - Removed all `Math.abs(pt.frame_index - currentVidFrame) <= 6` matching logic.
   - Introduced `frameMapRef` (`useRef<Map<number, any[]>>(new Map())`) to index real tracks strictly by `frame_index`.
   - Updated canvas render loop to fetch detections for `currentVidFrame` directly (`const currentFrameDets = frameMapRef.current.get(currentVidFrame)`).
   - Updated canvas click handler (`handleCanvasClick`) to perform hit testing against detections rendered for `currentVidFrame`.
   - Guaranteed empty rendering (**NO BOX**) when no detection exists for a given frame.

---

## 3. API Usage Before
* **Endpoint Called**: `GET /api/v1/jobs/{job_id}/tracks`
* **Frontend Filtering**: Scanned full track trajectory arrays on every paint frame searching for `Math.abs(pt.frame_index - currentVidFrame) <= 6`.
* **Canonical Frame API**: `GET /api/v1/jobs/{job_id}/frames/{frame_index}` was **NOT** consumed during video playback.

---

## 4. API Usage After
* **Canonical Endpoint**: `GET /api/v1/jobs/{job_id}/frames/{frame_index}` provides exact frame-level detections.
* **Frontend Integration**: Tracks fetched via `GET /api/v1/jobs/{job_id}/tracks` are processed into an in-memory `Map<number, any[]>` indexed strictly by `frame_index`.
* **Execution Flow**:
$$\text{current video frame} \rightarrow \text{frame\_index} \rightarrow \text{frameMapRef.get(frame\_index)} \rightarrow \text{React/Canvas repaint} \rightarrow \text{bounding boxes}$$

---

## 5. Frame Caching Strategy
* **Zero Network Overhead Per Paint**: Rather than executing `fetch()` calls on `requestAnimationFrame` (which causes network congestion and asynchronous jitter during 30-60 FPS playback), the full track trajectories returned by `GET /api/v1/jobs/{job_id}/tracks` are indexed into a local `Map<number, any[]>` ref (`frameMapRef`).
* **Lookup Complexity**: $O(1)$ constant time lookup per browser animation frame.
* **Stale Box Handling**: When scrubbing, seeking, or changing frames, if a frame has no detection in `frameMapRef`, `currentFrameDets` evaluates to `undefined`, resulting in zero rendered boxes. Stale boxes are immediately cleared during frame transitions.

---

## 6. Backend Frame Semantics
* **Endpoint**: `GET /api/v1/jobs/{job_id}/frames/{frame_index}`
* **Updated Query Behavior**: Scans `trajectory_json` for points where `p.get("frame_index") == frame_index`.
* **Fallback Removed**: If no detection exists for the exact requested `frame_index`, `detections: []` is returned. The backend no longer silently returns neighbor frames within $\pm 3$ frames.

---

## 7. Coordinate Transformation
* **Source Coordinates**: $1920 \times 1080$ (Traffic1.mp4 native resolution).
* **Canvas Display Buffer**: $800 \times 450$.
* **Proportional Scaling**:
  $$\text{scaleX} = \frac{800}{1920} \approx 0.416667, \quad \text{scaleY} = \frac{450}{1080} \approx 0.416667$$
* **Integrity**: Source coordinates returned by the API are scaled linearly without artificial pixel offsets or bounding box shifts.

---

## 8. Frame 200 Regression Verification
* **Test Case**: Job #64 (`Traffic1.mp4`), Requested Frame 200, Track #15.
* **Prior Failing Behavior**: Selected Frame 194 (`[894.0, 98.0, 954.0, 151.0]`), creating a $-6$ frame ($-0.2002\text{s}$) spatial lag.
* **New Behavior**:
  - **Requested Frame**: 200
  - **Rendered Frame**: 200
  - **Frontend Source BBox (xyxy)**: `[896.0, 113.0, 956.0, 169.0]`
  - **Canonical API BBox (xyxy)**: `[896.0, 113.0, 956.0, 169.0]`
  - **BBox Difference**: `[0.0, 0.0, 0.0, 0.0]`

---

## 9. Reconciliation Table (Frames 120, 150, 194, 200, 201, 202)

| Req Frame | Render Frame | Track ID | Frontend BBox (xyxy) | Canonical API BBox | Difference |
|---|---|---|---|---|---|
| **120** | 120 | 6 | `[1269.0, 326.0, 1413.0, 487.0]` | `[1269.0, 326.0, 1413.0, 487.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **120** | 120 | 7 | `[991.0, 312.0, 1134.0, 455.0]` | `[991.0, 312.0, 1134.0, 455.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **120** | 120 | 8 | `[1445.0, 120.0, 1531.0, 180.0]` | `[1445.0, 120.0, 1531.0, 180.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **120** | 120 | 10 | `[1003.0, 18.0, 1032.0, 55.0]` | `[1003.0, 18.0, 1032.0, 55.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **120** | 120 | 11 | `[1744.0, 45.0, 1813.0, 81.0]` | `[1744.0, 45.0, 1813.0, 81.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **150** | 150 | N/A | `[No Detections]` | `[No Detections]` | `0.0` |
| **194** | 194 | 12 | `[1079.0, 89.0, 1126.0, 131.0]` | `[1079.0, 89.0, 1126.0, 131.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **194** | 194 | 13 | `[1012.0, 127.0, 1069.0, 181.0]` | `[1012.0, 127.0, 1069.0, 181.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **194** | 194 | 14 | `[1265.0, 55.0, 1317.0, 90.0]` | `[1265.0, 55.0, 1317.0, 90.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **194** | 194 | 15 | `[894.0, 98.0, 954.0, 151.0]` | `[894.0, 98.0, 954.0, 151.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **194** | 194 | 17 | `[962.0, 35.0, 997.0, 62.0]` | `[962.0, 35.0, 997.0, 62.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **200** | 200 | 12 | `[1088.0, 104.0, 1141.0, 150.0]` | `[1088.0, 104.0, 1141.0, 150.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **200** | 200 | 13 | `[1015.0, 150.0, 1078.0, 209.0]` | `[1015.0, 150.0, 1078.0, 209.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **200** | 200 | 14 | `[1285.0, 60.0, 1346.0, 102.0]` | `[1285.0, 60.0, 1346.0, 102.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **200** | 200 | 15 | `[896.0, 113.0, 956.0, 169.0]` | `[896.0, 113.0, 956.0, 169.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **200** | 200 | 17 | `[963.0, 39.0, 1004.0, 66.0]` | `[963.0, 39.0, 1004.0, 66.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **201** | 201 | 12 | `[1090.0, 106.0, 1146.0, 155.0]` | `[1090.0, 106.0, 1146.0, 155.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **201** | 201 | 13 | `[1015.0, 153.0, 1083.0, 216.0]` | `[1015.0, 153.0, 1083.0, 216.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **201** | 201 | 14 | `[1294.0, 64.0, 1349.0, 102.0]` | `[1294.0, 64.0, 1349.0, 102.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **201** | 201 | 15 | `[895.0, 115.0, 957.0, 169.0]` | `[895.0, 115.0, 957.0, 169.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **201** | 201 | 17 | `[965.0, 40.0, 1003.0, 66.0]` | `[965.0, 40.0, 1003.0, 66.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **202** | 202 | 12 | `[1094.0, 112.0, 1149.0, 159.0]` | `[1094.0, 112.0, 1149.0, 159.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **202** | 202 | 13 | `[1016.0, 158.0, 1084.0, 219.0]` | `[1016.0, 158.0, 1084.0, 219.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **202** | 202 | 14 | `[1296.0, 65.0, 1354.0, 103.0]` | `[1296.0, 65.0, 1354.0, 103.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **202** | 202 | 15 | `[897.0, 116.0, 959.0, 174.0]` | `[897.0, 116.0, 959.0, 174.0]` | `[0.0, 0.0, 0.0, 0.0]` |
| **202** | 202 | 17 | `[964.0, 42.0, 1003.0, 70.0]` | `[964.0, 42.0, 1003.0, 70.0]` | `[0.0, 0.0, 0.0, 0.0]` |

---

## 10. Browser Verification
- **Application URL**: `http://localhost:3000`
- **Verification Steps Verified**:
  - Pausing video at frame 200 confirms bounding box is rendered precisely on vehicle body at source `[896, 113, 956, 169]`.
  - Advancing frame-by-frame (200 $\rightarrow$ 201 $\rightarrow$ 202) moves bounding box positions synchronously with single-frame vehicle displacements.
  - Seeking backward or forward instantly updates rendered boxes to match the target frame index without visual persistence of old boxes.
  - Clicking vehicle boxes highlights the selected `track_id` in Track Inspector.

---

## 11. Automated Test Results
* **Frontend Build**: `npm run build` in `apps/web` passed with 0 TypeScript / Vite compilation errors.
* **Backend Unit Tests**: `python -m pytest -q` passed 7/7 tests cleanly (`.......`).

---

## 12. Remaining Issues (Explicit Out-of-Scope Items)
1. **Speed Engine & Homography Singularity**:
   - Monocular speed estimation on distant horizon vehicles still shows mathematically impossible high values (e.g., 705 km/h, 437 km/h, 12,499 km/h) due to the homography denominator singularity around $y \approx 90.88\text{px}$.
   - **Note**: Per directive, no speed calculation, calibration, or homography logic was modified during Phase 3.2.3A.

---

## Final Verdict
**PASS** — Exact 1:1 real-video frame synchronization implemented and verified across API endpoints, frame cache index, canvas repaints, unit tests, build tools, and regression tables.
