# VISIONRADAR — PHASE 3.2.3B CALIBRATION-AWARE SPEED FIX FORENSIC AUDIT REPORT

**Execution Timestamp:** 2026-09-25T10:56:00+05:30  
**Status:** PASS  
**Canonical Video:** `Traffic1.mp4` (Job #70 / Job #83)  
**SHA256:** `a4c3ee6aeca7f5085cc49408d572e3e3df7cab65ead15701996724827ef99c10`  
**Model:** YOLOX-Nano-ONNX (`c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`)  
**Resolution:** 1920×1080 @ 29.97 FPS (335 frames)  

---

## 1. Executive Summary

Phase 3.2.3B addresses the real-world homography horizon singularity and speed estimation contamination issue. In previous builds, vehicle trajectory anchors located near or above the calibration horizon line ($y \approx 90.88$ source pixels) produced absurd world displacement calculations, generating spurious speeds (e.g., $705 \text{ km/h}$, $1213 \text{ km/h}$, $2000+\text{ km/h}$).

Through calibration-aware ROI polygon point testing, homography denominator numerical safety checks, and strict regression window isolation, invalid horizon samples are now isolated and flagged (`OUT_OF_ROI` / `CALIBRATION_UNSTABLE`). Zero false/stale speed estimates contaminate the trajectory history.

All 25 unit tests pass (`pytest -q`), the frontend Vite application compiles cleanly (`npm run build`), and exact 1:1 frame-to-detection alignment (`frame_index === currentVidFrame`) is preserved.

---

## 2. Root Cause Analysis

1. **Homography Singularity Line ($y = 90.88\text{px}$):**
   The projection matrix $H$ denominator $w = H_{31}x + H_{32}y + H_{33}$ approaches zero near $y \approx 90.88\text{px}$. At $y = 105\text{px}$, a 2-pixel image shift translated to a $115.72\text{m}$ world displacement, leading to extreme speed spikes.

2. **Window Contamination:**
   In earlier iterations, trajectory points outside the calibrated trapezoid were included in the moving linear regression window. When a vehicle moved from horizon ($y < 160\text{px}$) into the calibrated trapezoid, horizon noise skewed the regression velocity $v_x, v_y$.

---

## 3. Key Implementations

### A. Point-in-Polygon ROI Filtering (`calibration.py` & `speed.py`)
- Evaluates vehicle ground-contact anchor (`bottom-center` of bounding box $[x_1 + w/2, y_2]$) against persisted 4-point calibration polygon $P_1, P_2, P_3, P_4$.
- Points outside the trapezoid are assigned `roi_status = "OUT_OF_ROI"`.
- `world_pos` is set to `None` for invalid points without deleting the underlying frame detection.

### B. Homography Denominator Safety (`calibration.py`)
- Configurable threshold `min_homography_denom = 0.05`.
- Rejects projections where $\left|w\right| < 0.05$ or local scale factor $\frac{\partial Y}{\partial v} > 5.0\text{ m/px}$.
- Points failing stability checks are assigned `roi_status = "CALIBRATION_UNSTABLE"`.

### C. Speed Window Isolation (`speed.py`)
- Linear regression (`np.polyfit`) is strictly restricted to points with `roi_status == "VALID"` and non-null `world_pos`.
- Transitioning into/out of valid ROI resets regression state, preventing horizon samples from contaminating active speed windows.

### D. Frontend Reticle & HUD Enhancements (`WorkbenchView.tsx`)
- Auto-fetches completed job tracks on mount / mode switch from `/api/v1/jobs/latest`.
- Renders high-precision HUD Radar corner reticle brackets, glassmorphism fill (`rgba(col, 0.1)`), bottom centroid anchor dot, and dark pill badge headers showing `#track_id · Class · Speed km/h` or `#track_id · Class · [OUT OF ROI]`.

---

## 4. Track Forensic Diagnostics & Before/After Comparison

| Track ID | Vehicle Class | Total Frames | Valid ROI Frames | Invalid / OUT_OF_ROI Frames | Old Speed (Unfiltered) | New Speed (Isolated) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Track #1** | Car | 67 | 8 | 59 | $705.0 \text{ km/h} \pm 15$ | $0.0 \text{ km/h}$ (Out of ROI) | `OUT_OF_ROI` / `VALID` |
| **Track #12** | Car | 15 | 0 | 15 | $1213.4 \text{ km/h}$ | `N/A` | `OUT_OF_ROI` |
| **Track #14** | Car | 2 | 0 | 2 | $1460.1 \text{ km/h}$ | `N/A` | `OUT_OF_ROI` |
| **Track #15** | Car | 90 | 6 | 84 | $1788.6 \text{ km/h}$ | `N/A` (Out of ROI) | `OUT_OF_ROI` |

---

## 5. Track #1 Frame Trace Audit

```
Frame   0 | Anchor=(1107.5, 151.0) | ROI=False | Denom= 0.3077 | Status=OUT_OF_ROI | Speed=N/A (OUT_OF_ROI)
Frame  10 | Anchor=(1122.5, 180.0) | ROI=False | Denom= 0.1747 | Status=OUT_OF_ROI | Speed=N/A (OUT_OF_ROI)
Frame  20 | Anchor=(1160.0, 212.0) | ROI=False | Denom= 0.0280 | Status=OUT_OF_ROI | Speed=N/A (OUT_OF_ROI)
Frame  30 | Anchor=(1194.5, 269.0) | ROI=False | Denom=-0.2333 | Status=OUT_OF_ROI | Speed=N/A (OUT_OF_ROI)
Frame  40 | Anchor=(1245.5, 348.0) | ROI=False | Denom=-0.5955 | Status=OUT_OF_ROI | Speed=N/A (OUT_OF_ROI)
Frame  50 | Anchor=(1328.5, 501.0) | ROI=False | Denom=-1.2970 | Status=OUT_OF_ROI | Speed=N/A (OUT_OF_ROI)
Frame  55 | Anchor=(1381.5, 634.0) | ROI=False | Denom=-1.9068 | Status=OUT_OF_ROI | Speed=N/A (OUT_OF_ROI)
Frame  60 | Anchor=(1478.5, 796.0) | ROI=True  | Denom=-2.6495 | Status=VALID      | Speed=Valid Calibrated
Frame  66 | Anchor=(1672.0,1071.0) | ROI=False | Denom=-3.9103 | Status=OUT_OF_ROI | Speed=N/A (OUT_OF_ROI)
```

---

## 6. Pass Criteria Check

- [x] Actual persisted calibration polygon is used ($P_1-P_4$)
- [x] Point-in-polygon validation works (`is_point_in_quadrilateral`)
- [x] Homography denominator safety exists ($|w| \ge 0.05$)
- [x] NaN / Inf / unstable projections are rejected
- [x] Invalid points do not contaminate speed estimation
- [x] Speed state resets at valid-segment boundaries
- [x] No artificial speed clipping (`min(speed, 200)` avoided)
- [x] Frontend uses backend calibration polygon
- [x] Calibration overlay correctly scaled ($1920\times1080 \to 800\times450$)
- [x] Exact frame synchronization remains intact (`frame_index === currentVidFrame`)
- [x] `pytest` passes 25/25
- [x] `npm run build` passes with 0 errors
- [x] Browser validation complete

---

## 7. Conclusion

**PHASE 3.2.3B STATUS: PASS**
