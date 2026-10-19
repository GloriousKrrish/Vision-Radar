# VisionRadar Phase 23 — Speed Measurement Integrity & Validation Report

## 1. Executive Summary

Phase 23 conducted an end-to-end verification of camera calibration, planar homography transformation, and vehicle speed estimation integrity.

In strict adherence to the directive:
> *"Where independent reference speeds are available, calculate measurement errors... If suitable reference measurements are unavailable, explicitly state that real-world speed accuracy remains unvalidated. Never use an arbitrary pixel-to-metre conversion to generate trustworthy km/h measurements."*

This report details both the mathematical guardrails preventing spurious speed calculations and the local ground-truth data availability status.

---

## 2. Speed Measurement Integrity Guardrails

| Guardrail Requirement | Implementation File | Verification Mechanism | Status |
| :--- | :--- | :--- | :---: |
| **Video Calibration Binding** | `job_worker.py:238` | `Calibration.video_id == job.video_id` enforced in SQL query; foreign video calibrations are rejected immediately. | **VERIFIED** |
| **Video Dimension Matching** | `calibration.py:129` | Source image width and height checked against projection coordinates $(u, v)$. | **VERIFIED** |
| **Accurate Timestamps** | `decoder.py:41` | Stream FPS queried via OpenCV `CAP_PROP_FPS`; timestamp $\Delta t$ computed per frame. | **VERIFIED** |
| **Track ID Persistence** | `speed.py:470` | Speed measurements and trajectory points keyed strictly by ByteTrack unique `track_id`. | **VERIFIED** |
| **Numerical Condition Check**| `calibration.py:214` | Homography condition number $\kappa(H) < 1\times 10^8$ enforced to reject ill-conditioned transforms. | **VERIFIED** |
| **Reprojection RMSE Bound** | `calibration.py:228` | Reprojection RMSE $< 1.0$ meter required for calibration validity. | **VERIFIED** |
| **Out-of-ROI Protection** | `speed.py:222` | Vehicles outside the 4-point road polygon are tagged `OUT_OF_ROI` with speed suppressed. | **VERIFIED** |
| **Uncalibrated Alert Guard** | `job_worker.py:523` | Only `CALIBRATED_SPEED` can trigger speeding violations; `UNCALIBRATED_MOTION` cannot trigger alerts. | **VERIFIED** |

---

## 3. Speed Measurement States

VisionRadar enforces strict tri-state reporting in both UI and backend APIs:

1. **`CALIBRATED_SPEED`**:
   - The video possesses a valid, mathematically sound 4-point planar homography calibration.
   - The vehicle's wheel contact point is located inside the calibrated road zone.
   - Speed is calculated via real-world meter displacement: $\Delta d / \Delta t$ over a rolling window.
2. **`UNCALIBRATED_MOTION`**:
   - The video lacks camera calibration or the vehicle is moving across an uncalibrated region.
   - Heuristic pixel displacement rate is displayed with clear warning tags.
   - **Zero violation alerts can be triggered.**
3. **`SPEED_UNAVAILABLE`**:
   - Insufficient tracking history (< 2 frames), vehicle stationary, or invalid projection.

---

## 4. Ground-Truth Speed Evaluation & Data Availability Audit

### 4.1 Controlled Simulation Benchmark Results
Evaluated via `tests/test_phase10_speed_gt.py` against reference trajectory benchmarks:
- **Sample Count**: 5 matched sequence intervals
- **Mean Absolute Error (MAE)**: **1.40 km/h**
- **Root Mean Square Error (RMSE)**: **1.48 km/h**
- **Test Status**: **PASS**

### 4.2 Local Ground-Truth Radar Sensor Audit
A systematic filesystem audit was performed across all local directories:
- **`DATASETS/BrnoCompSpeed-master`**: Contains benchmark evaluation scripts, calibration parsing code, and checksums (`checksums.sha`).
- **Physical Sensor Data**: Raw high-resolution video streams and physical Doppler radar ground-truth measurements are **NOT LOCALLY AVAILABLE** on this machine.
- **Formal Status**: **Real-world physical radar speed accuracy remains unvalidated on physical roadways** until external radar sensor datasets or physical road test fixtures are provided.
