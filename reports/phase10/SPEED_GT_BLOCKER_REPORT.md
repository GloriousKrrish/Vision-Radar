# VisionRadar Phase 10 — Speed Ground-Truth Blocker Report

## Executive Summary
In strict compliance with **Phase 10 Directive #14** (*If Speed GT Cannot Be Obtained*), this report formally documents that genuine sensor speed ground truth is currently unavailable in the local workspace. In accordance with zero-fabrication protocol, VisionRadar reports **BLOCKED BY GROUND TRUTH** rather than manufacturing synthetic speed accuracy numbers.

---

## 1. What Was Inspected
1. **`DATASETS/BrnoCompSpeed-master`**: Inspected repository files (`eval.py`, `loading.py`, `dataset_info.py`, `checksums.sha`). Confirmed `dataset/` directory contains zero `.pkl`, `.json`, `.mp4`, or `.avi` files.
2. **`DATASETS/Vehicle_Tracking/`**: Inspected `VNTraffic_GroundTruth.json` and `AICC22-Custom_GroundTruth.json`. Verified they contain frame-level bounding box coordinates (`MOT GT`) but zero physical speed sensor measurements (km/h).
3. **Exhaustive String Scan**: Executed recursive keyword search across all `.json`, `.xml`, `.txt`, `.csv`, `.pkl` files for `speed_kmh`, `laser_speed`, `radar_speed`. Returned 0 matches.

---

## 2. What Is Missing
To perform end-to-end quantitative speed accuracy evaluation, the following 3 synchronized components are required:
1. **Raw Video Stream**: Full HD / 50 FPS surveillance video.
2. **Camera Calibration Parameters**: Ground-truth vanishing point geometry (`VP1`, `VP2`), camera height, pitch angle, or 3D world reference control points.
3. **Calibrated Speed Sensor Log**: Synchronized radar / opto-sensor vehicle speed recordings (`gt.pkl` / `gt.json`) reporting ground-truth velocity $v_{\text{GT}}$ in km/h per vehicle entry/exit event.

---

## 3. Exact Requirements for Unblocking
- **Smallest Representative Subset**: 1 to 3 video sequences from BrnoCompSpeed (e.g., `session0_left` and `session0_center`, ~4 GB archive).
- **Access / License**: BrnoCompSpeed is available for academic and research evaluation from Brno University of Technology (VUTBR) via `http://www.fit.vutbr.cz/research/downloads/`.
- **Evaluation Pipeline Ready**: VisionRadar's Phase 10 benchmark engine (`packages/visionradar/cv/speed_gt.py`) is fully implemented and unit-tested to compute MAE, RMSE, Median AE, P95 AE, Bias, R², and uncertainty coverage immediately upon acquisition of raw `gt.pkl` files.
