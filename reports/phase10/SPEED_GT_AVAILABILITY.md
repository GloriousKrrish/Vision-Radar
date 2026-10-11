# VisionRadar Phase 10 — Speed Ground-Truth Availability Audit

## Executive Summary
An exhaustive audit of all local datasets in `DATASETS/` was conducted to identify sensor-measured vehicle speed ground truth (laser, radar, opto-sensor, or high-precision GPS speed annotations in km/h).

---

## 1. Local Dataset Ground-Truth Audit Table

| Candidate Dataset | Local Files Present | Video Data | Camera Calibration | Sensor Speed GT (km/h) | Status | Audit Findings |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **BrnoCompSpeed-master** | Evaluation python code (`eval.py`, `config.py`) | NO | NO | **NO** | **NOT AVAILABLE** | Local repository contains ONLY evaluation scripts. Raw video files, `gt.pkl`, and `info.json` sensor measurement files are missing from local disk. |
| **Vehicle_Tracking (VNTraffic)** | Video MP4 + MOT JSON | YES | NO | **NO** | **NOT AVAILABLE** | JSON files contain bounding box frame coordinates for tracking (MOT). Zero sensor-measured metric speeds (km/h). |
| **Vehicle_Tracking (AICC22-Custom)** | Video MP4 + MOT JSON | YES | NO | **NO** | **NOT AVAILABLE** | JSON files contain tracking IDs and bounding box tracks. Zero metric speed ground truth. |
| **vehicle-detection-project-main** | Static images + YOLO TXTs | NO | NO | **NO** | **NOT AVAILABLE** | Object detection bounding boxes (`class cx cy w h`). Zero speed measurements. |
| **DETRAC-Train-Detections / Test-Tra** | Detection TXTs | NO | NO | **NO** | **NOT AVAILABLE** | Baseline detector bounding box coordinates. Zero speed ground truth. |
| **ITD-Indian-traffic-dataset** | Documentation PNGs | NO | NO | **NO** | **NOT AVAILABLE** | Documentation charts. Zero raw data. |

---

## 2. Scientific Conclusion
- **Total Local Speed GT Records**: `0`
- **Audit Result**: `NOT AVAILABLE` across all local datasets.
- **Trajectory GT vs Speed GT**: Trajectory bounding boxes in `VNTraffic` and `AICC22-Custom` provide vehicle positions in pixels, but without calibrated camera geometry or synchronized radar speed logs, pixel displacement cannot be converted into sensor speed ground truth without introducing synthetic assumptions.
