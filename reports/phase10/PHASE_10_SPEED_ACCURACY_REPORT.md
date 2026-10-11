# VisionRadar Phase 10 — Speed Ground-Truth Acquisition & Accuracy Benchmark Report

## Executive Summary
Phase 10 executed an exhaustive scientific investigation to benchmark VisionRadar's metric speed accuracy against independent ground-truth sensor measurements.

---

## 1. Objective & Methodology
- **Objective**: Determine quantitative metric speed error (MAE, RMSE, Median AE, P95 AE, Bias, R²) of VisionRadar's multi-estimator speed engine against independent radar/laser ground truth.
- **Zero-Fabrication Protocol**: VisionRadar strictly forbids inventing ground-truth speeds or creating fake accuracy curves.

---

## 2. Local Ground-Truth Audit
- **BrnoCompSpeed Audit**: Evaluated local `BrnoCompSpeed-master` repository. Confirmed Python evaluation code is present, but raw video streams and `gt.pkl` sensor measurement logs are missing on local disk.
- **MOT Benchmark Audit**: `VNTraffic` and `AICC22-Custom` provide tracking bounding boxes (pixel trajectories), but lack metric speed sensor annotations (km/h).

---

## 3. Ground-Truth & Evaluation Architecture (`packages/visionradar/cv/speed_gt.py`)
Implemented production-grade speed GT evaluation pipeline:
- **`GroundTruthSpeedRecord`**: Canonical schema (`dataset`, `sequence_id`, `vehicle_id`, `timestamp`, `frame_index`, `speed_kmh`, `source`, `measurement_type`, `confidence`, `camera_id`, `license_metadata`).
- **Spatio-Temporal Vehicle Matching**: Explicit matching (`match_tracks_to_gt`) matching estimated tracks to GT vehicles based on temporal overlap and spatial correlation.
- **Statistical Benchmark Engine**: Computes MAE, RMSE, Median AE, P95 AE, Bias, MAPE, R², speed range breakdown, observation window slicing (0.5s to 5.0s), heuristic uncertainty coverage (1σ, 2σ, 3σ), and calibration sensitivity (-5% to +5%).

---

## 4. Final Benchmark Status & Next Action

- **Speed GT Source Found?**: `NO (Local repository code only; raw dataset files missing on disk)`
- **GT Vehicles Matched**: `0 (Ground truth data files required)`
- **Final Phase 10 Status**: **`BLOCKED BY GROUND TRUTH`** (Acceptable scientific outcome per Phase 10 Directive #14)
- **Production System State**: Unchanged (`data/models/yolox_nano.onnx`, ByteTrack, Phase 8D calibration guard, Phase 7 speed engine fully active and verified across all tests).
