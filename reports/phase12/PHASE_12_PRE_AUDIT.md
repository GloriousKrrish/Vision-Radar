# VisionRadar — Phase 12 Forensic Pre-Audit Report
**Phase:** 12 — Research-Grade Evaluation & Benchmarking  
**Status:** COMPLETE  
**Audit Date:** 2026-09-27  

---

## 1. Executive Summary

This forensic pre-audit evaluates the existing codebase, dataset registry, evaluation modules, and test infrastructure to establish the exact baseline before executing Phase 12 research-grade benchmarking. 

VisionRadar is currently validated through Phase 11 (Traffic Intelligence). The perception pipeline consists of:
- **Production Detector:** YOLOX-Nano ONNX (`data/models/yolox_nano.onnx`, SHA256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`)
- **Production Tracker:** ByteTrack
- **Traffic Intelligence Suite:** Vehicle Counting, Flow, Occupancy, Density, Lane Intelligence, Queue Detection, Wrong-Way Detection, Stopped Vehicle Detection, Congestion Engine, Event Engine.

---

## 2. Existing Evaluation Infrastructure

The repository contains pre-existing modular evaluation components in `packages/visionradar/`:
- `packages/visionradar/eval/dataset_registry.py`: `DatasetRegistry` providing machine-readable metadata for local datasets in `DATASETS/`.
- `packages/visionradar/eval/dataset_adapters.py`: Loaders for YOLO format detection labels and MOT JSON trajectory tracking annotations.
- `packages/visionradar/eval/evaluator.py`: `BaselineEvaluator` performing ONNX inference, letterboxing (640x640), NMS, IoU matching, Precision, Recall, F1, AP@0.50 calculation.
- `packages/visionradar/benchmarks/`: Infrastructure for matcher, metric calculation, and visualization.

---

## 3. Dataset & Ground-Truth Inventory

| Dataset ID | Directory Path | Samples / Videos | Ground-Truth Type | Usability Status | Evaluation Task |
|---|---|---|---|---|---|
| `vehicle-detection-project-main` | `DATASETS/vehicle-detection-project-main` | 874 images (714 train, 80 val, 80 test) | Detection (YOLO txt) | **USABLE** | Primary Object Detection Benchmark |
| `Vehicle_Tracking` | `DATASETS/Vehicle_Tracking` | 4 video sequences (`VNTraffic`, `AICC22-Custom`) | Tracking & Trajectory (MOT JSON) | **USABLE** | Multi-Object Tracking & Trajectory Benchmark |
| `training` | `DATASETS/training` | 117 images | Detection (YOLO txt) | **USABLE** | Secondary Detection Subset |
| `valid-fine-tuning` | `DATASETS/valid-fine-tuning` | 11 images | Detection (YOLO txt) | **USABLE** | Micro-Validation Set |
| `valid-self-learning` | `DATASETS/valid-self-learning` | 6 images | Detection (YOLO txt) | **USABLE** | Micro-Sanity Check |
| `ITD-Indian-traffic-dataset-main` | `DATASETS/ITD-Indian-traffic-dataset-main` | 10 images | Unannotated | **QUALITATIVE ONLY** | Qualitative Failure Case Analysis |
| `DETRAC-Test-Tra` / `DETRAC-Train-Detections` | `DATASETS/DETRAC-*` | 0 images / 361 annotation files | Tracking (TXT) | **ANNOTATIONS ONLY** | Annotation Schema Integration |
| `BrnoCompSpeed-master` | `DATASETS/BrnoCompSpeed-master` | Code only | Speed Radar GT | **UNAVAILABLE** | Code inspection only (Raw videos & GT missing) |

---

## 4. Ground-Truth Taxonomy & Status

| GT Category | Status | Dataset Sources | Supported Metrics |
|---|---|---|---|
| **A. Detection GT** | **AVAILABLE** | `vehicle-detection-project-main` (874 images), `training` (117 images) | Precision, Recall, F1, AP@0.50, mAP@0.50, TP, FP, FN, Per-Class Breakdown |
| **B. Tracking GT** | **AVAILABLE** | `Vehicle_Tracking` (4 video sequences, 17,221 trajectory points) | Track Recall, ID Precision, ID Recall, IDF1, MOTA, MOTP, ID Switches, Fragmentation |
| **C. Event GT** | **ENGINEERING-VALIDATED** | Synthetic & 4 Real Traffic Videos (`Traffic1.mp4`, `synthetic_highway.mp4`, etc.) | Temporal Hysteresis, State Machine Transitions, Event Trigger Consistency |
| **D. Speed GT** | **BLOCKED / N/A** | `BrnoCompSpeed-master` (Missing raw files) | **BLOCKED** — Independent physical radar speed GT unavailable locally |

---

## 5. Identified Gaps & Proposed Research Matrix

To transform the evaluation into a comprehensive research paper evaluation, Phase 12 will execute the following controlled experiments:

1. **Detection & Tracking Benchmark**: Evaluate production YOLOX + ByteTrack against all available local ground truth.
2. **Traffic Intelligence Audit**: Quantify Counting, Flow, Occupancy, Density, Lanes, Queue, Wrong-Way, Stopped Vehicle, and Congestion on 4 real traffic videos.
3. **Calibration Perturbation Sensitivity**: Controlled perturbation (±1px, ±2px, ±5px, ±10px) to measure downstream impact on trajectory, distance, speed estimate, speed uncertainty, and road density.
4. **Detector Threshold Sensitivity**: Evaluate confidence thresholds (0.20, 0.25, 0.30, 0.35, 0.40, 0.50) on detection recall/precision, track count, vehicle count, and processing FPS.
5. **Resolution & Performance Scaling**: Benchmark resolutions (320, 416, 512, 640) for latency, inference time, processing FPS, and Real-Time Factor (RTF).
6. **Frame-Skip Temporal Ablation**: Compare frame skipping (1x, 2x, 3x, 4x) for throughput vs tracking continuity and event stability.
7. **Failure Case Mining**: Categorize edge-case failures into `reports/phase12/failure_cases.json`.
8. **Machine-Readable Artifacts**: Save `ground_truth_inventory.json`, `phase12_results.json`, `failure_cases.json`, research charts in `reports/phase12/figures/`, and the paper report `PHASE_12_RESEARCH_EVALUATION_REPORT.md`.
