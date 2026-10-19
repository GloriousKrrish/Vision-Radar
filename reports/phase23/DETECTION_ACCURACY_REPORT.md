# VisionRadar Phase 23 — Detection Accuracy & Small-Object Evaluation Report

## 1. Executive Summary

Phase 23 systematically evaluates vehicle detection quality, small-object sensitivity, and bounding-box localization using the baseline YOLOX-Nano ONNX detector (`data/models/yolox_nano.onnx`, SHA-256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`).

All evaluations were conducted against locked reference traffic datasets:
1. **`vehicle-detection-project-main` (Test Split)**: 80 natural traffic images, 96 ground-truth vehicle annotations.
2. **`AICC22-Custom` Highway Video Sequence**: Real surveillance highway footage (1280x720) with 1,185 annotated vehicle instances, specifically containing small and distant vehicles (< 25px height).

---

## 2. Empirical Detection Benchmarks

### 2.1 Comparative Configuration Evaluation (Test Split)

| Configuration Profile | Input Resolution | NMS Mode | Precision | Recall | F1-Score | True Positive | False Positive | False Negative | Mean Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Production** | $640 \times 640$ | Class-Agnostic | **0.6195** | **0.7292** | **0.6699** | 70 | 43 | 26 | 97.96 ms | 10.2 FPS |
| **High-Res Class-NMS** | $640 \times 640$ | Class-Specific | **0.5612** | **0.8125** | **0.6638** | 78 | 61 | 18 | 88.94 ms | 11.2 FPS |
| **Optimized Native 416** | $416 \times 416$ | Class-Specific | **0.5695** | **0.8958** | **0.6964** | 86 | 65 | 10 | **44.20 ms** | **22.6 FPS** |

> [!NOTE]
> Running YOLOX-Nano at its native $416 \times 416$ resolution aligns with Megvii's training receptive field. Recall jumps from **72.92% to 89.58% (+16.66% absolute improvement)** while latency drops by **2.21x (from 98.0 ms to 44.2 ms)**.

---

### 2.2 Per-Class Detection Performance Breakdown

| Vehicle Class | GT Count | Baseline Recall | Native 416 Recall | Baseline Precision | Native 416 Precision | Native 416 F1 | Operational Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Car** | 28 | 42.86% (12 TP) | **71.43% (20 TP)** | 0.3429 | **0.4348** | **0.5405** | Drastic recall recovery on typical passenger cars. |
| **Bus** | 28 | 89.29% (25 TP) | **96.43% (27 TP)** | 0.7576 | **0.7105** | **0.8182** | Near-perfect recall on commercial public transit. |
| **Truck** | 25 | 80.00% (20 TP) | **96.00% (24 TP)** | 0.6250 | **0.4615** | **0.6234** | High recall on heavy logistics vehicles. |
| **Motorcycle** | 15 | 86.67% (13 TP) | **100.0% (15 TP)** | 1.0000 | **1.0000** | **1.0000** | 100% precision & recall with zero false positives. |
| **Total** | **96** | **72.92% (70 TP)** | **89.58% (86 TP)** | **0.6195** | **0.5695** | **0.6964** | **16 more vehicles detected per 100**. |

---

## 3. Small-Object & Distant Vehicle Evaluation

To empirically resolve the reported baseline finding ("vehicles smaller than 25 pixels in height may be missed"), real highway surveillance frames from `AICC22-Custom` were evaluated across vehicle height tiers:

| Vehicle Spatial Scale | Pixel Height | Total Evaluated Instances | Recall @ 416x416 | Recall @ 640x640 | Root Cause Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Distant Vehicles** | **< 25 px** | 43 | **79.1%** (34 / 43) | **88.4%** (38 / 43) | At 416x416, 20px objects scale to ~7px feature map activations, causing occasional feature extinction. At 640x640, spatial detail is retained for an extra ~9% recall on far-field objects. |
| **Mid-to-Close Vehicles** | **$\ge$ 25 px** | 258 | **96.1%** (248 / 258) | **96.5%** (249 / 258) | Parity: Both resolutions achieve >96% recall once vehicles enter the primary surveillance zone. |

---

## 4. Confidence Threshold Sweep on Native 416

| Confidence Threshold | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives | Operational Recommendation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.15** | 0.5028 | **0.9375** | 0.6545 | 90 | 89 | 6 | High noise; false detections on roadside furniture. |
| **0.20** | 0.5394 | 0.9271 | 0.6820 | 89 | 76 | 7 | Good sensitivity; slightly elevated false alarms. |
| **0.25 (Default)** | 0.5695 | 0.8958 | 0.6964 | 86 | 65 | 10 | **Recommended balanced operational baseline.** |
| **0.30** | 0.6187 | 0.8958 | **0.7319** | 86 | 53 | 10 | **Optimal operational threshold** (F1 = 0.7319). |
| **0.35** | **0.6538** | 0.8854 | **0.7522** | 85 | 45 | 11 | **Peak F1 score** with 88.5% recall. |

---

## 5. Architectural Recommendations & Profile Registry

VisionRadar preserves YOLOX-Nano and provides runtime selectable profiles in `packages/visionradar/cv/detection/registry.py`:
- `yolox-nano-optimized` / `yolox-fast`: $416 \times 416$, class-specific NMS, conf=0.30 (22–26 FPS CPU, 89.6% recall).
- `yolox-highres`: $640 \times 640$, class-specific NMS, conf=0.25 (11–12 FPS CPU, 88.4% recall on distant vehicles < 25px).
- `yolox-baseline`: $640 \times 640$, class-agnostic NMS (preserved exact baseline for zero-regression rollback).
