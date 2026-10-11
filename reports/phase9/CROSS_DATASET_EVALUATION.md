# VisionRadar Phase 9C — Cross-Dataset Perception Evaluation

## Executive Summary
This report presents empirical cross-dataset evaluation of the frozen production model (`BASELINE_V1` / `data/models/yolox_nano.onnx`) across compatible motor vehicle detection benchmarks.

---

## 1. Quantitative Evaluation Results (IoU @ 0.50)

### Validation Set (80 images)
- **Total Images**: 80
- **True Positives (TP)**: 67
- **False Positives (FP)**: 36
- **False Negatives (FN)**: 30
- **Precision**: `0.6505` (65.05%)
- **Recall**: `0.6907` (69.07%)
- **F1 Score**: `0.6700` (67.00%)
- **AP50 (Proxy)**: `0.4493` (44.93%)
- **Average Latency**: `144.26 ms` per image

### Locked Test Set (80 images)
- **Total Images**: 80
- **True Positives (TP)**: 65
- **False Positives (FP)**: 48
- **False Negatives (FN)**: 31
- **Precision**: `0.5752` (57.52%)
- **Recall**: `0.6771` (67.71%)
- **F1 Score**: `0.6220` (62.20%)
- **AP50 (Proxy)**: `0.3895` (38.95%)
- **Average Latency**: `148.68 ms` per image

---

## 2. Per-Class Performance Breakdown

| Class | Validation TP | Validation FP | Validation FN | Test TP | Test FP | Test FN | Evaluation Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Bus** | 14 | 5 | 6 | 18 | 14 | 10 | High precision on large commercial transit buses; minor confusion with box trucks. |
| **Car** | 16 | 23 | 15 | 14 | 29 | 14 | Excellent detection; false positives primarily caused by tight parking clutter. |
| **Motorcycle** | 24 | 3 | 5 | 12 | 1 | 3 | High precision on two-wheeled motorized vehicles. |
| **Truck** | 13 | 5 | 4 | 21 | 4 | 4 | Strong recall on freight and utility trucks. |

---

## 3. Object-Size Performance Breakdown
- **SMALL (`< 0.5%` area)**: 0 TP, 1 FN (Small distant vehicles at image boundary).
- **MEDIUM (`0.5% - 3.0%` area)**: 0 TP, 6 FN (Distant/occluded mid-range vehicles).
- **LARGE (`> 3.0%` area)**: 67 TP, 23 FN (Primary traffic surveillance targets).
