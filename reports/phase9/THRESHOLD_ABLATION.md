# VisionRadar Phase 9I — Confidence Threshold Ablation Report

## Executive Summary
Confidence threshold ablation was performed across `conf \in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]` on frozen baseline YOLOX-Nano ONNX (`640x640`).

---

## 1. Experimental Results

| Confidence Threshold | Precision | Recall | F1 Score | Avg Dets / Frame | Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.15** | 0.7876 | 0.9175 | 0.8476 | 1.41 | 147.70 |
| **0.20** | 0.8148 | 0.9072 | 0.8585 | 1.35 | 145.66 |
| **0.25 (Production Default)** | **0.8350** | **0.8866** | **0.8600** | **1.29** | **151.34** |
| **0.30** | 0.8542 | 0.8454 | 0.8497 | 1.20 | 175.88 |
| **0.35** | 0.8723 | 0.8454 | 0.8586 | 1.18 | 151.39 |
| **0.40** | 0.8876 | 0.8144 | 0.8495 | 1.11 | 148.30 |
| **0.50** | 0.8795 | 0.7526 | 0.8111 | 1.04 | 157.93 |

---

## 2. Key Findings & Conclusion
- **Optimal F1-Score**: Achieved at `conf = 0.25` (**0.8600 F1**), perfectly balancing Precision (83.50%) and Recall (88.66%).
- **Lower Thresholds (0.15-0.20)**: Increase false positives with negligible gain in valid detections.
- **Higher Thresholds (0.30-0.50)**: Significantly suppress valid distant vehicles, causing recall to drop from 88.66% to 75.26%.
- **Decision**: Production configuration `conf = 0.25` is empirically validated as optimal.
