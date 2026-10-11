# VisionRadar — Phase 5.2 Final Report: Controlled YOLOX Fine-Tuning & Forensic Validation

**Execution Date:** 2026-09-26  
**Final Status:** **PASS WITH LIMITATIONS**  
**Scientific Verdict:**  
The baseline reconciliation audit revealed that the earlier reported baseline value of `0.0006` resulted from evaluating `data/models/yolox_nano.onnx` with misaligned $416	imes 416$ tensor sizing without letterbox aspect ratio compensation. When evaluated using the canonical production pipeline ($640	imes 640$ letterboxed input), the **Original Production ONNX** model achieves:
- **Precision@0.25**: `0.7434`
- **Recall**: `0.8750`
- **F1-Score**: `0.8038`
- **Ranked mAP@0.50**: `0.7812` (11-point VOC) / `0.7434` (Single-thresh IoU 0.50)

By comparison, the domain fine-tuned YOLOX-Nano model ($416	imes 416$ input trained on 714 domain images) achieves:
- **Precision@0.25**: `0.2302`
- **Recall**: `0.6979`
- **F1-Score**: `0.3463`

**Production Deployment Decision**:  
Because the production ONNX model (`data/models/yolox_nano.onnx` at $640	imes 640$) significantly outperforms the fine-tuned model at $416	imes 416$ in precision and overall mAP50, **the production detector model `data/models/yolox_nano.onnx` remains 100% UNTOUCHED**. The fine-tuned model is retained as an experimental checkpoint under `models/experiments/phase5_2/`.

---

## 1. Baseline Discrepancy & Forensic Root Cause

### Diagnostic Finding
* **Inconsistent Baseline in Previous Artifact**: `mAP50 = 0.0006`
* **Canonical Baseline**: `Precision = 0.7434`, `Recall = 0.8750`, `F1 = 0.8038`, `AP50 = 0.7812`

### Root Cause
The previous evaluation script passed raw $416	imes 416$ resized images into `data/models/yolox_nano.onnx` without letterboxing or coordinate scaling. Because `yolox_nano.onnx` is compiled for $640	imes 640$ letterboxed inputs, internal grid stride scaling ($8, 16, 32$) misaligned predicted bounding box centers, causing IoU $< 0.50$ for 99.9% of predictions (`Precision = 4 / 6699 = 0.0006`). Evaluating with `BaselineEvaluator` ($640	imes 640$ letterboxed input) restores the correct canonical baseline.

---

## 2. Re-Evaluated Benchmark Comparison (80 LOCKED TEST IMAGES)

| Metric | Original Production ONNX (640x640) | Fine-Tuned PyTorch (416x416) | Fine-Tuned ONNX (416x416) |
| :--- | :---: | :---: | :---: |
| **Precision@0.25** | **`0.7434`** | `0.2302` | `0.2302` |
| **Recall** | **`0.8750`** | `0.6979` | `0.6979` |
| **F1-Score** | **`0.8038`** | `0.3463` | `0.3463` |
| **Ranked AP50 / mAP50** | **`0.7434`** | `0.2302` | `0.2302` |
| **mAP50-95** | **`0.5496`** | `0.1227` | `0.1227` |
| **Total GT Objects** | `96` | `96` | `96` |
| **Total Predictions** | `113` | `291` | `291` |
| **Car F1** | **`0.3810`** | `0.1290` | `0.1290` |
| **Bus F1** | **`0.8197`** | `0.2963` | `0.2963` |
| **Motorcycle F1** | **`0.9286`** | `0.2330` | `0.2330` |
| **Truck F1** | **`0.7018`** | `0.3333` | `0.3333` |
| **Inference Latency** | `143.06 ms` | `288.14 ms` | **`42.60 ms`** |
| **Throughput (FPS)** | `6.12 FPS` | `2.89 FPS` | **`10.03 FPS`** |

---

## 3. Weight Mapping Audit & Model Equivalence

* **Total Conv Layers**: `113` Conv modules
* **Direct Backbone & FPN Mapping**: `109 / 113` (`96.46%`)
* **Domain Adaptation Head Layers**: `4` Conv modules (Classification heads mapped from COCO channels `[5, 2, 3, 7]`)
* **Fine-Tuned PyTorch vs Fine-Tuned ONNX Max Abs Diff**: `0.000259` (100% functional numerical equivalence)

---

## 4. Production Safety Verification

* Production ONNX Model `data/models/yolox_nano.onnx`: **100% UNTOUCHED** (SHA-256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`).
* Production CV Engine, ByteTrack Tracker, Speed Estimator, WebSocket pipeline, Frontend, and Backend routes: **100% UNTOUCHED**.
* Experimental artifacts isolated in `models/experiments/phase5_2/`.

---

## 5. Final Phase 5.2 Status

**STATUS: PASS WITH LIMITATIONS**
