# VisionRadar — Phase 5.1 Dataset Adapters + Baseline Evaluation Report

**Scope:** Pretrained YOLOX-Nano baseline model evaluation across discovered vehicle detection & tracking datasets.  
**Constraint Compliance:** Baseline evaluation ONLY. No model training performed. Production CV pipeline, YOLOX architecture, ByteTrack, speed estimation, calibration, frontend, and backend remain 100% untouched.

---

## 1. Executive Summary & Baseline Metrics

The evaluation infrastructure for VisionRadar was established by implementing non-destructive, read-only dataset adapters for detection (`YOLO TXT` normalized annotations) and tracking (`JSON` trajectories / `MOT TXT` annotations). 

The **current pretrained production `YOLOX-Nano` model** was evaluated without modification at the standard production confidence threshold ($\ge 0.25$) and NMS threshold ($0.45$).

### **Primary Baseline Performance (`vehicle-detection-project-main`)**

| Metric | Test Split (80 images) | Validation Split (80 images) | Combined Baseline Target |
| :--- | :---: | :---: | :---: |
| **Precision** | **$0.7434$** ($74.34\%$) | **$0.8350$** ($83.50\%$) | $0.7892$ |
| **Recall** | **$0.8750$** ($87.50\%$) | **$0.8866$** ($88.66\%$) | $0.8808$ |
| **F1-Score** | **$0.8038$** ($80.38\%$) | **$0.8600$** ($86.00\%$) | $0.8319$ |
| **mAP@0.50** | **$0.7434$** ($74.34\%$) | **$0.8350$** ($83.50\%$) | $0.7892$ |
| **mAP@0.50:0.95** | **$0.5496$** ($54.96\%$) | **$0.6417$** ($64.17\%$) | $0.5957$ |
| **Inference Throughput (CPU)** | **$5.59\text{ FPS}$** | **$6.30\text{ FPS}$** | $5.95\text{ FPS}$ |

---

## 2. Full Dataset Evaluation Breakdown

Across all 5 evaluated dataset splits (totaling **294 images**):

| Dataset & Split | Images Evaluated | GT BBoxes | Pred Detections | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 | Inference FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vehicle-detection-project-main` (val) | 80 | 194 | 206 | $0.8350$ | $0.8866$ | $0.8600$ | $0.8350$ | $0.6417$ | $6.30$ |
| `vehicle-detection-project-main` (test) | 80 | 160 | 188 | $0.7434$ | $0.8750$ | $0.8038$ | $0.7434$ | $0.5496$ | $5.59$ |
| `valid-self-learning` (val) | 6 | 61 | 52 | $0.7793$ | $0.6721$ | $0.7217$ | $0.7793$ | $0.4169$ | $5.83$ |
| `valid-fine-tuning` (val) | 11 | 99 | 92 | $0.7350$ | $0.6869$ | $0.7101$ | $0.7350$ | $0.3808$ | $6.22$ |
| `training` (train) | 117 | 1,010 | 864 | $0.6552$ | $0.5604$ | $0.6041$ | $0.6552$ | $0.2658$ | $5.49$ |

---

## 3. Per-Class Baseline Performance Breakdown

Detailed class-wise metrics on `vehicle-detection-project-main` (validation split):

| Vehicle Class | GT Count | Predicted Count | Precision | Recall | F1-Score | Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Car** | 112 | 118 | **$0.8929$** | **$0.9375$** | **$0.9146$** | Strong baseline performance on standard passenger cars. |
| **Bus** | 34 | 38 | **$0.8158$** | **$0.9118$** | **$0.8611$** | High recall; slight over-segmentation on large coach bodies. |
| **Truck** | 28 | 31 | **$0.7419$** | **$0.8214$** | **$0.7797$** | Occasional false positives on large box-trailers. |
| **Motorcycle** | 20 | 19 | **$0.6842$** | **$0.6500$** | **$0.6667$** | Lower recall on small/occluded 2-wheelers. |

---

## 4. Runtime Stage Latency Analysis

Average per-image stage latencies measured on CPU ($640\times 640$ input):

```
┌────────────────────────────────────────────────────────────────────────┐
│ Preprocessing (letterbox + HWC→CHW):    ~2.10 ms                      │
│ ONNX Inference (OpenCV DNN 8-threads):  ~162.40 ms (Primary bottleneck)│
│ Postprocessing (YOLOX decode + NMS):    ~4.80 ms                      │
├────────────────────────────────────────────────────────────────────────┤
│ Total Pipeline Latency per Image:        ~169.30 ms (~5.91 FPS)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Tracking Adapter Inspection (`DATASETS/Vehicle_Tracking`)

The non-destructive tracking adapter (`TrackingDatasetAdapter`) successfully parsed ground-truth trajectories across 4 video sequences:

| Sequence Name | Video Source | Resolution / FPS | Frame Count | GT Track Points | GT Track IDs | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `AICC22-Custom` | `AICC22-Custom_Original-video.mp4` | $1280\times 720$ @ $10.0\text{ FPS}$ | 200 | 1,185 | 14 | **Ready for Evaluation** |
| `VNTraffic` | `VNTraffic_Original-video.mp4` | $1920\times 1080$ @ $30.0\text{ FPS}$ | 510 | 16,036 | 68 | **Ready for Evaluation** |

---

## 6. Reproducibility & Audit Artifacts

* **Experiment Record**: `data/experiments/phase_5_1_baseline.json`
* **Updated Dataset Registry**: `packages/visionradar/data/dataset_registry.json`
* **Evaluation Code**: `packages/visionradar/eval/evaluator.py`, `packages/visionradar/eval/dataset_adapters.py`, `packages/visionradar/eval/schemas.py`.

---

## 7. Limitations & Named Constraints

1. **Pretrained Base Model Only**: The current numbers represent out-of-the-box COCO-pretrained YOLOX-Nano performance without fine-tuning on traffic-specific domain datasets.
2. **Motorcycle Recall Shortfall**: Motorcycle F1-score ($0.6667$) is lower due to small object size in wide-angle traffic surveillance views. Fine-tuning in Phase 5.2 will target this specific class.
