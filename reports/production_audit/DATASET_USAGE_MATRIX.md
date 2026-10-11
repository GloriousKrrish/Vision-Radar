# VISIONRADAR — DATASET USAGE MATRIX

**Audit Date:** September 27, 2026  
**Audit Target:** Full Codebase & Dataset Directory (`DATASETS/`, `data/`)

---

## 1. Comprehensive Dataset Matrix

| Dataset | Contents | Training Usage | Evaluation Usage | Live Production Usage | Potential Future Usage | Key Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`vehicle-detection-project-main`** | 880 vehicle images (800 train, 50 val, 30 test) with YOLO TXT annotations | Tried in Phase 5.2 (Rejected) | Yes (Perception benchmark) | **NO (0%)** | Domain adaptation evaluation | Tiny sample size (880 images); causes severe overfitting when fine-tuning YOLOX |
| **`training`** | 117 vehicle images with YOLO TXT annotations | Tried in Phase 5.2 (Rejected) | Yes (Perception benchmark) | **NO (0%)** | Quick smoke testing | Extremely small (117 images); insufficient for deep learning feature learning |
| **`valid-fine-tuning`** | 11 vehicle images with YOLO TXT annotations | No | Yes (Validation split) | **NO (0%)** | Validation sanity check | Only 11 images; statistically uninformative |
| **`valid-self-learning`** | 6 vehicle images with YOLO TXT annotations | No | Yes (Validation split) | **NO (0%)** | Self-learning evaluation | Only 6 images |
| **`ITD-Indian-traffic-dataset-main`** | 10 unannotated Indian traffic sample photos | No | No (Visual inspection only) | **NO (0%)** | Visual domain inspection | No annotations, no bounding boxes |
| **`BrnoCompSpeed-master`** | Monocular speed estimation evaluation scripts & benchmark tools | No | Yes (Phase 3.3 / Phase 10 speed benchmark) | **NO (0%)** | Speed accuracy benchmarking | External benchmark code; does not contain raw video files locally |
| **`DETRAC-Test-Tra`** | UA-DETRAC MOT test trajectory annotations (119 TXT files) | No | Yes (Phase 6 perception quality evaluation) | **NO (0%)** | MOT tracking evaluation | Annotations only; requires matching raw video frames |
| **`DETRAC-Train-Detections`** | UA-DETRAC training detector output files (242 TXT files) | No | Yes (Tracking benchmark) | **NO (0%)** | Tracker offline benchmarking | Baseline detector text outputs only |
| **`Vehicle_Tracking`** | 4 custom traffic videos with trajectory annotations | No | Yes (Phase 8 validation) | **NO (0%)** | Ground-truth video evaluation | Limited to 4 camera angles |
| **`Accident`** | 6,191 static CCTV accident images + Kaggle notebook | No | No | **NO (0%)** | Off-line scene-level accident classification | Static images only; no bounding boxes, no video clips, no temporal tracks |

---

## 2. Production Model Provenance Summary

> **PRODUCTION TRUTH STATEMENT:**  
> **"Production YOLOX-Nano is not trained on the local dataset."**

- **Deployed Model File:** `data/models/yolox_nano.onnx`
- **SHA-256 Hash:** `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
- **Origin:** Standard official pre-trained YOLOX-Nano checkpoint trained on the 80-class MS-COCO dataset.
- **Fine-Tuning Outcome:** Fine-tuning on local datasets in Phase 5.2 resulted in catastrophic forgetting and degraded detection mAP. The fine-tuned weights were rejected, and the production system retains the standard COCO ONNX weights.
