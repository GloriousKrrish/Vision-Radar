# VisionRadar — Phase 6 Canonical Dataset Registry

**Date:** 2026-09-26  
**Auditor:** Antigravity AI Perception Team  
**Registry Module:** `packages/visionradar/eval/dataset_registry.py`  
**Root Path:** `DATASETS/`

---

## 1. Overview & Machine-Readable Schema

The VisionRadar Dataset Registry provides a machine-readable, schema-enforced index (`DatasetRecord`) of all research datasets stored locally in `DATASETS/`.

```json
{
  "name": "string",
  "path": "string",
  "type": "string",
  "size_mb": 0.0,
  "sample_count": 0,
  "video_count": 0,
  "annotation_count": 0,
  "classes": ["string"],
  "annotation_format": "yolo_txt | mot_json | detrac_txt | none",
  "ground_truth_types": ["detection", "tracking", "trajectory", "speed", "anomaly"],
  "splits": ["train", "val", "test"],
  "license": "string",
  "recommended_uses": ["string"],
  "limitations": ["string"],
  "hash": "string"
}
```

---

## 2. Audited Dataset Inventory

| Dataset Name | Size (MB) | Samples / Images | Videos | Label Format | Ground Truth Types | Status & Usability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`vehicle-detection-project-main`** | 76.68 | 880 | 0 | `yolo_txt` | Detection | **PRIMARY DETECTION BENCHMARK** |
| **`Vehicle_Tracking`** | 39.33 | 0 | 4 | `mot_json` | Tracking, Trajectory | **PRIMARY TRACKING BENCHMARK** |
| **`training`** | 7.82 | 117 | 0 | `yolo_txt` | Detection | **SECONDARY DETECTION EVALUATION** |
| **`valid-fine-tuning`** | 0.89 | 11 | 0 | `yolo_txt` | Detection | **FINE-TUNING VALIDATION SET** |
| **`valid-self-learning`** | 0.47 | 6 | 0 | `yolo_txt` | Detection | **SELF-LEARNING BENCHMARK** |
| **`ITD-Indian-traffic-dataset-main`** | 13.75 | 10 | 0 | `none` | None | **QUALITATIVE INDIAN DOMAIN ANALYSIS** |
| **`DETRAC-Train-Detections`** | 162.70 | 0 | 0 | `detrac_txt` | Detection, Tracking | **SCHEMA INTEGRATION ONLY** (Frames Missing) |
| **`DETRAC-Test-Tra`** | 312.23 | 0 | 0 | `detrac_txt` | Detection, Tracking | **SCHEMA INTEGRATION ONLY** (Frames Missing) |
| **`BrnoCompSpeed-master`** | 0.05 | 0 | 0 | `brno_py` | Speed | **SPEED INFRASTRUCTURE ONLY** (Not Locally Available) |

---

## 3. Ground Truth Task Matrix

| Dataset | Detection GT | Tracking GT | Trajectory GT | Speed GT | Anomaly GT | VisionRadar Target Component |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `vehicle-detection-project-main` | **YES** | NO | NO | NO | NO | Detection Quality Engine / YOLOX Evaluator |
| `Vehicle_Tracking` | **YES** | **YES** | **YES** | NO | NO | ByteTrack Tracker / Trajectory Quality Engine |
| `training` | **YES** | NO | NO | NO | NO | Secondary Detection Evaluator |
| `valid-fine-tuning` | **YES** | NO | NO | NO | NO | Validation Subset |
| `valid-self-learning` | **YES** | NO | NO | NO | NO | Self-Learning Pipeline |
| `ITD-Indian-traffic-dataset-main` | NO | NO | NO | NO | NO | Domain Shift & Failure Case Mining |
| `DETRAC-*` | **YES** | **YES** | NO | NO | NO | Multi-Object Tracking Benchmark Infrastructure |
| `BrnoCompSpeed-master` | NO | NO | NO | **SCHEMA** | NO | Speed Estimator Calibration Infrastructure |

---

## 4. Key Dataset Auditing Discoveries

1. **Detection Ground Truth**: `vehicle-detection-project-main` provides 880 annotated images with 4 classes (`bus`, `car`, `motorcycle`, `truck`). The 80 locked test images provide the canonical benchmark.
2. **Tracking & Trajectory Ground Truth**: `Vehicle_Tracking` provides 4 complete video sequences with 17,221 ground-truth trajectory points across 4 vehicle classes.
3. **Speed Ground Truth Status**: `BrnoCompSpeed-master` repository code is present, but high-resolution raw video files and radar speed ground truth measurements are **NOT LOCALLY AVAILABLE**. Speed adapters report `NOT LOCALLY AVAILABLE` transparently.
4. **Indian Traffic Domain Status**: `ITD-Indian-traffic-dataset-main` contains 10 unannotated images. Used strictly for qualitative inspection of dense traffic and auto-rickshaw domain shifts.
5. **No Fabricated Data**: No missing ground truth labels or speed data are fabricated.
