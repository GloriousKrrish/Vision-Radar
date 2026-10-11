# VisionRadar — Phase 5.0 Dataset Inventory, Validation & Experiment Planning Report

**Scope:** Forensics audit of `VisionRadar/DATASETS/` folder.  
**Constraint Compliance:** Read-only audit. No datasets deleted, moved, modified, converted, or trained. Production CV pipeline, YOLOX detector, ByteTrack, speed estimation, calibration, frontend, and backend remain 100% untouched.

---

## 1. Executive Summary

A comprehensive forensic audit of the `VisionRadar/DATASETS/` directory was conducted to discover, categorize, and validate all dataset assets currently available for VisionRadar development.

* **Total Dataset Storage:** `696.64 MB` (730,480,006 bytes across 2,490 files).
* **Total Image Samples:** `1,014` annotated & unannotated images.
* **Total Video Clips:** `4` annotated video clips (Vietnam Traffic & AI City Custom sequences).
* **Total Annotation Files:** `1,256` bounding box & trajectory files (YOLO TXT, MOT TXT, JSON, XML).
* **Complete Datasets Identified:** `6` ready datasets (`vehicle-detection-project-main`, `training`, `valid-fine-tuning`, `valid-self-learning`, `Vehicle_Tracking`, `DETRAC-Train-Detections`).
* **Code/Metadata-Only Repositories:** `2` repos (`BrnoCompSpeed-master`, `DETRAC-Test-Tra`).
* **Unannotated Repositories:** `1` repo (`ITD-Indian-traffic-dataset-main`).
* **Incomplete / Partial Downloads:** `3` `.crdownload` files ($\approx 93.73\text{ MB}$ total).

---

## 2. Dataset Inventory

| Dataset Name | Disk Size | File Count | Image Count | Video Count | Annotation Count | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vehicle-detection-project-main` | $76.68\text{ MB}$ | 1,772 | 880 | 0 | 877 | **Complete** |
| `training` | $7.82\text{ MB}$ | 234 | 117 | 0 | 117 | **Complete** |
| `valid-fine-tuning` | $0.89\text{ MB}$ | 22 | 11 | 0 | 11 | **Complete** |
| `valid-self-learning` | $0.47\text{ MB}$ | 12 | 6 | 0 | 6 | **Complete** |
| `Vehicle_Tracking` | $39.33\text{ MB}$ | 8 | 0 | 4 | 4 | **Complete** |
| `DETRAC-Train-Detections` | $162.70\text{ MB}$ | 242 | 0 | 0 | 242 | **Complete Detections** |
| `DETRAC-Test-Tra` | $312.23\text{ MB}$ | 171 | 0 | 0 | 119 | **Complete Annotations/Code** |
| `ITD-Indian-traffic-dataset-main` | $13.75\text{ MB}$ | 17 | 10 | 0 | 0 | **Unannotated Samples** |
| `BrnoCompSpeed-master` | $0.05\text{ MB}$ | 9 | 0 | 0 | 0 | **Code Only** |
| `Unconfirmed 238577.crdownload` | $45.24\text{ MB}$ | 1 | 0 | 0 | 0 | ⚠️ **Incomplete Download** |
| `Unconfirmed 241433.crdownload` | $22.63\text{ MB}$ | 1 | 0 | 0 | 0 | ⚠️ **Incomplete Download** |
| `Unconfirmed 421580.crdownload` | $25.86\text{ MB}$ | 1 | 0 | 0 | 0 | ⚠️ **Incomplete Download** |

---

## 3. Storage Usage Breakdown

```
DATASETS/ (696.64 MB)
├── DETRAC-Test-Tra ────────────────────────────── 312.23 MB (44.8%)
├── DETRAC-Train-Detections ────────────────────── 162.70 MB (23.4%)
├── vehicle-detection-project-main ───────────────  76.68 MB (11.0%)
├── Incomplete Downloads (.crdownload) ───────────  93.73 MB (13.5%)
├── Vehicle_Tracking ─────────────────────────────  39.33 MB ( 5.6%)
├── ITD-Indian-traffic-dataset-main ──────────────  13.75 MB ( 2.0%)
├── training ─────────────────────────────────────   7.82 MB ( 1.1%)
└── valid-fine-tuning / valid-self-learning ──────   1.36 MB ( 0.2%)
```

---

## 4. Image & Video Analysis

| Dataset | Media Type | Resolution(s) | FPS | Frame Count / Duration | Lighting / Environment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vehicle-detection-project-main` | Images | $640\times 640$, $1920\times 1080$, $1332\times 880$ | N/A | 880 images | Daytime urban & highway |
| `training` | Images | $640\times 640$ | N/A | 117 images | Daytime traffic |
| `valid-fine-tuning` | Images | $640\times 640$ | N/A | 11 images | Daytime traffic |
| `valid-self-learning` | Images | $640\times 640$ | N/A | 6 images | Daytime traffic |
| `ITD-Indian-traffic-dataset-main` | Images | $1440\times 1080$, $1920\times 1080$, $3000\times 2250$ | N/A | 10 images | Indian urban roads (Daytime) |
| `Vehicle_Tracking` (AICC22) | Video | $1280\times 720$ | $10.0$ | 200 frames ($20.0\text{s}$) | Highway surveillance |
| `Vehicle_Tracking` (VNTraffic) | Video | $1920\times 1080$ | $30.0$ | 510 frames ($17.0\text{s}$) | Dense Asian motorcycle/car traffic |

---

## 5. Annotation Forensics

### **Format A: YOLO Normalized TXT (`<class_id> <cx> <cy> <w> <h>`)**
* Found in: `vehicle-detection-project-main`, `training`, `valid-fine-tuning`, `valid-self-learning`.
* Snippet example (`training/labels/04308_20231013_125549_jpg.rf.txt`):
  ```text
  1 0.193947 0.646705 0.075699 0.123805
  1 0.222077 0.512823 0.059974 0.105890
  ```

### **Format B: MOT Tracking TXT (`<frame_idx>,<target_id>,<left>,<top>,<width>,<height>,<conf>,-1,-1,-1`)**
* Found in: `Vehicle_Tracking/AICC22-Custom/AICC22-Custom_GroundTruth.txt`, `DETRAC-Train-Detections`.
* Snippet example:
  ```text
  3,1,163,293,335,139,1.0,-1,-1,-1
  3,2,845,179,58,50,1.0,-1,-1,-1
  ```

### **Format C: Ground-Truth Trajectory JSON**
* Found in: `Vehicle_Tracking/AICC22-Custom/AICC22-Custom_GroundTruth.json`.
* Snippet example:
  ```json
  {
    "1": {
      "frames": ["./train/S01/c002/img1/000003.jpg"],
      "boxes": [[163, 293, 335, 139]]
    }
  }
  ```

---

## 6. Class Analysis

### **Relevant Vehicle Classes (Directly Useful to VisionRadar)**
* `Car` / `car`
* `Truck` / `truck`
* `Bus` / `bus`
* `Motorcycle` / `motorcycle`
* `Van` / `van`

### **Other Classes (Filtered Out)**
* `Person` / `pedestrian`
* `Bicycle`
* `Tricycle`
* `Traffic light` / `sign`

---

## 7. VisionRadar Task Relevance Matrix

Descriptive usefulness scoring per dataset:

| Dataset Name | Detection | Tracking | Counting | Speed | Calibration | Anomaly | Indian Traffic | Fine-Tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vehicle-detection-project-main` | **YES** | NO | **YES** | NO | NO | NO | NO | **YES** |
| `training` | **YES** | NO | **YES** | NO | NO | NO | NO | **YES** |
| `valid-fine-tuning` | **YES** | NO | **YES** | NO | NO | NO | NO | **YES** |
| `valid-self-learning` | **YES** | NO | **YES** | NO | NO | NO | NO | **YES** |
| `Vehicle_Tracking` | **YES** | **YES** | **YES** | **YES** | **YES** | NO | **PARTIAL** | **YES** |
| `DETRAC-Train-Detections` | **YES** | **YES** | **YES** | NO | NO | NO | NO | NO |
| `DETRAC-Test-Tra` | NO | **YES** | **YES** | NO | NO | NO | NO | NO |
| `ITD-Indian-traffic-dataset-main` | NO | NO | NO | NO | NO | NO | **YES** | NO |
| `BrnoCompSpeed-master` | NO | NO | NO | **YES** | **YES** | NO | NO | NO |
| `Unconfirmed *.crdownload` | NO | NO | NO | NO | NO | NO | NO | NO |

---

## 8. Duplicate / Redundancy Findings

* **`training` / `valid-fine-tuning` / `valid-self-learning` vs `vehicle-detection-project-main`**:
  * All 4 datasets originate from Roboflow exports using standard 4-class COCO mapping (`0: bus, 1: car, 2: motorcycle, 3: truck`).
  * Filename hashing confirmed `0` duplicate filenames across sets (they represent independent train/val split batches).
* **DETRAC Repositories**:
  * `DETRAC-Train-Detections-*` contains pre-computed baseline detector outputs (ACF, CompACT, R-CNN) for training sequences.
  * `DETRAC-Test-Tra` contains benchmark trajectory output files and evaluation scripts. Neither folder contains original image sequence frames.

---

## 9. Incomplete Download Analysis

| Filename | Disk Size | Likely Source | Status | Action Required |
| :--- | :--- | :--- | :--- | :--- |
| `Unconfirmed 238577.crdownload` | $45.24\text{ MB}$ | Chrome browser download | **Incomplete** | Resume download or re-fetch source archive. |
| `Unconfirmed 241433.crdownload` | $22.63\text{ MB}$ | Chrome browser download | **Incomplete** | Resume download or re-fetch source archive. |
| `Unconfirmed 421580.crdownload` | $25.86\text{ MB}$ | Chrome browser download | **Incomplete** | Resume download or re-fetch source archive. |

---

## 10. Deep-Dive Inspections: BrnoCompSpeed, ITD, DETRAC

### **10.1 BrnoCompSpeed (`BrnoCompSpeed-master`)**
* **Contents**: Contains evaluation Python scripts (`evaluate.py`, `speed_metrics.py`), dataset parser scripts, and checksum files.
* **Video/Image Files**: **None present**.
* **Ground-Truth Files**: Parser scripts present; actual distance/speed ground-truth `.mat` / `.json` files are fetched remotely by the python scripts.
* **Calibration Info**: Homography calibration parsing logic is present in Python scripts.

### **10.2 ITD (`ITD-Indian-traffic-dataset-main`)**
* **Contents**: 10 high-resolution sample images ($1440\times 1080$ to $3000\times 2250$) representing Indian traffic scenes (auto-rickshaws, motorcycles, crowded intersections).
* **Annotations**: **None present** (sample repository from IEEE paper).
* **YOLOX Fine-Tuning**: Cannot be used immediately for supervised fine-tuning without manual annotation or pseudo-labeling.

### **10.3 DETRAC (`DETRAC-Train-Detections-*` & `DETRAC-Test-Tra`)**
* **Contents**: `DETRAC-Train-Detections` contains 242 baseline bounding box text files. `DETRAC-Test-Tra` contains 119 trajectory evaluation files and 46 MATLAB benchmark scripts.
* **Raw Video / Frame Sequences**: **None present**. Image frames were not downloaded alongside detection TXT files.

---

## 11. Recommended Future Experimentation Sequence

1. **Phase 5.1 (Dataset Standardization & Conversion)**: Convert Roboflow YOLO TXT annotations (`vehicle-detection-project-main` + `training`) to COCO JSON format.
2. **Phase 5.2 (YOLOX Fine-Tuning)**: Fine-tune `YOLOX-Nano` on combined vehicle dataset to improve motorcycle & truck detection recall on dense traffic clips.
3. **Phase 5.3 (Tracking Evaluation)**: Evaluate ByteTrack performance using `Vehicle_Tracking` MP4 video clips and ground-truth JSON trajectories.

---

## 12. Risks & Missing Information

1. **Missing Image Sequences for DETRAC & BrnoCompSpeed**: Both repositories contain evaluation scripts and detection bounding boxes, but lack the raw image frames/videos required for visual playback.
2. **Unannotated Indian Traffic Dataset**: `ITD` images need bounding box annotations before fine-tuning YOLOX on Indian traffic.
3. **Interrupted Browser Downloads**: 3 `.crdownload` files remain incomplete and cannot be extracted until downloads finish.
