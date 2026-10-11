# VisionRadar Phase 9A — Dataset Capability Matrix

## 1. Overview & Inventory
A systematic audit of all locally available datasets in `DATASETS/` was conducted. The local workspace contains **6 distinct dataset assets** spanning static image detection, video tracking benchmarks, speed ground truths, and evaluation toolkits.

## 2. Dataset Capability Matrix

| Dataset | Images | Videos | Annotations Format | Classes (Count) | Split | Detection GT | Tracking GT | Speed GT | Indian Traffic? | Usable Train | Usable Eval |
| :--- | :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **vehicle-detection-project-main** | 874 | 0 | YOLO txt (normalized cx, cy, w, h) | bus, car, motorcycle, truck (4) | Train (714), Val (80), Test (80) | YES | NO | NO | NO | YES | YES |
| **training / valid-fine-tuning / valid-self-learning** | 134 | 0 | YOLO txt (normalized cx, cy, w, h) | Class IDs 0,1,2,3,4 (Bus, Car, Motorcycle, Truck, Van) (5) | Train (117), Val-Fine (11), Val-Self (6) | YES | NO | NO | NO | YES | YES |
| **Vehicle_Tracking (VNTraffic & AICC22-Custom)** | 0 | 4 | MOT JSON (track_id, frame, bbox) | Vehicle (Generic) (1) | Video Sequences (VNTraffic 2x, AICC22-Custom 2x) | YES | YES | NO | YES | NO | YES |
| **DETRAC-Train-Detections / DETRAC-Test-Tra** | 0 | 0 | MOT / DETRAC TXT Detections | car, bus, van, others (4) | Train Detections (242 TXTs), Test Sequences (119 TXTs) | YES | NO | NO | NO | NO | YES |
| **BrnoCompSpeed-master** | 0 | 0 | BrnoCompSpeed PKL / TXT speed GT | Vehicle (Generic) (1) | Benchmark Evaluation Scripts | NO | NO | YES | NO | NO | YES |
| **ITD-Indian-traffic-dataset-main** | 10 | 0 | Documentation PNGs | N/A (0) | Documentation Assets | NO | NO | NO | YES | NO | NO |

## 3. Detailed Audit Summaries

### 3.1 vehicle-detection-project-main
- **Content**: 874 images annotated in standard YOLO format (`class cx cy w h`).
- **Classes**: `0: bus`, `1: car`, `2: motorcycle`, `3: truck`.
- **Splits**: 714 training images, 80 validation images, 80 test images.
- **Utility**: High-quality benchmark for detection evaluation and candidate fine-tuning.

### 3.2 training / valid-fine-tuning / valid-self-learning
- **Content**: 134 total images annotated in YOLO format.
- **Classes**: 5 classes (`0: bus`, `1: car`, `2: motorcycle`, `3: truck`, `4: van`).
- **Utility**: Supplemental validation set for multi-class vehicle detection.

### 3.3 Vehicle_Tracking (VNTraffic & AICC22-Custom)
- **Content**: 4 high-definition traffic surveillance video files (`Traffic1.mp4`, `VNTraffic_Original-video.mp4`, `AICC22-Custom_Original-video.mp4`, `synthetic_highway.mp4`) with accompanying MOT JSON ground truth files.
- **Utility**: Evaluation of video detection, ByteTrack tracking continuity, and traffic counting under complex real-world conditions.

### 3.4 DETRAC-Train-Detections & DETRAC-Test-Tra
- **Content**: Pre-computed baseline bounding box text files (ACF, CompACT, DPM, R-CNN) and sequence evaluation files.
- **Utility**: Reference annotations for sequence-level evaluation.

### 3.5 BrnoCompSpeed-master
- **Content**: Python codebase and ground truth speed metrics for BrnoCompSpeed benchmark.
- **Utility**: Ground truth speed calculation reference.

### 3.6 ITD-Indian-traffic-dataset-main
- **Content**: Documentation assets and static confusion matrix charts.
- **Utility**: Reference documentation only.