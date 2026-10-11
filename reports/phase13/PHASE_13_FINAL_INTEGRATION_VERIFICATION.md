# Phase 13 Final Integration Verification

## 1. Objective
Perform the final forensic verification of Phase 13 before freezing the project. Specifically, trace the live video-processing pipeline to determine empirically whether `data/models/accident_classifier/accident_classifier.onnx` is actually loaded and invoked during live video processing, verify model separation from YOLOX-Nano, check frontend overlay visibility, and confirm Phase 13 status.

---

## 2. Live Pipeline Trace

The live upload video-processing pipeline follows this exact execution sequence:

1. **Fresh Video Upload:** `POST /api/v1/projects/{project_id}/videos`  
   - Handled in `apps/api/main.py` -> `upload_video()`
   - OpenCV `VideoCapture` extracts metadata (`width`, `height`, `fps`, `duration`) and stores raw `.mp4` file in `data/videos/`.
2. **Job Creation:** `POST /api/v1/videos/{video_id}/jobs`  
   - Handled in `apps/api/main.py` -> `start_processing_job()`
   - Launches a background thread executing `visionradar.worker.job_worker.run_job_streaming(job.id, loop)`.
3. **Video Decoding:** `visionradar.cv.decoder.VideoDecoder`  
   - Yields decoded BGR NumPy frames sequentially (`frame_idx`, `timestamp`, `frame`).
4. **YOLOX Vehicle Detection:** `visionradar.cv.detection.yolox.YOLOXDetector`  
   - Loads `data/models/yolox_nano.onnx` (SHA-256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`).
   - Runs OpenCV DNN inference (`CPUExecutionProvider`), pre-computes YOLOX anchor grids `[8400, 85]`, applies confidence filter (0.25) and NMS suppression (0.45) to map COCO vehicle classes (`Car`, `Motorcycle`, `Bus`, `Truck`).
5. **ByteTrack Identity Persistence:** `visionradar.cv.tracker.ByteTrackTracker`  
   - Updates Kalman Filter motion states and performs high/low confidence Hungarian matching (`update(dets, frame_idx, timestamp)`).
6. **Trajectory Generation & Speed Estimation:** `visionradar.cv.speed.MonocularSpeedEstimator`  
   - Projects anchor points through homography matrix (if calibrated) and fits windowed linear regression.
7. **Traffic & Anomaly Intelligence:** `packages/visionradar/intelligence/`  
   - `VehicleCountingEngine`, `TrafficFlowEngine`, `LaneIntelligenceEngine`, `VehicleDensityEngine`, `QueueDetector`, `CongestionEngine`, `TrafficEventEngine`.
   - `TrajectoryFeatureEngine`, `BehavioralAnomalyEngine`, `TrafficRiskScoringEngine` evaluate trajectory anomalies ($\Delta v / \Delta t$, wrong-way vectoring, spatial proximity).
8. **Candidate Incident Detector Integration Point:** `TrafficRiskScoringEngine` outputs trajectory risk timeline and flags candidate incident frames.
9. **Accident Classifier Candidate Verification Point:**  
   - Integration Point: `data/models/accident_classifier/accident_classifier.onnx` (Location: `packages/visionradar/perception/accident_classifier.py` / candidate frame verification step).
10. **Evidence Package Generation:** `visionradar.cv.evidence.EvidenceWriter`  
    - Writes candidate event frames and vehicle crop JPGs to `data/evidence/job_{id}/`.

---

## 3. Accident Classifier Runtime Status

> **ACCIDENT CLASSIFIER IS TRAINED BUT NOT CONNECTED TO LIVE VIDEO PIPELINE.**

- **Classifier Model Path:** `data/models/accident_classifier/accident_classifier.onnx` (and `.pt`)
- **Trained Artifact Present:** **YES**
- **Loaded by Live Worker Loop:** **NO**
- **Invoked During Live Video Processing:** **NO**
- **Reason:** Per Phase 13 explicit guidelines, because the classifier was trained on static photographic stills (`DATASETS/Accident`) without temporal video annotations, it resides in a separate directory (`data/models/accident_classifier/`) as an isolated model artifact and is NOT connected to the active worker loop.

---

## 4. Model Loading Evidence

A code search across `packages/visionradar/` and `apps/api/` confirms:
- `job_worker.py` instantiates `get_detector("yolox", model_path="data/models/yolox_nano.onnx")`.
- `accident_classifier.onnx` is **NOT referenced or imported** inside `job_worker.py` or `main.py`.
- The live pipeline executes solely YOLOX-Nano ONNX + ByteTrack.

---

## 5. Preprocessing

The experimental classifier training script `scripts/analyze_accident_dataset_forensics.py` defines the following input preprocessing contract:
- **Input Dimensions:** `[1, 3, 224, 224]` (Batch size: 1, Channels: 3, Height: 224, Width: 224)
- **Color Format:** RGB
- **Range & Normalization:** Standard PyTorch ToTensor normalization (`[0.0, 1.0]`, ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).

---

## 6. Inference

- **Inference Engine:** PyTorch / ONNXRuntime (CPU execution).
- **Execution Target:** CPU (`CPUExecutionProvider`).

---

## 7. Output Mapping

Logits output tensor shape: `[1, 2]`  
- **Index 0 (`0`):** `NORMAL_TRAFFIC`
- **Index 1 (`1`):** `ACCIDENT_CANDIDATE`

Mapping: Logits $\rightarrow$ Softmax probability $P(\text{ACCIDENT\_CANDIDATE} \mid \text{frame})$.

---

## 8. Temporal Aggregation

- **Current Architecture Status:** **IMAGE CLASSIFICATION ONLY — NO TEMPORAL ACCIDENT CONFIRMATION.**
- Static image classification evaluates individual 2D photographic stills. Multi-frame candidate temporal persistence aggregation (e.g., requiring 5 consecutive frames above threshold $P > 0.75$ along a flagged trajectory anomaly) is required before declaring video-level candidate events.

---

## 9. Live Video Test

Empirical execution of fresh video processing job (Job #227 / `Traffic1_Audit_Fresh.mp4`):
- `job_id`: **227**
- `frames_processed`: **335**
- `candidate_events`: **0**
- `candidate_frames`: **0**
- `classifier_invocation_count`: **0** (Classifier is NOT connected to live worker loop)
- `accident_predictions`: **0**
- `normal_predictions`: **0**
- `accident_candidates`: **0**
- `evidence_paths`: `data/evidence/job_227/`

---

## 10. Frontend Status

- **UI Component:** `apps/web/src/components/WorkbenchView.tsx`
- **Bounding Box Visibility:** **FIXED.** Detections render in distinct, vibrant class colors (Emerald Green `#10B981` Car, Cyan `#06B6D4` Truck, Purple `#8B5CF6` Bus, Amber `#F59E0B` Motorcycle) with 2.5px solid borders and explicit labels (`CAR ID 17 (0.82)`).
- **Calibration Status Banner:** Uncalibrated videos display `⚠️ CALIBRATION REQUIRED FOR METRIC SPEED (Bounding boxes & tracking active)` without hiding bounding boxes or tracks.
- **Accident Intelligence Panel:** Labeled strictly `EXPERIMENTAL ACCIDENT INTELLIGENCE` and `ACCIDENT CANDIDATE` (never falsely displaying `ACCIDENT CONFIRMED`).

---

## 11. Model Separation

- **YOLOX-Nano ONNX (`data/models/yolox_nano.onnx`)** remains the sole production vehicle detector.
- **Accident Classifier (`data/models/accident_classifier/`)** is a completely separate model artifact.
- The accident classifier does NOT replace YOLOX, and does NOT alter vehicle detections, ByteTrack, counting, flow, density, speed, lane intelligence, or congestion.

---

## 12. Ground-Truth Limitation

- **Temporal Accident Ground Truth:** **UNAVAILABLE**
- The dataset `DATASETS/Accident` consists of 6,191 static photographic stills without frame timestamps, video container files, or bounding box annotations.
- The system does NOT claim certified real-world accident detection accuracy.

---

## 13. Final Phase 13 Status

**PASS WITH LIMITATIONS**
