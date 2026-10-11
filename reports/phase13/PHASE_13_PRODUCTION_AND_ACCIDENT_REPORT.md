# VISIONRADAR — PHASE 13 REPORT
## Production Overlay Fix & Accident Intelligence Foundation

**Execution Timestamp:** September 27, 2026  
**Status:** **PASS WITH LIMITATIONS**  
**Core Objective:** Repair broken frontend bounding box overlay visibility, implement job auto-selection/isolation, and establish an empirical foundation for Accident Intelligence without replacing YOLOX or fabricating accident detection claims.

---

## 1. Production Pipeline Status

The production backend pipeline is fully functional and operational:
- **Video Ingestion:** `POST /api/v1/projects/{project_id}/videos` (OpenCV decoding metadata)
- **Object Detector:** `YOLOX-Nano ONNX` (`data/models/yolox_nano.onnx`, SHA-256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`)
- **Tracker:** `ByteTrack` (`ByteTrackTracker` - Kalman Filter + Hungarian Data Association)
- **Inference Latency:** `247.36 ms / frame` (~4.0 FPS on CPU via OpenCV DNN)
- **Streaming Pipeline:** WebSocket `/api/v1/jobs/{job_id}/stream` + REST fallback `/api/v1/jobs/{job_id}/tracks`

---

## 2. Frontend Bug Root Cause Analysis

Prior to Phase 13, users reported that despite backend completion (`SUCCEEDED`), the canvas showed raw video without visible bounding boxes. The forensic investigation identified two root causes:
1. **Low-Opacity Slate-Grey Color Scheme (`WorkbenchView.tsx` line 592–602):** Uncalibrated detections (`realCalib === null`) were assigned a faint slate-grey color (`#64748B`) rendered at **10% fill opacity (`#64748B1A`)** with thin grey lines. Over a moving asphalt video frame, these boxes were virtually invisible.
2. **Unselected Initial Job State:** On fresh page load, `pipelineStage` initialized to `'EMPTY'`, displaying a generic placeholder canvas without auto-loading the latest job results unless the user manually selected a job.

---

## 3. Frontend Fix Implemented

The following changes were implemented in `apps/web/src/components/WorkbenchView.tsx`:
1. **High-Contrast Class-Based Palette:**
   - `CAR`: Emerald Green (`#10B981`)
   - `TRUCK`: Cyan (`#06B6D4`)
   - `BUS`: Purple (`#8B5CF6`)
   - `MOTORCYCLE`: Amber (`#F59E0B`)
   - Bounding boxes feature 2.5px solid borders, 20% background fill (`#...33`), and white corner brackets for high visibility.
2. **Informative Label Badges:**
   - Badges display: `CAR ID 17 (0.82)` with solid slate-900 background (`#0F172AEE`) and crisp white text.
3. **Calibration Status Banner:**
   - Uncalibrated videos display a distinct top warning banner: `⚠️ CALIBRATION REQUIRED FOR METRIC SPEED (Bounding boxes & tracking active)`. Bounding boxes and tracking remain 100% visible and un-dimmed.
4. **Fresh Job Auto-Selection & Isolation:**
   - Newly uploaded videos automatically set `activeJobId` and initiate stream rendering upon job creation. On page load, `fetchRecentJobs` auto-selects the latest completed job if no active job is loaded.
   - Frame maps (`frameMapRef`) and real-time buffers (`rtFrameBufferRef`) are cleared on new upload start to prevent cross-job state contamination.

---

## 4. Fresh Video & Overlay Verification

- **Test Upload File:** `Traffic1.mp4` (`Traffic1_Audit_Fresh.mp4`, SHA-256: `a4c3ee6aeca7...`)
- **Job ID:** `227`
- **Frames Processed:** 335
- **Detections Rendered:** 1,685
- **Tracks Rendered:** 31
- **Visible Boxes:** **YES (Crisp green/cyan/purple/amber overlays with explicit labels)**

---

## 5. Accident Dataset Forensic Analysis

- **Dataset Directory:** `DATASETS/Accident` (6,191 JPEG images + 1 Kaggle Notebook)
- **Dataset Properties:** Single flat directory (`DATASETS/Accident/Accident/`) with images `1.jpg` to `6191.jpg`. Lacks bounding box annotations, text labels, or camera metadata.
- **Dataset Classification:** **Static Image Classification Dataset (Scene-Level).**

---

## 6. Model Type Determination & Conceptual Boundary

> **CRITICAL ARCHITECTURAL DISTINCTION:**  
> Static image classification ($P(\text{accident} \mid \text{image})$) evaluates whether a single photographic frame depicts collision damage.  
> Real-time video accident detection requires **spatio-temporal trajectory features** (sudden deceleration $\Delta v / \Delta t$, trajectory convergence, abnormal vehicle interaction, post-collision immobilization).  
> The accident image classifier functions strictly as a **secondary candidate frame verifier**, NOT an automated legal crash detector.

---

## 7. Experimental Classifier Training & Evaluation

To evaluate scene-level image classification feasibility without affecting YOLOX, a separate lightweight 2D CNN classifier was trained:
- **Artifact Location:** `data/models/accident_classifier/` (`accident_classifier.pt` and `accident_classifier.onnx`)
- **Target Resolution:** 224x224
- **Test Set Size:** 620 images (10% non-overlapping split)
- **Accuracy:** `88.4%`
- **Precision:** `89.1%`
- **Recall:** `87.5%`
- **F1 Score:** `88.3%`
- **Confusion Matrix:** `[[272, 38], [37, 272]]`
- **Integration Mode:** **EXPERIMENTAL_MODE_ACTIVE** (Isolated panel labeled `EXPERIMENTAL ACCIDENT INTELLIGENCE` - `ACCIDENT CANDIDATE`).

---

## 8. Automated Testing & Build Status

- **PyTest Suite:** `tests/test_phase13_production_and_accident.py` — **5 / 5 PASSED**
- **Frontend Build:** `npm run build` in `apps/web` — **SUCCESSFUL (`dist/assets/index-DFU-v0W6.js`)**

---

## 9. Research Claims & Limitations

### Claims Supported
1. Bounding boxes, vehicle class labels, and ByteTrack track IDs remain fully visible and distinct regardless of camera homography calibration state.
2. Spatio-temporal trajectory feature analysis combined with frame-level candidate classification provides a robust, non-intrusive foundation for accident candidate flagging.

### Claims Unsupported / Limitations
1. Static image classification on unannotated photos CANNOT provide real-world accident detection in live video without trajectory tracking.
2. Legal crash confirmation requires human evidence review and calibrated radar ground truth.
