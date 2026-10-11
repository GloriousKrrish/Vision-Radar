# VisionRadar — Phase 6 Baseline Perception Stack Audit

**Date:** 2026-09-26  
**Auditor:** Antigravity Perception Engineering Team  
**Scope:** Complete End-to-End Perception Pipeline Baseline Audit  
**Production Model:** `data/models/yolox_nano.onnx` ($640\times 640$ Letterboxed, CPU Execution Engine)

---

## 1. End-to-End Execution Trace

The production perception pipeline follows a strictly sequential, single-threaded processing loop (`PerceptionPipeline` in `packages/visionradar/perception/pipeline.py`):

```
[Video File / Stream]
       │
       ▼
1. Capture & Decode (VideoSource / OpenCV VideoCapture)
       │ (FrameItem: frame BGR, frame_id, capture_ts)
       ▼
2. Preprocessing (letterbox_preprocess to 640x640, BGR uint8 -> float32)
       │ (blob [1, 3, 640, 640], scale, pad)
       ▼
3. YOLOX ONNX Inference Engine (onnxruntime CPUExecutionProvider, intra_op_threads=8)
       │ (raw_outputs [1, 8400, 85])
       ▼
4. Postprocessing & NMS (postprocess_yolox_output: sigmoid, conf_thresh=0.25, nms_thresh=0.45)
       │ (List[VehicleDetection])
       ▼
5. Multi-Object Tracker (PerceptionTracker / ByteTrackTracker, max_age=30, iou_thresh=0.3)
       │ (List[TrackedVehicle] -> List[VehicleTrack])
       ▼
6. Object Lifecycle Management (ObjectLifecycleManager: confirmation_frames=2, grace_window=15)
       │ (confirmed_tracks: List[VehicleTrack], lifecycle_events: List[ObjectLifecycleEvent])
       ▼
7. FrameResult Assembly & Metrics Recording (FrameResult schema)
       │ (result_ts, stage_timings_ms)
       ▼
8. Broadcast Push (Asyncio Queue -> WebSocket /api/v1/jobs/{job_id}/stream)
       │
       ▼
9. Frontend Workbench View (WorkbenchView.tsx / WebGL Canvas overlay & Track Inspector)
```

---

## 2. Component-by-Component Baseline Audit

### 2.1 Preprocessing (`preprocess.py`)
* **Aspect Ratio Maintenance**: Uses letterbox padding with grey fill (`114, 114, 114`).
* **Target Resolution**: Fixed $640\times 640$ spatial resolution.
* **Input Tensor**: `[1, 3, 640, 640]` float32 array in `(B, C, H, W)` layout.
* **Scale and Pad Parameters**: Extracted per-frame `(scale, (pad_w, pad_h))` for coordinate mapping back to original frame dimensions.

### 2.2 Model & Inference (`inference_engine.py`)
* **Weights**: `data/models/yolox_nano.onnx` (SHA-256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`).
* **Runtime**: `onnxruntime.InferenceSession` with `CPUExecutionProvider`.
* **Execution Latency**: $\approx 37 - 45\text{ ms}$ per frame on CPU.
* **Output Tensor**: Single output tensor `[1, 8400, 85]` containing $8400$ grid anchors across strides 8, 16, and 32.

### 2.3 Postprocessing & NMS (`postprocess.py`)
* **Score Decoding**: Objectness and COCO class probabilities are sigmoided.
* **Filtering**: Confidence threshold `0.25`.
* **COCO Class Filter**: Retains COCO indices 2 (`car`), 3 (`motorcycle`), 5 (`bus`), 7 (`truck`).
* **NMS**: Per-class Non-Maximum Suppression at IoU `0.45`.
* **Output**: `List[VehicleDetection]`.

### 2.4 Multi-Object Tracking (`tracker.py` & `cv/tracker.py`)
* **Algorithm**: ByteTrack multi-object tracking.
* **Association Logic**: Two-stage association (high-confidence IoU greedy matching $\rightarrow$ low-confidence / distance fallback matching).
* **Track Expiry**: `max_age = 30` frames.
* **Current Limitations**:
  * Track IDs are assigned sequentially (`track_id = 1, 2, 3...`).
  * Basic velocity estimation ($dx, dy$ pixel delta over last 2 points).
  * No explicit detection quality metadata or trajectory validity flags.

### 2.5 Object Lifecycle Manager (`lifecycle.py`)
* **States**: `NEW`, `ACTIVE`, `LOST`, `REMOVED`.
* **Confirmation Threshold**: `confirmation_frames = 2`.
* **Grace Window**: `grace_window = 15` frames before removal.

### 2.6 Trajectory Data (`cv/trajectory.py`)
* **Storage**: `VehicleTrajectory` holds an ordered list of `TrajectoryPoint` (frame_index, timestamp, bbox, anchor_pixel).
* **Current Limitations**:
  * No validation flags for impossible motion, spatial jitter, or calibration boundary compliance.

### 2.7 Calibration Integration (`cv/calibration.py` / `cv/homography.py`)
* **Planar Calibration**: 4-point polygon selection mapped to world bird's-eye coordinates ($X, Y$ meters).
* **Current Gap**: Calibration validity is calculated during speed estimation, but trajectory points are not flagged for ROI membership or numerical homography stability in real-time perception metadata.

### 2.8 Frontend Workbench Inspector (`WorkbenchView.tsx`)
* **Rendering**: Canvas overlay for bounding boxes, track IDs, velocity vectors, and speed pill overlays.
* **Current Inspector Gap**: Selected track panel displays basic speed, lane, and direction, but lacks detailed perception quality indicators (quality score, class consistency, missing frame count, trajectory validity, motion anomaly flags).

---

## 3. Key Baseline Audit Findings & Required Phase 6 Improvements

1. **Detection Metadata Gap**: Raw detections currently lack fine-grained diagnostic quality flags (`TOO_SMALL`, `EDGE_CLIPPED`, `OUTSIDE_ROI`, `SUSPICIOUS_GEOMETRY`, `DUPLICATE_CANDIDATE`).
2. **Class Flip Dynamics**: Vehicles can occasionally flip classes across frames if raw detector confidence fluctuates. A track-level class consensus mechanism is needed without overwriting raw class data.
3. **Tracking & Occlusion Quality Diagnostics**: Missing explicit gap classification (`SHORT_GAP`, `MEDIUM_GAP`, `LONG_GAP`), track quality score breakdown, and clear separation between `OBSERVED_INTERNAL_ID_CHANGE` vs. `GROUND_TRUTH_MOT_ID_SWITCH`.
4. **Trajectory & Motion Safety**: No diagnostic guard for physically impossible jumps or spatial jitter before passing trajectories down to speed estimation.
5. **Calibration Polygon Safety**: Trajectory points must be explicitly marked as `CALIBRATION_VALID`, `OUT_OF_ROI`, `CALIBRATION_UNSTABLE`, or `WORLD_COORDINATE_INVALID`.

---

## 4. Phase 6 Architectural Guarantees

* **Detector Weights**: `data/models/yolox_nano.onnx` remains 100% UNTOUCHED.
* **Tracker**: ByteTrack remains the single underlying multi-object tracker.
* **Performance Overhead Constraint**: Quality Engine additions must add $<10\%$ latency overhead to total perception processing time.
* **Live Inference**: WebSocket streaming `/api/v1/jobs/{job_id}/stream` and real-time inference remain 100% operational.
