# VisionRadar Product Recovery — Architecture Assessment

## 1. Architectural Philosophy & Vision

VisionRadar is designed as an edge-feasible, uncertainty-aware vision intelligence platform. Rather than attempting to run oversized deep learning models that lock up consumer or edge hardware, VisionRadar combines:
1. **Lightweight Edge Detection**: Quantizable YOLOX-Nano ONNX.
2. **Deterministic Tracking**: ByteTrack bounding box association.
3. **Mathematical Rigor**: Planar homography perspective transformation mapping wheel contact points to real-world meters.
4. **Decoupled Architecture**: High-frame-rate display (~60 FPS) decoupled from AI inference (~7.1 FPS).
5. **Uncertainty-Aware Metrics**: Speed measurements bound to calibration validity with error decomposition.

---

## 2. Component Architecture & Data Flow

```
[ Video / Camera Feed ]
          │
          ▼
 [ VideoDecoder (OpenCV) ]
          │
          ├────────────────────────────────────────┐
          │ (Raw Frames)                           │ (Playback Feed)
          ▼                                        ▼
[ YOLOX-Nano ONNX Detector ]             [ HTML5 <video> Element ]
          │ (BBoxes: x1,y1,x2,y2)                  │
          ▼                                        │ (~60 FPS render)
  [ ByteTrack Tracker ]                            │
          │ (Track IDs)                            ▼
          ▼                              [ Canvas 2D Overlay ]
[ HomographyCalibrator ]                 (Nearest-frame sync)
          │ (Pixels -> Meters)                     ▲
          ▼                                        │ (WebSocket / REST)
[ SpeedEstimator (Δd / Δt) ]                       │
          │                                        │
          ▼                                        │
[ SQLite WAL Database ] ───────────────────────────┘
```

---

## 3. Modular Product Readiness Assessment

| Module | Purpose | Status in Codebase | Readiness for Production |
| :--- | :--- | :--- | :--- |
| **Ingestion Engine** | Video upload, SHA-256 integrity, OpenCV decoding | Implemented | **Ready** (handles corrupt files & formats) |
| **Worker Engine** | Threaded worker with cancellation and queue backpressure | Implemented | **Ready** (Scenarios A–H verified) |
| **Detection Engine** | Vehicle detection (car, truck, bus, motorcycle) | Implemented (YOLOX-Nano) | **Ready** (Edge CPU baseline verified) |
| **Tracking Engine** | ByteTrack ID association | Implemented | **Ready** (Occlusion handling verified) |
| **Calibration Engine**| 4-point homography, condition number check, RMSE | Implemented | **Ready** (Guardrails prevent invalid speeds) |
| **Speed Engine** | Monocular speed estimation with rolling window | Implemented | **Ready** (Tri-state validation enforced) |
| **Traffic Analytics** | Density, flow, queue estimation, classification | Implemented | **Operational** (requires calibrated camera) |
| **Reporting Engine** | PDF report compilation with violation evidence | Implemented (`pdf_report.py`) | **Operational** |
| **Hardware Sensor** | Edge NPU / Coral Edge TPU acceleration | Architecture & Reference | **Edge firmware integration needed** |
| **General Vision** | Pedestrians, road hazards, fire/smoke, debris | Architecture proposed | **Future milestone** |

---

## 4. Commercial Product Thinking for Real Organizations

To serve real transport consultants, traffic research institutes, and fleet operators:

1. **Practical Commercial Workflow**:
   - **Upload & Verification**: User uploads traffic surveillance video. Integrity is verified.
   - **Calibration Setup**: User defines 4 road points with known real-world distances (e.g. lane markers 10m apart).
   - **Automated Processing**: Asynchronous worker processes video with honest progress bars.
   - **Interactive Workbench**: User reviews traffic flow, speed distribution, and individual vehicle trajectories with zero video freezing.
   - **Audit-Ready Evidence**: Export PDF reports containing cropped violation evidence, timestamps, and mathematical uncertainty bounds.

2. **Commercial Value Proposition**:
   - Eliminates need for expensive physical radar or lidar units ($15k–$50k per intersection).
   - Turns standard CCTV cameras into calibrated speed and traffic intelligence sensors.
   - Honest, legally defensible metrics: speed is marked `UNCALIBRATED_MOTION` if camera geometry is not established, preventing false enforcement actions.
