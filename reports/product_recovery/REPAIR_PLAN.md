# VisionRadar Product Recovery — Repair Plan & Product Roadmap

## 1. Executive Summary

This repair plan outlines the verified engineering repairs implemented during the product recovery milestone, immediate low-risk enhancements, and the structured roadmap to commercial readiness.

---

## 2. Completed & Verified Repairs

| Component | Defect | Root Cause | Solution Implemented | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Worker Engine** | Job cancellation ignored | Missing cancellation checks | Thread registry + `_cancel_events` + pre-loop checks | **VERIFIED** |
| **Worker Engine** | Zombie jobs on server restart | No startup reconciliation | Startup hook marks stale jobs `FAILED` | **VERIFIED** |
| **Worker Engine** | WebSocket memory leak | Unbounded queue | Bounded `Queue(50)` + terminal message preservation | **VERIFIED** |
| **Workbench UI** | Stale boxes on seek | `latestInferenceFrame` bypass | Strict temporal proximity `|closest - current| <= 30` | **VERIFIED** |
| **Workbench UI** | Visual flicker on job finish | Premature buffer clear | Atomic buffer handoff after DB indexing | **VERIFIED** |
| **Workbench UI** | UI clutter in crowd scenes | Simultaneous label render | `Minimal`, `Standard`, `Intelligence` visualization modes | **VERIFIED** |
| **Calibration** | Impossible speed calculation | Unchecked homography matrix | Condition number < 1e8 and RMSE < 1.0m enforced | **VERIFIED** |
| **Speed Engine** | Uncalibrated speed claims | Missing calibration check | Tri-state: `CALIBRATED_SPEED`, `UNCALIBRATED_MOTION` | **VERIFIED** |

---

## 3. Near-Term Stability Enhancements

1. **Explicit Multi-Job Rejection / Single-Active Policy**:
   - For single-workstation deployments, enforce a clear UI notification when a new job is queued or supercedes an existing one.
2. **Explicit Calibration Prompting**:
   - In Workbench, if a video has no calibration, display an informative banner guiding the user to the Calibration tool before expecting speed metrics.
3. **Automated Evidence Pruning**:
   - Implement scheduled disk cleanup for old violation snapshot images in `data/evidence/` to prevent unbounded disk growth.

---

## 4. Long-Term Commercial Roadmap

1. **Hardware-Accelerated YOLO Sensor (INT8 NPU/VPU)**:
   - Provide INT8 quantized models for Coral Edge TPU / RK3588 NPU, reducing inference latency from 138 ms to < 15 ms, enabling 30+ FPS live stream processing.
2. **Modular General Vision Architecture**:
   - Introduce pluggable model heads for pedestrian safety, wrong-way detection, and road hazard identification without bloating the core traffic detector.
3. **High-Scale Message Sink**:
   - Allow enterprise users to route violation events to Redis / Kafka / Postgres rather than local SQLite for city-scale deployments.
