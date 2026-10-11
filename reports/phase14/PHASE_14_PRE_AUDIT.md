# VISIONRADAR — PHASE 14 PRE-AUDIT REPORT

**Date:** 2026-09-27  
**Status:** COMPLETE  
**Auditor:** Antigravity AI Coding Assistant  

---

## 1. Executive Summary

Before implementing Phase 14 Advanced Traffic Anomaly & Behavioral Intelligence, a comprehensive code audit of all existing packages, modules, schemas, data models, APIs, and tests was conducted to prevent code duplication, preserve existing API contracts, and build upon established perception and traffic intelligence layers.

---

## 2. Existing Perception & CV Stack

- **Detector:** `YOLOX-Nano ONNX` (`data/models/yolox_nano.onnx`, SHA-256: `c789161ed...`)
- **Tracker:** `ByteTrackTracker` (`packages/visionradar/cv/tracker.py`) — Single persistent multi-object tracking instance.
- **Speed Estimator:** `MonocularSpeedEstimator` (`packages/visionradar/perception/speed.py`) — IPM homography speed estimation with explicit status flagging (`VALID`, `OUT_OF_ROI`, `INSUFFICIENT_DATA`, `CALIBRATION_REQUIRED`).
- **Accident Classifier:** `AccidentClassifierService` (`packages/visionradar/perception/accident_classifier.py`) — Secondary candidate-frame verifier invoked lazily only when candidate frames exist.

---

## 3. Existing Traffic Intelligence Stack

- **Vehicle Counting Engine:** `VehicleCountingEngine` (`packages/visionradar/intelligence/traffic/counting.py`)
- **Traffic Flow Engine:** `TrafficFlowEngine` (`packages/visionradar/intelligence/traffic/flow.py`)
- **Lane Intelligence Engine:** `LaneIntelligenceEngine` (`packages/visionradar/intelligence/traffic/lanes.py`)
- **Vehicle Density Engine:** `VehicleDensityEngine` (`packages/visionradar/intelligence/traffic/density.py`)
- **Queue Detector:** `QueueDetector` (`packages/visionradar/intelligence/traffic/queue.py`)
- **Congestion Engine:** `CongestionEngine` (`packages/visionradar/intelligence/traffic/congestion.py`)

---

## 4. Existing Event Engine

- **Event Engine Core:** `TrafficEventEngine` (`packages/visionradar/intelligence/events/engine.py`) with temporal state machines (`TrackEventState`).
- **Event Detectors:**
  - `StoppedVehicleDetector` (`stopped_vehicle.py`): Stationary vehicle detection ($\le 3.0\text{ km/h}$, $30$ frames).
  - `WrongWayDetector` (`wrong_way.py`): Opposing trajectory flow detection.
  - `SuddenDecelerationDetector` (`deceleration.py`): Deceleration spikes ($\ge 25\text{ km/h/s}$).

---

## 5. Existing Anomaly & Behavioral Intelligence Stack

- **Schemas:** `packages/visionradar/intelligence/anomaly/schemas.py` defining `TrajectoryFeatures`, `TrajectoryAnomaly`, `CongestionTransition`, `VehicleBehaviorProfile`, `TrafficRiskScore`.
- **Feature Engine:** `TrajectoryFeatureEngine` (`features.py`) — 18 raw trajectory features.
- **Behavioral Anomaly Engine:** `BehavioralAnomalyEngine` (`engine.py`) — Rule-based and robust Z-score statistical anomaly scoring.
- **Congestion Transition Engine:** `CongestionTransitionEngine` (`congestion_transitions.py`) — State transitions (`FREE_FLOW`, `MODERATE`, `CONGESTED`, `STOP_AND_GO`).
- **Vehicle Behavior Profile Engine:** `VehicleBehaviorProfileEngine` (`behavior.py`) — Summary profiles per vehicle.
- **Traffic Risk Scoring Engine:** `TrafficRiskScoringEngine` (`risk.py`) — Aggregate frame risk index ($0 - 100$).

---

## 6. Existing Evidence & Database Models

- **Evidence Package Writer:** `EvidenceWriter` (`packages/visionradar/intelligence/violations/evidence.py`) creating `data/evidence/job_{job_id}/...` with `before.jpg`, `event.jpg`, `after.jpg`, and `metadata.json`.
- **Database Models:** SQLite ORM models (`Job`, `Violation`, `Evidence`, `TrafficEvent`) in `apps/api/models.py`.

---

## 7. Existing REST API & Frontend Workbench

- **REST Endpoints:** FastAPI app in `apps/api/main.py` exposing `/api/v1/jobs/{job_id}`, `/api/v1/jobs/{job_id}/events`, `/api/v1/jobs/{job_id}/accident-candidates`, `/api/v1/jobs/{job_id}/anomaly-report`.
- **Frontend Workbench:** React component (`apps/web/src/components/WorkbenchView.tsx`) with detection canvas, real-time overlays, traffic metrics bar, accident verifier panel, and anomaly view.

---

## 8. Integration Plan for Phase 14

1. **Extend Trajectory Feature Engine (Part 2):** Add formal feature metadata schema (value, unit, source, validity, quality) to every extracted feature.
2. **Establish Statistical Baseline (Part 3 & 5):** Build robust statistical baseline (median, MAD, z-score, local lane/traffic context) without black-box ML.
3. **Expand Vehicle & Traffic-Level Anomaly Engine (Parts 4, 8, 9, 10, 11):** Implement formal Anomaly State Machines, multi-vehicle interaction anomalies, traffic-level flow/density/queue anomalies, and multi-signal event correlation.
4. **Explainability & Evidence (Parts 6, 7, 12):** Include explicit feature contributions, human-readable explainability strings, and structured evidence packaging.
5. **Experiments & Benchmarking (Parts 13, 14, 15, 16, 17):** Conduct ablation experiments (trajectory vs lane vs traffic context), sensitivity sweeps, real-video benchmark runs, and latency measurements.
6. **Frontend & API (Parts 18, 19):** Expose detailed anomaly reports via REST API and render dedicated Anomaly Intelligence timeline/cards in Workbench UI.
7. **Testing (Part 20):** Create comprehensive unit test suite `tests/test_phase14_anomaly_intelligence.py` and run full suite (180+ tests) + `npm run build`.

---
*Pre-audit completed successfully. Ready for Phase 14 implementation.*
