# Phase 15 — Research Workbench & Experiment Explainability Pre-Audit

## 1. Executive Summary
Phase 15 introduces a production-grade **Research Workbench and Experiment Explainability System** for VisionRadar. The system will enable researchers to register, record, execute, compare, explain, and export reproducible computer vision experiments backed by full provenance, configuration snapshots, metric classification, failure case tracking, and anomaly explainability.

---

## 2. Existing System Inventory & Capabilities

### Verified & Reusable Capabilities:
1. **Perception Engine (`packages/visionradar/cv/`, `perception/`):**
   - YOLOX-Nano ONNX detector (`data/models/yolox_nano.onnx`).
   - ByteTrack persistent multi-object tracker (`tracker.py`).
   - Monocular speed estimator with homography calibration (`speed.py`).
2. **Intelligence Engines (`packages/visionradar/intelligence/`):**
   - Traffic counting, flow, lane utilization, density, queue, and congestion engines (`traffic/`).
   - Traffic event detection engine (`events/`).
   - Trajectory feature extraction and Behavioral anomaly engine with robust Median & MAD baselines (`anomaly/`).
   - Secondary candidate frame verifier using ResNet18 accident classifier (`perception/accident_classifier.py`).
3. **Benchmarking & Dataset Adapter Infrastructure (`packages/visionradar/benchmarks/`, `eval/`):**
   - Dataset registry and adapters for CityFlow, AI City Challenge, VNTraffic, Synthetic Highway (`eval/dataset_adapters.py`, `dataset_registry.py`).
   - Speed ground-truth matcher (`benchmarks/matcher.py`).
4. **Data Storage & Evidence Infrastructure (`packages/visionradar/models/`, `cv/evidence.py`):**
   - SQLite database (`visionradar.db`) with `Project`, `Video`, `Calibration`, `ProcessingJob`, `Track`, `SpeedMeasurement`, `Violation`, `Evidence`, `BenchmarkExperimentDB`.
   - `EvidenceWriter` producing structured evidence packages (`before.jpg`, `event.jpg`, `after.jpg`, `metadata.json`).

---

## 3. Missing Capabilities Required for Phase 15

1. **Persistent Research Experiment Registry (`packages/visionradar/research/registry.py`):**
   - Unique, immutable experiment identifiers (e.g., `EXP-2026-000001`).
   - Complete metadata capture (`dataset_id`, `video_id`, `video_hash`, `model_id`, `model_hash`, `pipeline_version`, `software_version`, `configuration_hash`, `calibration_hash`, `duration`, `frame_count`, `fps`, `resolution`, `evaluation_protocol`, `ground_truth_status`, `result_status`).
2. **Immutable Configuration Snapshots (`packages/visionradar/research/config_snapshot.py`):**
   - Capture of detector, tracker, speed estimator, traffic intelligence, anomaly policy, and accident classifier configurations into a deterministic JSON snapshot.
3. **Structured Metric Provenance (`packages/visionradar/research/provenance.py`):**
   - Wrapping every metric value with explicit provenance: `metric_name`, `value`, `unit`, `source`, `experiment_id`, `calculation_method`, `ground_truth_status`, `confidence`, `notes`.
4. **Strict Four-Tier Results Classification:**
   - Every experiment metric and output classified strictly as `GT-VALIDATED`, `ENGINEERING-VALIDATED`, `QUALITATIVE`, or `NOT-EVALUABLE`.
5. **Scientific Experiment Comparison Engine (`packages/visionradar/research/comparison.py`):**
   - Side-by-side comparison of Experiment A vs. Experiment B with `UNCHANGED`, `CHANGED`, and `NOT AVAILABLE` deltas. Zero rankings, zero subjective "winner" scores.
6. **Real Ablation Workbench Engine (`packages/visionradar/research/ablation.py`):**
   - Execution and recording of parameter sensitivity and feature ablation experiments (confidence threshold, resolution, tracking max age, anomaly threshold, persistence frames, context filtering).
7. **Failure Case Registry & Workbench (`packages/visionradar/research/failure_cases.py`):**
   - Structured logging of failure cases across 11 standardized categories (`small distant object`, `occlusion`, `ID fragmentation`, `calibration instability`, `perspective distortion`, `detector miss`, `false detection`, `abnormal trajectory`, `anomaly false candidate`, `accident-classifier ambiguity`, `insufficient trajectory history`).
8. **Anomaly Explainability Engine (`packages/visionradar/research/explainability.py`):**
   - Deep explanation payloads detailing WHAT, WHEN, WHERE, WHICH, WHY (rule/feature contributions), HOW STRONG, CONTEXT, BASELINE (Median, MAD, Z-score), PERSISTENCE, POLICY, VALIDATION, and EVIDENCE.
9. **Automated Markdown Research Report Generator (`packages/visionradar/research/report_generator.py`):**
   - Generation of a 19-section research report directly from experiment JSON manifests with zero hardcoded values.
10. **REST API Extensions (`apps/api/routes_research.py`):**
    - Clean REST endpoints for experiments, comparison, ablations, failure cases, explainability, artifacts, and report generation.
11. **Frontend Research Workbench (`apps/web/src/components/ResearchWorkbench/`):**
    - High-density research lab aesthetic UI (light theme: white, slate, indigo, sky, emerald, amber, red) covering Experiments, Detail, Compare, Ablations, Failure Cases, Explainability, and Reproducibility.

---

## 4. Planned Files to Create & Modify

### New Files to Create:
- `packages/visionradar/research/__init__.py`
- `packages/visionradar/research/schemas.py`
- `packages/visionradar/research/registry.py`
- `packages/visionradar/research/config_snapshot.py`
- `packages/visionradar/research/provenance.py`
- `packages/visionradar/research/comparison.py`
- `packages/visionradar/research/ablation.py`
- `packages/visionradar/research/failure_cases.py`
- `packages/visionradar/research/explainability.py`
- `packages/visionradar/research/report_generator.py`
- `apps/api/routes_research.py`
- `apps/web/src/components/ResearchWorkbench.tsx` (or subdirectory of workbench views)
- `tests/test_phase15_research_workbench.py`
- `reports/phase15/PHASE_15_RESEARCH_WORKBENCH_REPORT.md`

### Files to Modify:
- `packages/visionradar/models/entities.py` (Add `ResearchExperiment`, `ResearchAblation`, `ResearchFailureCase` ORM tables)
- `apps/api/main.py` (Register `routes_research` router)
- `apps/web/src/App.tsx` (Add `Research Workbench` tab navigation)

### Files to Keep Untouched (Preserve Phase 1-14 Core Integrity):
- `data/models/yolox_nano.onnx`
- `packages/visionradar/cv/detector.py`
- `packages/visionradar/cv/tracker.py`
- `packages/visionradar/cv/speed.py`
- `packages/visionradar/intelligence/anomaly/engine.py`
- `packages/visionradar/worker/job_worker.py` (core job loop)

---

## 5. Architectural Flow Diagram

```
                              USER / RESEARCHER
                                      │
                                      ▼
                      Frontend Research Workbench UI
                        (React / Vite Light Theme)
                                      │
                                      ▼
                      REST API Router (routes_research.py)
                                      │
             ┌────────────────────────┼────────────────────────┐
             ▼                        ▼                        ▼
  Experiment Registry       Ablation Workbench     Failure Case & Explainability
  (registry.py)             (ablation.py)           (explainability.py)
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      ▼
                     Research SQLite Database Tables
              (ResearchExperiment, ResearchAblation, FailureCase)
                                      │
                                      ▼
                     Research Artifact Manifest & Report
                 (reports/phase15/exp_{id}/manifest.json & report.md)
```

Pre-audit complete. Proceeding with Phase 15 implementation.
