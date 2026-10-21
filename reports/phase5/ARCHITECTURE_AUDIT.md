# VisionRadar Phase 5: Architecture & Repository Audit

**Author**: Principal Software Architect & Data Engineering Lead  
**Date**: October 21, 2026  
**Status**: AUDIT COMPLETE  
**Deliverable**: `reports/phase5/ARCHITECTURE_AUDIT.md`  

---

## 1. Executive Overview

This audit comprehensively maps the existing architecture of the **VisionRadar — Universal Video Intelligence Platform** across its backend services, database entities, asynchronous job workers, real-time WebSocket messaging layer, and frontend workbenches.

The goal of Phase 5 is to establish:
1. **A unified shared job context** synchronizing persistent jobs and analysis metadata across all workbenches (`Workbench`, `Research Workbench`, `Calibration`, `Speed Benchmark`, `Experiments`, and `Architecture & APIs`).
2. **An enterprise-grade Excel intelligence export engine (`.xlsx`)** delivering multi-sheet workbooks with complete data provenance for both **Traffic Intelligence** and **Examination Intelligence** domains.

---

## 2. Component Organization & Architecture Mapping

```mermaid
graph TD
    Client[React SPA Frontend: Vite + TypeScript]
    API[FastAPI Gateway: port 8000]
    DB[(SQLite WAL Database: visionradar.db)]
    Workers[Async Background Worker Pool]
    WS[WebSocket Streaming Bus: /api/v1/jobs/{id}/stream]
    ExcelEngine[Excel Export Engine: openpyxl]
    PDFEngine[PDF Generator: ReportLab]

    Client -->|HTTP REST APIs| API
    Client -->|WebSocket Envelopes| WS
    API -->|ORM Transactions| DB
    API -->|Dispatch Jobs| Workers
    Workers -->|State & Progress Updates| DB
    Workers -->|Frame Envelopes| WS
    API -->|Generate Workbooks| ExcelEngine
    API -->|Generate PDF Reports| PDFEngine
    ExcelEngine -->|Query Job, Tracks, Events, Speed, Calibration| DB
```

---

## 3. Tracing a Real Job Lifecycle

### 3.1 Job Ingestion & Creation
1. **Video Registration**: An MP4/MKV video is uploaded via `POST /api/v1/videos/upload` or selected from disk. A record is created in `videos` with duration, resolution, FPS, and SHA-256 hash.
2. **Scene Routing (Optional Layer B)**: `POST /api/v1/scene/classify` analyzes initial video frames and recommends either the `traffic` or `examination` pipeline.
3. **Job Creation**: `POST /api/v1/videos/{video_id}/jobs` is called with:
   - `config_json`: module selection (`traffic` vs `examination`), speed limit parameters, processing mode (`balanced`, `high_speed`, `forensic`), stride, seating zones, or roster roles.
   - `calibration_id`: reference to persisted `calibrations` record (for traffic road-plane projection).
4. **State Transition**: `ProcessingJob` record is initialized with:
   - `status = "QUEUED"`
   - `stage = "Decoder"`
   - `progress_pct = 0.0`

### 3.2 Job Execution
- **Traffic Pipeline (`run_job`)**:
  - Background thread runs `run_job_streaming(job_id)`.
  - Spawns YOLOX-Nano inference + vehicle ByteTrack + homography speed estimation + violation detection.
  - Emits real-time `CommonEnvelope` frames over WebSocket queue.
  - Finalizes by persisting `Track`, `SpeedMeasurement`, and `Violation` entities into `visionradar.db`.
- **Examination Pipeline (`run_examination_job`)**:
  - Background thread runs `run_examination_job(job_id)`.
  - Spawns person detection (calibrated 0.20 threshold) + PersonTracker (occlusion interpolation) + object tracker + seating zone manager + behavioral event engine.
  - Emits real-time `CommonEnvelope` frames.
  - Finalizes by persisting `ExaminationTrackRecord` (with `assigned_role` and `role_source`), `ExaminationEventRecord`, and `telemetry_json`.

### 3.3 Result Retrieval
- Standardized Layer D Common Envelope: `GET /api/v1/jobs/{job_id}/results` delivers a domain-agnostic summary wrapper.
- Domain-Specific Endpoints:
  - Traffic: `GET /api/v1/jobs/{job_id}/tracks`, `GET /api/v1/violations?job_id={job_id}`
  - Examination: `GET /api/v1/jobs/{job_id}/examination/results`, `GET /api/v1/jobs/{job_id}/examination/events`

---

## 4. Current State: Shared vs. Independent Components

| System Dimension | Shared Across Workbenches | Independent / Fragmented |
| :--- | :--- | :--- |
| **Persistent Job ID** | Yes (`ProcessingJob.id` in DB). | No shared React Context in frontend; each tab (`WorkbenchView`, `ResearchWorkbenchView`) re-implements job selection independently. |
| **Database Storage** | Yes (`data/visionradar.db` SQLite with WAL). | Some views (`BenchmarkView`, `ArchitectureView`) rely on static fallbacks when specific endpoints are empty. |
| **Calibration Persistence** | Yes (`calibrations` table with versioning). | `CalibrationView` operates on pixel/world coordinate scratchpad rather than binding directly to active video jobs. |
| **Telemetry & Metrics** | Yes (`telemetry_json` on `processing_jobs`). | Export capabilities are fragmented: Traffic has PDF export, Examination has JSON/HTML export, neither has comprehensive Excel (`.xlsx`). |
| **Domain Segregation** | Strict logical separation in worker pipelines (`traffic` vs `examination`). | `WorkbenchView.tsx` canvas handles both domains, but filters (e.g. vehicle class filter) previously leaked into examination rendering. |

---

## 5. Architectural Requirements for Phase 5

### 5.1 Shared Job Context (Phase B)
A unified state provider must be introduced to:
- Bind the currently selected `activeJobId` across all tabs.
- When a user selects a job in the main Workbench, that job context must be accessible in `Research Workbench`, `Calibration`, and `Speed Benchmark` without full page reloads.
- Provide data availability indicators (`has_tracks`, `has_violations`, `has_examination_events`, `has_speed_analytics`, `has_calibration`).
- Prevent cross-job data leakage by clearing stale memory buffers upon job switching.

### 5.2 Professional Excel Export Engine (Phases C, D, E)
A modular Python service (`visionradar.services.excel_export.ExcelExportEngine`) using `openpyxl`:
- Must support **Live Snapshot** (in-flight partial data while video is processing) and **Final Export** (all finalized database records).
- Generates valid `.xlsx` workbooks with professional corporate styling (Navy/Indigo header theme, auto-filters, frozen panes, zebra striping, currency/numeric formatting).
- Dynamically selects domain-specific worksheets:
  1. `Executive Summary` (metadata, KPIs, calibration status)
  2. `Processing Metadata` (hardware telemetry, FPS, stride, timestamps)
  3. `Detection Records` (per-frame detections, confidence, bounding boxes)
  4. `Unique Tracks` (track IDs, duration, start/end frames, assigned zones/roles)
  5. `Traffic Events` (violations, vehicle class, speed over limit, review status)
  6. `Speed Analytics` (instantaneous/smoothed speed, uncertainty intervals)
  7. `Calibration Metadata` (homography matrix, image/world points, RMSE, camera pitch/height)
  8. `Examination Events` (candidate behavioral anomalies, priority, evidence strength, supporting frames)
  9. `Research & Experiments` (experimental ablation diffs, benchmark metrics)
  10. `Data Dictionary` (complete semantic specification of columns and units)
  11. `Export Metadata` (provenance, generation timestamp, cryptographic hash)

---

## 6. Audit Conclusion & Next Steps

The existing repository architecture is clean, modular, and robust. We do **not** need to rewrite any core pipeline logic or database registries. We will:
1. Extend `packages/visionradar/services/excel_export.py` with an enterprise-grade `openpyxl` workbook generator.
2. Expose `GET /api/v1/jobs/{job_id}/export/excel` with mode selector (`final` vs `snapshot`).
3. Implement a shared job context on the frontend (`SharedJobContext.tsx`) and connect the Excel export modal to the workbenches.
4. Validate with automated unit tests and workbook inspection.
