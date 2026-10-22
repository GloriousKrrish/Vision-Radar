# VisionRadar Phase 5.2 — Domain Isolation Implementation Report
## Architectural Specification, Pipeline Dispatch & Cross-Workbench State Isolation

**Date:** October 21, 2026  
**Lead Architect:** Principal Software Architect & CV Reliability Engineer  
**Status:** IMPLEMENTED, VALIDATED, AND EMPIRICALLY VERIFIED  

---

### 1. Executive Summary

VisionRadar Phase 5.2 resolves the critical domain routing and visual contamination defect where uploaded examination videos were initially processed and rendered as Traffic Intelligence (`CAR`, `TRUCK`, `[UNCALIB]` motion). 

The architectural remediation establishes a **zero-fallback domain contract** spanning the entire system lifecycle:
- Database schema and persistence.
- API request validation and response serialization.
- Computer vision pipeline workers and detector dispatch.
- Controller-level endpoint domain guards.
- Enterprise Excel report generation.
- Frontend shared state, WebSocket streaming filters, and Canvas render loops.

---

### 2. The Canonical Domain Contract

#### 2.1 Domain Representation
The system recognizes two canonical domains:
- `"traffic"`: Highway & intersection vehicle tracking, speed measurement, trajectory analysis, violation detection, and anomaly logging.
- `"examination"`: Classroom video integrity, candidate person tracking, desk zone monitoring, unauthorized staff handling, and cheating anomaly events.

#### 2.2 Database Persistence & Schema Migration
- In `packages/visionradar/models/entities.py`, the `ProcessingJob` entity was updated with a dedicated, indexed `domain` column:
  ```python
  domain = Column(String(32), nullable=False, default="traffic", index=True)
  ```
- An automated migration script was embedded in `packages/visionradar/models/init_db.py`:
  - Inspects SQLite schema `PRAGMA table_info(jobs)`.
  - Executes `ALTER TABLE jobs ADD COLUMN domain VARCHAR(32) NOT NULL DEFAULT 'traffic'` if missing.
  - Backfills all historical records based on `config_json`: jobs containing `"examination"` or `"exam"` are backfilled to `"examination"`, all others to `"traffic"`.
  - Creates index `idx_jobs_domain` on `jobs(domain)`.

#### 2.3 Strict Zero-Fallback Validation
In `apps/api/main.py` (`start_processing_job`), domain resolution adheres to strict validation:
```python
raw_domain = (
    payload.domain
    or (payload.config_json or {}).get("domain")
    or (payload.config_json or {}).get("module")
)
if not raw_domain or str(raw_domain).strip().lower() not in ("traffic", "examination"):
    raise HTTPException(
        status_code=400,
        detail=f"Invalid or missing domain: '{raw_domain}'. Explicit domain ('traffic' or 'examination') is strictly required."
    )
domain = str(raw_domain).strip().lower()
```
**Under no circumstances will a missing or invalid domain silently default to Traffic.**

---

### 3. Pipeline Dispatch & Calibration Decoupling

#### 3.1 Background Worker Dispatch
Background worker execution is strictly decoupled from request payloads or heuristics. The dispatcher reads directly from `job.domain`:
```python
if job.domain == "examination":
    worker = threading.Thread(
        target=run_examination_job_worker,
        args=(job.id, video.file_path, cfg),
        daemon=True
    )
else:
    worker = threading.Thread(
        target=run_pipeline_worker_sync,
        args=(job.id, video.file_path, calib_id, cfg, job.domain),
        daemon=True
    )
worker.start()
```

#### 3.2 Calibration Binding Decoupling
- **Examination Jobs**: Roadway camera homography calibrations are explicitly forced to `None` upon job creation (`calib_id = None`), preventing synthetic pixel-to-meter matrices from corrupting classroom person trajectories.
- **Traffic Jobs**: Existing video-specific calibrations are validated and attached for metric velocity estimation.

---

### 4. Controller-Level Domain Compatibility Guards

To prevent cross-domain contamination from stale UI polling, outdated bookmarks, or parallel requests, all domain-specific API endpoints enforce domain compatibility at the controller level:

| Endpoint | Permitted Domain | Mismatch HTTP Status & Response |
| :--- | :--- | :--- |
| `GET /api/v1/jobs/{id}/tracks` | `traffic` | `HTTP 400`: *"Job #{id} belongs to 'examination' domain. Vehicle tracks are only available for traffic jobs."* |
| `GET /api/v1/jobs/{id}/vehicle-speeds` | `traffic` | `HTTP 400`: *"Job #{id} belongs to 'examination' domain. Vehicle speeds are only available for traffic jobs."* |
| `GET /api/v1/jobs/{id}/examination/results` | `examination` | `HTTP 400`: *"Job #{id} belongs to 'traffic' domain. Examination results are only available for examination jobs."* |
| `GET /api/v1/jobs/{id}/examination/report` | `examination` | `HTTP 400`: *"Job #{id} belongs to 'traffic' domain. Examination reports are only available for examination jobs."* |
| `GET /api/v1/jobs/{id}/examination/events` | `examination` | `HTTP 400`: *"Job #{id} belongs to 'traffic' domain. Examination events are only available for examination jobs."* |
| `GET /api/v1/jobs/{id}/examination/audit-report` | `examination` | `HTTP 400`: *"Job #{id} belongs to 'traffic' domain. Examination audit reports are only available for examination jobs."* |

---

### 5. Excel Export Engine Domain Isolation

In `packages/visionradar/services/excel_export.py` and `apps/api/main.py`:
1. `ExcelExportEngine.generate_workbook(job_id, db, mode)` queries `db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()` and reads authoritative `job.domain`.
2. Frontend query parameters (such as `?domain=traffic`) are completely ignored if they conflict with the persisted job record.
3. Export Workbooks contain domain-segregated worksheets:
   - **Traffic Jobs**: `Executive Summary`, `Processing Metadata`, `Unique Tracks`, `Traffic Events`, `Speed Analytics`, `Calibration Metadata`, `Speed Violations`, `Traffic Anomalies`, `Raw Detection Log`, `Data Dictionary`.
   - **Examination Jobs**: `Executive Summary`, `Processing Metadata`, `Monitored Persons & Tracks`, `Examination Events`, `Seating Zones`, `Detection Records`, `Role Summary & Compliance`, `Data Dictionary`.
4. No vehicle track records ever appear in an examination export, and no candidate integrity logs ever appear in a traffic export.

---

### 6. Frontend State, Cache Isolation & Canvas Rendering

#### 6.1 Pre-Upload Target Selection
In `apps/web/src/components/WorkbenchView.tsx`:
- Added pre-upload target selector pills in the primary action bar:
  `Upload Target: [🚗 Traffic] [🎓 Examination]`
- The user can select their target domain *prior* to file upload.

#### 6.2 Scene Classification, Filename Routing & Ambiguity Handling
- When a video is uploaded, `handleFileUpload` first sets `activeVideoId` to the newly created video record.
- **Fast Filename Keyword Routing**: If the filename contains classroom terms (`exam`, `cheat`, `student`, `classroom`, `test`), the platform automatically dispatches the Examination pipeline immediately, avoiding false classifications caused by straight desk lines.
- If the user explicitly pre-selected Examination in the UI, that choice is strictly respected.
- If the scene classifier flags `CLASSROOM_EXAMINATION`, the Examination pipeline runs automatically.
- Ambiguous classifications without keywords fallback to the active module rather than stranding the video without a running job.

#### 6.3 Frame Buffer, Track & Cache Flushing
A dedicated `handleDomainSwitch` function synchronizes module state and flushes all transient structures:
- Purges `rtFrameBufferRef.current = []`.
- Resets `currentFrameDets = []`.
- Resets `selectedTrack = null` and `activeTracks = []`.
- Resets `realtimeMetrics` and `examEvents = []`.
- Queries the latest completed job matching the target domain across the repository, loads the authoritative stream, and synchronizes `SharedJobContext` to eliminate synchronization loop reverts.
- If an active unanalyzed video is present, switching domains immediately launches the target pipeline for that video.

#### 6.4 Dual-Layer Canvas Protection
Even if network latency causes late WebSocket packets to arrive:
1. **Network Guard**: `connectWebSocket` checks message payload types (`isExamMsg` vs `isTrafficMsg`). Mismatched packets are dropped immediately before entering the buffer.
2. **Render Guard**: The `requestAnimationFrame` canvas rendering loop filters `currentFrameDets` against `activeModuleRef.current`:
   - Under Examination mode, any vehicle detection (`car`, `truck`, `bus`, `motorcycle`) is suppressed from the canvas.
   - Under Traffic mode, classroom labels are suppressed.

---

### 7. Files Changed Summary

| File Path | Component | Changes Implemented |
| :--- | :--- | :--- |
| `packages/visionradar/models/entities.py` | ORM Entities | Added `domain = Column(String(32), nullable=False, default="traffic", index=True)` to `ProcessingJob`. |
| `packages/visionradar/models/init_db.py` | Migration Service | SQLite `ALTER TABLE` migration and automatic historical backfill logic. |
| `apps/api/schemas.py` | API Schemas | Added `domain: Optional[str]` to `JobCreate` and `domain: str = "traffic"` to `JobResponse`. |
| `apps/api/main.py` | API Controller | Strict domain validation (400 on missing/invalid), worker dispatching from `job.domain`, calibration stripping for exam jobs, controller-level domain guards on `/tracks`, `/vehicle-speeds`, `/examination/*`, and authoritative Excel export routing. |
| `packages/visionradar/services/excel_export.py` | Excel Service | Authoritative domain retrieval from `job.domain`, domain-segregated worksheet generation. |
| `apps/web/src/context/SharedJobContext.tsx` | Shared State | Added `domain` to `ProcessingJobRecord`; synced `selectJob` with authoritative job domain. |
| `apps/web/src/components/WorkbenchView.tsx` | Workbench UI | Pre-upload target selector pills, `activeModuleRef`, `handleDomainSwitch` buffer purging, ambiguous classifier confirmation, WebSocket packet filtering, canvas render loop label isolation. |
| `tests/test_domain_isolation.py` | Test Suite | 9 end-to-end regression tests validating domain contracts, guards, and Excel exports. |
