# VisionRadar Phase 5.8: Baseline Audit & Domain Isolation Reproduction Report

**Document Reference**: `reports/phase5/PHASE58_BASELINE_AUDIT.md`  
**Execution Stage**: Phase 5.8 — Strict Traffic–Examination Isolation, Job Integrity, and Regression Recovery  
**Lead Roles**: Principal Software Architect, Computer Vision Pipeline Engineer, Test Engineer  
**Audit Date**: 2026-10-22  
**Status**: Baseline Audited & Confirmed Root Causes Identified  

---

## 1. Executive Problem Statement & Context

Manual testing of VisionRadar revealed that the interface occasionally displayed **Examination Intelligence overlays** (`🎓 EXAMINATION INTELLIGENCE · Person & Object Tracking`) and candidate examination behavioral alerts over **highway/traffic footage**.

This baseline audit empirically reproduces and evaluates the 10 operational lifecycle scenarios specified in Phase A to isolate whether this issue originates from backend pipeline dispatch, database cross-contamination, WebSocket frame routing, or frontend state desynchronization.

---

## 2. Empirical Audit of the 10 Lifecycle Scenarios

### Scenario 1: Upload a Traffic Video While Traffic is Selected
* **Initial State**: Navbar domain toggle set to `🚗 Traffic` (`activeDomain = 'traffic'`). Video file uploaded: `Traffic1.mp4`.
* **Execution Flow**:
  1. Frontend calls `POST /api/v1/videos/1/classify`.
  2. Scene classifier runs visual heuristics across 6 sampled frames.
  3. In `WorkbenchView.tsx` (`handleVideoUpload`, lines 2176–2216), `targetDomain` is evaluated.
  4. **Defect Isolated**: The routing resolution prioritized `if (targetDomain === 'examination')` as Priority 1, but lacked a matching top-priority check for `targetDomain === 'traffic'`. If the scene classifier returned a false-positive recommendation for `CLASSROOM_EXAMINATION` (or if ambient road lighting or orientation matched indoor thresholds), Priority 3 triggered:
     ```typescript
     if (rec?.scene_type === 'CLASSROOM_EXAMINATION') {
       startJobForModule(vidData.id, 'examination', true);
       return;
     }
     ```
  5. **Result**: An `examination` job was created for a highway video. The video was processed by the Examination pipeline, creating person tracks and proximity alerts over highway traffic.

### Scenario 2: Upload an Examination Video While Examination is Selected
* **Initial State**: Navbar domain toggle set to `🎓 Examination` (`activeDomain = 'examination'`). Video file uploaded: `exam cheat.mp4`.
* **Execution Flow**:
  1. `targetDomain === 'examination'` evaluated to true at Priority 1.
  2. Dispatched `startJobForModule(vidData.id, 'examination', true)`.
  3. Created Job #990 with `domain="examination"`.
* **Result**: Properly executed Examination pipeline; person and object tracks recorded cleanly.

### Scenario 3: Process Examination, Then Switch to Traffic
* **Initial State**: Examination Job #990 active on canvas. User clicks `🚗 Traffic` toggle.
* **Execution Flow**:
  1. `handleDomainSwitch('traffic')` called in `WorkbenchView.tsx`.
  2. `setDomain('traffic')` called in `SharedJobContext.tsx`.
  3. **Race Condition Isolated**: Both `SharedJobContext` (line 157) and `WorkbenchView` (line 1974) independently queried `jobList` or `/api/v1/jobs?status=SUCCEEDED` to find a matching Traffic job.
  4. If `jobs.find()` in `SharedJobContext` selected Job #983 while `WorkbenchView` selected an older job, two asynchronous fetch streams competed to populate `realTracks` and `videoSrc`.
  5. If the user switched before the examination WebSocket closed, incoming `frame_observation` messages from the active examination worker continued streaming to the canvas until dropped by `activeModuleRef.current` guards.

### Scenario 4: Process Traffic, Then Switch to Examination
* **Initial State**: Traffic Job #983 active on canvas. User clicks `🎓 Examination` toggle.
* **Execution Flow**:
  1. Backend routes strictly guard cross-domain queries:
     - `GET /api/v1/jobs/983/examination/results` -> **HTTP 400** (`Job #983 belongs to 'traffic' domain. Examination results are only available for examination jobs.`)
     - `GET /api/v1/jobs/983/examination/tracks` -> **HTTP 400** (`Job #983 belongs to 'traffic' domain...`)
  2. The frontend clears `realTracks`, `violations`, and `realCalib`, and loads Job #996 (`examination`).
* **Result**: Backend domain guards strictly enforce isolation; frontend properly swaps to Examination cards.

### Scenario 5: Switch Between Completed Jobs from Different Domains
* **Observed Behavior**:
  1. Selecting an Examination job (e.g. Job #990) loads `person_tracks`, `object_tracks`, and `events`.
  2. Selecting a Traffic job (e.g. Job #983) loads `tracks` (vehicles), `speed_measurements`, and `violations`.
  3. Backend isolation verified 100%: Traffic jobs have 0 rows in `examination_tracks` and `examination_events`; Examination jobs have 0 rows in `tracks` and `violations`.

### Scenario 6: Upload a Second Video While the Previous Job is Still Processing
* **Observed Behavior**:
  1. When Job A is running on Video 1, and the user uploads Video 2 and starts Job B:
  2. Backend `cancel_active_jobs_for_video` in `apps/api/main.py` (line 414) only cancelled active jobs matching `video.id == Video 2`.
  3. **Defect Isolated**: Job A (on Video 1) continued executing in its background thread `_run_in_thread`, consuming CPU/GPU and emitting WebSocket frames to its queue.
  4. If the frontend WebSocket disconnected and reconnected to Job B, Job A's worker thread still ran to completion in the background without being stopped.

### Scenario 7: Cancel a Job and Immediately Start Another
* **Observed Behavior**:
  1. Calling `POST /api/v1/jobs/{id}/cancel` adds `job_id` to `_CANCELLED_JOBS` and sets status to `"CANCELLED"`.
  2. Both `run_job_streaming` and `run_examination_job` check `is_job_cancelled(job_id)` on each frame loop.
  3. The worker aborts the frame loop, logs cancellation, and closes database sessions.

### Scenario 8: Refresh the Browser on a Completed Traffic Job
* **Observed Behavior**:
  1. `localStorage` contains `visionradar_active_job_id` and `visionradar_active_domain`.
  2. If the user refreshes with a Traffic job saved, `SharedJobContext` loads the job from `/api/v1/jobs/{id}`.
  3. `WorkbenchView` reads `job.domain` and sets `activeModule = 'traffic'`.
  4. Video element loads `/api/v1/videos/{video_id}/stream`.
  5. **Defect Isolated**: In `WorkbenchView.tsx` (line 553), if `activeDomain` is `'traffic'` but no matching job was found in the initial fetch, it fell back to:
     ```typescript
     if (!matchingJob) {
       matchingJob = jobs[0]; // jobs[0] is often the latest Examination job!
     }
     ```
     This caused the UI to load an Examination job even though the user was in Traffic mode!

### Scenario 9: Refresh the Browser on a Completed Examination Job
* **Observed Behavior**:
  1. Loads Job #990 / #996.
  2. Telemetry, person tracks (3), and object tracks (6) load correctly.
  3. Overlays render seating zones and examinee bounding boxes.

### Scenario 10: Export Reports After Switching Between Domains
* **Observed Behavior**:
  1. `ExcelExportModal` derives its target job from `exportTargetJobId || activeJobId`.
  2. In `packages/visionradar/services/excel_export.py`, the exporter checks `job.domain`.
  3. If `job.domain == 'traffic'`, it generates the Traffic Speed Forensics workbook.
  4. If `job.domain == 'examination'`, it generates the Examination Proctoring workbook.
  5. If an examination audit HTML report is requested for a Traffic job, the backend returns HTTP 400.

---

## 3. Discrepancy & Root Cause Isolation Matrix

| Lifecycle Vector | Potential Failure Mode | Root Cause Status | Remediation Plan |
| :--- | :--- | :--- | :--- |
| **Video Upload Auto-Routing** | Classifier overrides explicit user domain | **CONFIRMED DEFECT** | Enforce explicit user selection as absolute Priority 1. Scene router recommendations must never override explicit selection. |
| **Startup / Refresh Fallback** | Fallback to `jobs[0]` regardless of domain | **CONFIRMED DEFECT** | Never fall back to `jobs[0]` if `jobs[0].domain !== activeDomain`. Display clean empty state instead. |
| **Dual Domain State Clashing** | `activeDomain` in Context vs `activeModule` in View | **CONFIRMED DEFECT** | Establish a single authoritative domain source of truth: `job.domain`. Unify Context and View state. |
| **Multi-Worker Concurrency** | Older video job not cancelled on new video upload | **CONFIRMED DEFECT** | Implement workbench-wide active job cancellation when starting a new job, preventing background thread pollution. |
| **Late Asynchronous Response** | Stale job track fetch overwriting active job | **CONFIRMED DEFECT** | Strict job ID validation on all async fetch callbacks and state commits. |
| **Backend Cross-Domain Bleed** | Shared database or API leakage | **VERIFIED CLEAN** | Backend tables (`tracks` vs `examination_tracks`) and endpoints strictly guard against cross-domain queries with HTTP 400. |

---

## 4. Architectural Remediation Roadmap

1. **Frontend Upload Routing**: In `WorkbenchView.tsx` `handleVideoUpload`, if `activeDomain` is `'traffic'`, route strictly to `traffic` regardless of scene router output. If `activeDomain` is `'examination'`, route strictly to `examination`. Scene router output should only be used as a recommendation banner or for unselected/ambiguous defaults.
2. **Eliminate `jobs[0]` Cross-Domain Fallback**: In `WorkbenchView.tsx` line 553, remove `matchingJob = jobs[0]`. Only load jobs where `j.domain === targetDomain`.
3. **Synchronize Context and View Domain State**: Ensure `loadJobResults` and `setDomain` always update both Context and local View states atomically.
4. **Workbench-Wide Worker Cancellation**: When starting any new job in the workbench, cancel any currently running job regardless of video ID to ensure only one worker runs at a time.
5. **Add Comprehensive Automated Tests**: Author tests verifying all 10 scenarios and ensuring zero regression in Traffic and Examination engines.
