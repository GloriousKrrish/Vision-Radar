# VisionRadar Phase 5.2 — Domain Routing & Isolation Test Results

**Date:** October 21, 2026  
**Test Lead:** Senior Computer Vision Pipeline & Backend Reliability Engineer  
**Status:** ALL 25 TEST CASES PASSED (100% SUCCESS RATE)  
**Frontend Build:** SUCCESSFUL (0 ERRORS, 0 WARNINGS)  

---

### 1. Test Execution Summary

A full empirical validation of the VisionRadar platform was conducted to verify domain isolation, pipeline routing, controller guards, cross-workbench synchronization, and enterprise Excel export integrity.

```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\admin\Downloads\VisionRadar - Claude(i)
configfile: pyproject.toml
plugins: anyio-4.12.1, langsmith-0.8.16, asyncio-1.4.0
collected 25 items

tests/test_domain_isolation.py::test_missing_or_invalid_domain_rejected_with_400 PASSED [  4%]
tests/test_domain_isolation.py::test_create_examination_job_persists_domain_and_omits_calibration PASSED [  8%]
tests/test_domain_isolation.py::test_create_traffic_job_persists_traffic_domain PASSED [ 12%]
tests/test_domain_isolation.py::test_domain_guard_blocks_cross_domain_track_and_speed_requests PASSED [ 16%]
tests/test_domain_isolation.py::test_domain_guard_blocks_cross_domain_examination_requests PASSED [ 20%]
tests/test_domain_isolation.py::test_excel_export_preview_derives_domain_from_persisted_job PASSED [ 24%]
tests/test_domain_isolation.py::test_excel_export_download_generates_domain_isolated_workbooks PASSED [ 28%]
tests/test_domain_isolation.py::test_job_switching_and_isolation PASSED  [ 32%]
tests/test_domain_isolation.py::test_backward_compatibility_job_entity_defaults PASSED [ 36%]
tests/test_speed.py::test_speed_estimation_on_linear_trajectory PASSED   [ 40%]
tests/test_speed.py::test_deterministic_speed_sanity_cases PASSED        [ 44%]
tests/test_examination_accuracy_audit.py::test_multiple_students_and_calibrated_threshold PASSED [ 48%]
tests/test_examination_accuracy_audit.py::test_brief_occlusion_interpolation_and_continuity PASSED [ 52%]
tests/test_examination_accuracy_audit.py::test_staff_exclusion_from_student_alerts PASSED [ 56%]
tests/test_examination_accuracy_audit.py::test_unknown_role_handling PASSED [ 60%]
tests/test_examination_accuracy_audit.py::test_calculator_discrimination_uncertainty PASSED [ 64%]
tests/test_examination_accuracy_audit.py::test_duplicate_event_suppression PASSED [ 68%]
tests/test_examination_accuracy_audit.py::test_forensic_audit_report_generation PASSED [ 72%]
tests/test_examination_accuracy_audit.py::test_traffic_pipeline_untouched_and_functional PASSED [ 76%]
tests/test_excel_export_and_sync.py::test_excel_export_traffic_job PASSED [ 80%]
tests/test_excel_export_and_sync.py::test_excel_export_examination_job PASSED [ 84%]
tests/test_excel_export_and_sync.py::test_excel_export_modes_final_vs_snapshot PASSED [ 88%]
tests/test_excel_export_and_sync.py::test_excel_export_empty_dataset_handling PASSED [ 92%]
tests/test_excel_export_and_sync.py::test_api_excel_export_preview_endpoint PASSED [ 96%]
tests/test_excel_export_and_sync.py::test_api_excel_export_download_endpoint PASSED [100%]

============================== 25 passed in 135.20s ===============================
```

---

### 2. Detailed Test Case Analysis

#### Suite A: Phase 5.2 Strict Domain Isolation (`tests/test_domain_isolation.py`)

| Test ID | Test Case | Target Requirement | Status | Empirical Evidence |
| :--- | :--- | :--- | :--- | :--- |
| `ISO-01` | `test_missing_or_invalid_domain_rejected_with_400` | Rejection of missing/invalid domains | **PASSED** | POST `/api/v1/videos/1/jobs` with missing domain returns `HTTP 400`: *"Invalid or missing domain"*. POST with domain="medical" returns `HTTP 400`: *"Explicit domain ('traffic' or 'examination') is strictly required."* |
| `ISO-02` | `test_create_examination_job_persists_domain_and_omits_calibration` | Examination persistence & calibration stripping | **PASSED** | Job created with `domain="examination"` persists `job.domain == "examination"` in database and forces `calibration_id = None`. |
| `ISO-03` | `test_create_traffic_job_persists_traffic_domain` | Traffic job creation & persistence | **PASSED** | Job created with `domain="traffic"` persists `job.domain == "traffic"`. |
| `ISO-04` | `test_domain_guard_blocks_cross_domain_track_and_speed_requests` | Controller guards against traffic endpoints on exam jobs | **PASSED** | GET `/api/v1/jobs/{exam_id}/tracks` returns `HTTP 400`: *"Job belongs to 'examination' domain. Vehicle tracks are only available for traffic jobs."* |
| `ISO-05` | `test_domain_guard_blocks_cross_domain_examination_requests` | Controller guards against exam endpoints on traffic jobs | **PASSED** | GET `/api/v1/jobs/{traffic_id}/examination/results` returns `HTTP 400`: *"Job belongs to 'traffic' domain. Examination results are only available for examination jobs."* |
| `ISO-06` | `test_excel_export_preview_derives_domain_from_persisted_job` | Authoritative domain in export preview | **PASSED** | Query parameter `?domain=traffic` passed on an examination job is ignored; response returns `domain="examination"` and sheets `['Monitored Persons & Tracks', 'Examination Events']`. |
| `ISO-07` | `test_excel_export_download_generates_domain_isolated_workbooks` | Worksheet isolation in generated binary workbooks | **PASSED** | Examination workbook contains `Monitored Persons & Tracks` and `Examination Events`, omitting `Speed Analytics` and `Unique Tracks`. Traffic workbook contains `Unique Tracks` and `Speed Analytics`, omitting examination sheets. |
| `ISO-08` | `test_job_switching_and_isolation` | Job switching domain integrity | **PASSED** | Every listed job exposes authoritative `domain` property matching its persisted database value. |
| `ISO-09` | `test_backward_compatibility_job_entity_defaults` | Legacy job backward compatibility | **PASSED** | Default column value for legacy jobs flushes cleanly as `"traffic"`. |

---

### 3. Frontend Production Build Verification

The React/Vite frontend was compiled for production:

```bash
cd apps/web && npm run build
```

**Captured Terminal Output:**
```
> visionradar-web@0.1.0 build
> tsc && vite build

vite v5.4.21 building for production...
transforming...
✓ 41 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.78 kB │ gzip:  0.45 kB
dist/assets/index-Cl-RzSZe.css   10.48 kB │ gzip:  3.10 kB
dist/assets/index-DXex4hY2.js   326.43 kB │ gzip: 94.17 kB
✓ built in 6.18s
```
- **TypeScript Compilation:** Zero errors.
- **Vite Bundle Generation:** Zero chunk errors.

---

### 4. Interactive Browser Verification

Using browser automation on `http://localhost:8080`:
1. **Pre-Upload Target Selection Verified:**
   - The top action bar displays `Upload Target: 🚗 Traffic / 🎓 Examination`.
   - Selecting `🎓 Examination` immediately establishes the active target module.
2. **Domain Switching Synchronization Verified:**
   - Clicking between `🚗 Traffic` and `🎓 Examination` successfully purges frame buffers (`rtFrameBufferRef`), resets active track arrays, and selects the most recent job matching that specific domain.
3. **Canvas Isolation Verified:**
   - Under Examination mode, vehicle tags (`CAR`, `TRUCK`) and velocity tags (`[UNCALIB]`) are completely absent from the canvas renderer.
   - Traffic calibration buttons ("Homography Calibration") are cleanly hidden in Examination mode to prevent accidental road-grid drawing over classroom desks.

---

### 5. Regression Matrix

| Subsystem | Baseline Status | Post-Fix Status | Regression Detected |
| :--- | :--- | :--- | :--- |
| **Traffic Speed Pipeline** | 2/2 Passed | 2/2 Passed | None |
| **Traffic Vehicle Tracking** | Functional | Functional | None |
| **Examination Audit & Accuracy** | 8/8 Passed | 8/8 Passed | None |
| **Excel Intelligence Export** | 6/6 Passed | 6/6 Passed | None |
| **FastAPI Backend Endpoints** | Online | Online (`/api/v1/health` 200 OK) | None |
| **Frontend Web App** | Online (port 8080) | Online (port 8080) | None |
