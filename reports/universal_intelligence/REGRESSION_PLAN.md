# VisionRadar — Universal Video Intelligence: Regression Plan (Stage 0)

**Date**: 2026-10-19  
**Status**: ACTIVE PROTOCOL  
**Enforcement**: Strict Zero-Regression  

---

## 1. Regression Testing Principles

Under the **Strict Zero-Regression** mandate, any defect or behavioral change introduced into existing traffic functionality is considered a release blocker.

Every stage of implementation must satisfy the **Four Regression Gates**:
1. **Pytest Gate**: All 244 existing automated tests must pass without regressions.
2. **Build Gate**: Frontend production build (`tsc && vite build`) must compile cleanly with 0 errors.
3. **Model Invariance Gate**: `data/models/yolox_nano.onnx` SHA256 must strictly remain `C789161ED43C8269FCD4E67C67EEEB4E80C622DA2EB296A20BC6007BD18A0B7D`.
4. **Data Isolation Gate**: Existing database tables (`videos`, `calibrations`, `processing_jobs`, `tracks`, `speed_measurements`, `violations`, `evidence`) must remain intact without destructive mutations.

---

## 2. Protected Traffic Pipeline Invariants

The following components and behaviors are **strictly locked**:

| Component | Invariant Property | Verification Method |
| :--- | :--- | :--- |
| **YOLOX-Nano ONNX** | Hash `C789...0B7D`, 416x416 RGB input, vehicle class mapping | `test_production_model_existence`, `test_coco_class_mapping` |
| **ByteTrack** | Two-stage association, 30-frame track buffer, IoU match threshold 0.8/0.5 | `test_track_id_stability`, `test_duplicate_detection_assignment_suppression` |
| **Camera Calibration** | 4-point image-to-world planar homography, camera pitch & height | `test_calibration.py`, `test_homography.py` |
| **Speed Estimation** | Kalman smoothing, physical plausibility clamps, latency compensation | `test_speed.py`, `test_speed_phase7.py` |
| **Speed Semantics** | Exactly 3 states: `CALIBRATED_SPEED`, `UNCALIBRATED_MOTION`, `SPEED_UNAVAILABLE` | `test_speed_provenance_schema`, `test_valid_speed_has_correct_status` |
| **WebSocket Streaming** | Bounded queue, oldest viz frame dropping under backpressure, terminal state guarantees | `test_phase3_4_realtime.py` |
| **Violation Evidence** | Crop + full-frame snapshot generation, legal provenance metadata | `test_evidence_provenance.py` |

---

## 3. Dedicated Test Suites to Add

To ensure comprehensive empirical coverage for new capabilities without altering existing test files, the following isolated test suites will be added:

### A. `tests/test_scene_router.py`
- Test 1: Traffic video classification with high confidence (`TRAFFIC`, recommended: `traffic`).
- Test 2: Classroom video classification (`CLASSROOM_EXAMINATION`, recommended: `examination`).
- Test 3: Unknown / non-target scene classification (`OTHER_UNKNOWN`).
- Test 4: Low-contrast or ambiguous video yields `AMBIGUOUS` with `requires_user_confirmation: True`.
- Test 5: Corrupt, unreadable, or 0-byte video handling without unhandled exceptions.
- Test 6: Fast sampling efficiency (decodes at most 8 frames, sub-500ms execution).
- Test 7: Manual user override persistence.

### B. `tests/test_examination_module.py`
- Test 1: Person detection bounding box extraction on examination frames.
- Test 2: Examination object detection (cell phones, notebooks/paper, laptops).
- Test 3: Independent person tracking with persistent track ID generation.
- Test 4: Seating zone polygon definition and person-to-seat assignment with uncertainty.
- Test 5: Observable event state machine: `POSSIBLE_PHONE_INTERACTION` temporal persistence.
- Test 6: Observable event state machine: `RESTRICTED_ZONE_ENTRY` polygon boundary crossing.
- Test 7: Observable event state machine: `LEAVING_SEATING_ZONE` detection.
- Test 8: Observable event neutrality: zero punitive, defamatory, or speculative labels.
- Test 9: Human-review status: all events initialized to `review_status: PENDING`.

### C. `tests/test_universal_pipeline_integration.py`
- Test 1: End-to-end routing: Upload -> Scene Router -> Traffic Pipeline for roadway video.
- Test 2: End-to-end routing: Upload -> Scene Router -> Examination Pipeline for classroom video.
- Test 3: Module isolation: Examination job execution leaves `tracks` and `speed_measurements` unchanged.
- Test 4: Module isolation: Traffic job execution leaves `examination_*` tables unchanged.
- Test 5: Concurrency safety: Worker queue prevents concurrent inference starvation on CPU.
- Test 6: Job cancellation: Cancelling an examination job terminates worker cleanly and frees resources.
- Test 7: API contract: Common results endpoint `/api/v1/jobs/{job_id}/results` delivers versioned payload.

---

## 4. Stage-by-Stage Verification Matrix

| Stage | Focus Area | Mandatory Regression Check |
| :--- | :--- | :--- |
| **Stage 1** | Module interfaces & data contracts | Run `pytest tests/test_models.py` + verify DB schema stability |
| **Stage 2** | Scene Router implementation | Run `pytest tests/test_scene_router.py` + all 244 baseline tests |
| **Stage 3** | Examination module skeleton & APIs | Run `pytest tests/test_api_endpoints.py` + verify endpoint isolation |
| **Stage 4** | Person & object detection | Verify CPU inference latency does not degrade background tasks |
| **Stage 5** | Observable event analysis | Run `pytest tests/test_examination_module.py` |
| **Stage 6** | Frontend Workbench integration | Run `npm run build` in `apps/web` + verify traffic controls |
| **Stage 7** | Storage & lifecycle safeguards | Run `pytest tests/test_universal_pipeline_integration.py` |
| **Stage 8** | Final regression & audit reports | Run full test suite (baseline 244 + new tests) + production build |

---

## 5. Rollback & Emergency Recovery Protocol

If any test in the 244 baseline suite fails or if the frontend build breaks during any stage:
1. **Halt Progression**: Immediately stop code modifications in that stage.
2. **Isolate Root Cause**: Inspect the Git diff of files touched in the failing stage.
3. **Revert Conflicting Edits**: Use `git checkout <file>` to revert modifications to shared files.
4. **Re-verify Baseline**: Run `pytest` to confirm return to the 244 passing baseline.
5. **Redesign Integration**: Re-implement using stricter file separation (e.g., dedicated route file or isolated helper) before proceeding.
