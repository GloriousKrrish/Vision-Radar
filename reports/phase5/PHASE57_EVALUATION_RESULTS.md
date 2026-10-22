# VisionRadar Phase 5.7: Empirical Evaluation Results & Metric Parity

**Document Reference**: `reports/phase5/PHASE57_EVALUATION_RESULTS.md`  
**Evaluation Scope**: Quantitative and Qualitative Empirical Audit of Examination Intelligence Pipeline  
**Target Sequence**: Job #990 (`exam cheat.mp4`, Video ID 552)  
**Lead Evaluator**: Principal Computer Vision Engineer, Data Integrity Engineer  
**Date**: 2026-10-22  

---

## 1. Evaluation Dataset & Environmental Limitations

| Parameter | Observed Characteristic | Methodological & Forensic Impact |
| :--- | :--- | :--- |
| **Video Identifier** | Video ID `552` (Job `#990`) | Persistent database record |
| **Filename** | `exam cheat.mp4` | Standard examination benchmark sequence |
| **Duration** | `14.967 seconds` (~15.0s) | Short observation window (449 frames @ 30 FPS) |
| **Spatial Resolution**| `480 x 854` pixels | Vertical portrait aspect ratio; limits distant facial feature resolution |
| **Perspective Angle** | Elevated side-frontal classroom angle | Foreground student is clear; background/peripheral students partially occluded |
| **Lighting & Contrast**| Standard indoor fluorescent | Minor shadow casting on desk surface; adequate SNR for person and phone detection |
| **Ground Truth Examinees** | 2 physically seated people + 1 peripheral boundary edge | Track #1 is present 100% of time; Track #2 enters at t=11.4s; Track #3 is a 2-frame edge detection at t=14.9s |

---

## 2. Before vs. After Metric Parity Audit

The table below contrasts the observed metric values across all tiers before and after the Phase 5.7 remediation:

| Metric Surface | Pre-Fix Value (Before) | Post-Fix Value (After) | Forensic Impact / Improvement |
| :--- | :---: | :---: | :--- |
| **Database `examination_tracks`** | 9 rows total | 9 rows total (3 person, 6 object) | Database preserved 100%; no destructive deletions |
| **Frontend `MONITORED PERSONS`** | **`9`** | **`3`** *(2 seated · 1 edge)* | **Fixed**: Object tracks are no longer counted as humans |
| **Frontend `ROSTER ROLES: Unknown`** | **`9`** | **`3`** | **Fixed**: 6 physical objects removed from examinee roster |
| **Frontend `ROSTER ROLES: Objects`** | *Unlisted* | **`6 Associated Objects`** | **Added**: Physical evidence materials explicitly listed |
| **API `GET .../examination/tracks`** | **`404 Not Found`** | **`200 OK`** (Partitioned JSON) | **Fixed**: Dedicated track inspection endpoint active |
| **API `GET .../examination/results`** | `tracks` = 9 mixed entities | `tracks` = 3 persons, `objects` = 6 | **Fixed**: Clear entity partition with server-side summary |
| **HTML Audit Report Seal** | Valid (1 card missing objects) | Valid (`VR-SEAL-C35B49ACA37D8C63...`) | **Enhanced**: Added Object KPI, Seated vs Edge, Data Dictionary |
| **Excel Export Executive Summary** | 3 persons, 6 objects | 3 persons, 2 sustained, 1 edge, 6 objects | **Enhanced**: Sub-breakdown explains 2 seated examinees + 1 edge detection |
| **Candidate Events Count** | 4 | 4 | Reconciled across all 5 tiers |
| **Review States Sum** | 2 Accepted + 2 Pending = 4 | 2 Accepted + 2 Pending = 4 | Exact 100% mathematical reconciliation |
| **Proctor Role Editing UI** | *Missing* | **Present in Sidebar Inspector** | Proctors can assign Student/Staff directly from UI |

---

## 3. Computer Vision & Tracking Reliability Evaluation

### 3.1 Tracking Continuity Analysis

| Track ID | Classification | Span (Frames) | Duration (s) | Continuity % | Occlusion & Trajectory Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | Seated Examinee | 0 – 448 | 14.97s | **100.0%** (449/449) | Full uninterrupted visual presence; zero ID switches; stable bounding box |
| **#2** | Seated Peer | 342 – 448 | 3.57s | **97.2%** (104/107) | Entered scene at frame 342; smooth IoU association; 3-frame occlusion interpolated |
| **#3** | Boundary Edge Detection | 447 – 448 | 0.07s | **100.0%** (2/2) | Fleeting detection right before video cut-off (x2=479px); classified as `is_transient = True` |

### 3.2 Physical Evidence Object Associations

| Object Track | Class Label | Frame Span | Spatial Association | Physical Context |
| :---: | :---: | :---: | :---: | :--- |
| **Track #1** | `book` | 309 – 362 | `person_1` (desk surface) | Examination notes / study booklet |
| **Track #2** | `book` | 315 – 448 | `person_1` (desk surface) | Examination question paper / desk materials |
| **Track #3** | `book` | 330 – 448 | `person_1` (desk surface) | Examination answer sheet |
| **Track #4** | `cell phone` | 396 – 443 | `person_1` (hand / lap) | Handheld electronic device (Phone / Scientific Calculator) |
| **Track #5** | `cell phone` | 420 – 448 | `person_1` (hand / desk) | Handheld electronic device (Phone / Scientific Calculator) |
| **Track #6** | `cell phone` | 426 – 448 | `person_1` (lap vicinity) | Handheld electronic device (Phone / Scientific Calculator) |

---

## 4. Object Taxonomy & Ambiguity Disclosure

MS-COCO 80 categories do not include an explicit `calculator` class. Handheld electronic devices with rectangular aspect ratios are evaluated under calibrated geometric heuristics.
- In accordance with ethical proctoring guidelines, these candidate detections are flagged as `POSSIBLE_PHONE_INTERACTION` and `UNCERTAIN_ELECTRONIC_DEVICE` with explicit uncertainty disclosures:
  > *"COCO 80 taxonomy lacks a dedicated calculator class. Handheld electronic devices are treated as ambiguous objects requiring physical invigilator verification before any violation finding."*
- System strictly avoids accusatory declarations of academic dishonesty.

---

## 5. Verification Test Suite Results

### 5.1 Phase 5.7 Test Suite (`tests/test_phase5_7_reconciliation.py`)
- `test_job990_database_ground_truth`: **PASSED**
- `test_job990_api_tracks_endpoint`: **PASSED**
- `test_job990_api_results_partitioning`: **PASSED**
- `test_job990_html_audit_report`: **PASSED**
- `test_job990_excel_export_reconciliation`: **PASSED**
- `test_role_update_and_staff_alert_suppression`: **PASSED**
- `test_cross_job_tracks_and_events_isolation`: **PASSED**
- `test_traffic_job_unmodified`: **PASSED**
- **Outcome**: **8 / 8 Passed (100%)**

### 5.2 Core Regression Suite
- `test_phase5_6_proximity.py`: 7 passed
- `test_phase5_5_regression.py`: 6 passed
- `test_phase5_4_examination_excel_intelligence.py`: 10 passed
- `test_domain_isolation.py`: 9 passed
- `test_examination_module.py`: 8 passed
- `test_speed.py`: 2 passed
- **Outcome**: **42 / 42 Passed (100%)**

### 5.3 Frontend Production Build
- Command: `npm run build` (`tsc && vite build`)
- Output: 41 modules transformed, `dist/index.html` (0.78 kB), `dist/assets/index-CH5NLylY.js` (336.92 kB)
- **Outcome**: **0 errors, 0 warnings (Exit Code: 0)**
