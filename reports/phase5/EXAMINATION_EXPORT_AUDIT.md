# VisionRadar Phase 5.4: Examination Excel Export Architecture Audit

**Audit Date**: 2026-10-21  
**Auditor**: Principal Data Engineer & Computer Vision Reporting Specialist  
**Target Job**: Job #825 (Domain: Examination Intelligence, Video: `exam cheat.mp4`, ID: 504)  
**Database**: SQLite (`data/visionradar.db` with WAL mode)  

---

## 1. Executive Summary & Problem Diagnosis

VisionRadar's Traffic Excel exporter has been validated and accepted by the user. The Examination exporter currently produces a downloadable `.xlsx` workbook, but empirical inspection of persisted database records versus generated sheets reveals significant gaps in traceability, data categorization, aggregation accuracy, and proctor evidence presentation.

The UI screenshot previously observed by the user reported:
- **Job ID**: #825
- **Domain**: Examination
- **Source Video**: `exam cheat.mp4`
- **Monitored Persons**: 9
- **Behavioral Events**: 4
- **Workbook Sheets**: 8

Our forensic database audit of Job #825 reveals that the UI count of **"9 monitored persons" is a severe aggregation defect**. The SQLite table `examination_tracks` contains 9 records for Job #825:
- **3 Person Tracks** (`entity_type == "person"`: Track #1, #2, #3)
- **6 Object Tracks** (`entity_type` is `object_book` [3 tracks] and `object_cell phone` [3 tracks])

The previous export engine and preview API counted all 9 rows indiscriminately, erroneously misclassifying books and mobile phones as "monitored persons". Furthermore, research experiment records from unrelated highway traffic benchmarks were leaking into classroom examination workbooks due to an unconstrained database query in `_build_research_experiments`.

---

## 2. End-to-End Data Flow Trace

```
1. Video Ingestion (`exam cheat.mp4`, 449 frames, 30 FPS, 14.97s)
   │
2. Examination Pipeline (`packages/visionradar/examination/pipeline.py`)
   ├── Person Detection & Tracking (YOLOX-Nano ONNX + PersonTracker) -> 3 Person Tracks
   ├── Object Detection & Association (ExaminationObjectTracker) -> 6 Object Tracks (3 books, 3 phones)
   └── ExaminationEventEngine -> 4 Candidate Events
   │
3. Database Persistence (`packages/visionradar/models/entities.py`)
   ├── `processing_jobs` -> Job #825 (status=SUCCEEDED, stage=Completed, telemetry_json saved)
   ├── `examination_tracks` -> 9 records (3 person, 6 object_*)
   ├── `examination_events` -> 4 records (2 ACCEPTED, 1 REJECTED, 1 PENDING)
   └── `examination_zones` -> 0 records (No zones configured for this video)
   │
4. API Retrieval (`apps/api/main.py`)
   ├── `/api/v1/jobs/825/export/excel/preview` -> Counts all 9 tracks as "tracks"; returns 8 planned sheets
   └── `/api/v1/jobs/825/export/excel` -> Calls `ExcelExportEngine.generate_workbook()`
   │
5. Export Aggregation & Workbook Generation (`packages/visionradar/services/excel_export.py`)
   ├── `_build_executive_summary` -> Filters `entity_type == "person"`, counts 3 persons (inconsistent with other sheets!)
   ├── `_build_examination_tracks` -> Dumps all 9 records (persons + books + phones) into one sheet
   ├── `_build_examination_events` -> Generates basic event table without evidence timeline
   ├── `_build_examination_zones` -> Returns None because 0 zones exist (sheet silently omitted)
   └── `_build_research_experiments` -> Dumps 5 unrelated highway traffic benchmarks into exam workbook!
```

---

## 3. Categorized Gap Analysis: Why Information Was Missing or Misleading

| Data Element | Current Status in DB | Issue Category | Root Cause & Evidence |
| :--- | :--- | :--- | :--- |
| **Monitored Person Count** | Exactly 3 person records (`id=1, 2, 3`) | **4. Incorrectly Aggregated** | `preview` API called `ExaminationTrackRecord.count()` without `entity_type == "person"`. The UI reported 9 persons instead of 3. |
| **Object Tracks (Phones, Books)** | 6 records in `examination_tracks` (`object_book`, `object_cell phone`) | **6. Present but Poorly Formatted** | Object tracks were dumped directly into the "Monitored Persons & Tracks" table, causing Track ID collisions (#1 person vs #1 book) and reporting "UNKNOWN" roles for books and phones. |
| **Student vs Staff Differentiation** | Roles default to `"UNKNOWN"` unless config provides roles | **1. Not Generated** / **5. Omitted by Exporter** | Model correctly avoids classifying roles by facial appearance. However, the exporter lacked explicit breakdown columns (Authorized Roster Match, Track Role Provenance). |
| **Proctor Review Decisions** | Persisted in `examination_events.review_status` (2 ACCEPTED, 1 REJECTED, 1 PENDING) | **5. Omitted by Exporter** | The workbook displayed review status in a single column in the event table but lacked an audit-grade "Event Review History" sheet detailing verdict reasons, reviewer notes, and dispositions. |
| **Chronological Evidence Timeline** | Persisted frame arrays in `supporting_objects_json["supporting_frames"]` | **5. Omitted by Exporter** | The detailed sequence of observed frames (e.g. 53 frames for phone interaction) was trapped inside a raw JSON dictionary and never exposed as a chronological evidence timeline. |
| **Seating Zones Sheet** | 0 records in `examination_zones` for Video #504 | **5. Omitted by Exporter** | When 0 zones exist, `_build_examination_zones` called `return` early. This caused the generated workbook to have 7 or 8 sheets while the preview API promised "Seating Zones", causing UI/workbook sheet mismatch. |
| **Research Experiments Leakage** | 5 records in `research_experiments` (Traffic benchmark data) | **4. Incorrectly Aggregated** | `_build_research_experiments` queried `ResearchExperimentDB.limit(25).all()` unconditionally, leaking traffic speed benchmark runs into classroom examination workbooks. |
| **Detection Records Sheet** | 603 sampled bounding box records | **6. Present but Poorly Formatted** | Sampled detections showed raw numbers without distinguishing person bounding boxes from object bounding boxes or providing timecode (`MM:SS.ms`). |

---

## 4. Forensic Inspection of Job #825 Persisted Data

### 4.1 Processing Job Record
- **ID**: `825`
- **Domain**: `examination`
- **Status**: `SUCCEEDED`
- **Stage**: `Completed`
- **Created At**: `2026-10-21 11:09:10 UTC`
- **Completed At**: `2026-10-21 11:09:42 UTC` (Runtime: 31.47s, Average FPS: 14.27)
- **Processed Frames**: `449 / 449`
- **Source Video**: ID 504 (`exam cheat.mp4`, 480x854, 30.0 FPS, 14.97s, H.264)

### 4.2 Track Breakdown in SQLite
```
Track ID | Entity Type        | First Frame | Last Frame | Duration | Assigned Zone | Observations
---------+--------------------+-------------+------------+----------+---------------+-------------
#1       | person             | 0           | 448        | 14.97s   | UNASSIGNED    | 449 frames
#2       | person             | 342         | 448        | 3.57s    | UNASSIGNED    | 104 frames
#3       | person             | 447         | 448        | 0.07s    | UNASSIGNED    | 2 frames
#1       | object_book        | 309         | 362        | 1.80s    | person_1      | 38 frames
#2       | object_book        | 315         | 448        | 4.47s    | person_1      | 106 frames
#3       | object_book        | 330         | 448        | 3.97s    | person_1      | 84 frames
#4       | object_cell phone  | 396         | 443        | 1.60s    | person_1      | 33 frames
#5       | object_cell phone  | 420         | 448        | 0.97s    | person_1      | 20 frames
#6       | object_cell phone  | 426         | 448        | 0.77s    | person_1      | 16 frames
```
*Crucial takeaway: Exactly 3 persons were tracked. Object tracks #1–#6 represent evidence items spatially associated with Person Track #1.*

### 4.3 Candidate Event Breakdown in SQLite
```
Event ID            | Event Type                   | Track | Priority | Review Status | Conf  | Evidence Strength | Supporting Frames
--------------------+------------------------------+-------+----------+---------------+-------+-------------------+------------------
evt_phone_88049038  | POSSIBLE_PHONE_INTERACTION   | #1    | AMBER    | REJECTED      | 0.206 | HIGH              | 53 frames
evt_phone_0c7e1881  | POSSIBLE_PHONE_INTERACTION   | #2    | AMBER    | ACCEPTED      | 0.206 | HIGH              | 47 frames
evt_prox_c71564f8   | PEER_PROXIMITY               | #1    | AMBER    | PENDING       | 0.780 | MEDIUM            | 50 frames
evt_trans_14d46d78  | POSSIBLE_OBJECT_TRANSFER     | #1    | AMBER    | ACCEPTED      | 0.740 | MEDIUM            | 1 frame
```
*Crucial takeaway: The review decisions are already captured: 2 Accepted, 1 Rejected, 1 Pending.*

---

## 5. Architectural Upgrade Plan for Phase 5.4

To transform this into an enterprise-grade, traceable examination report:
1. **Preserve Traffic Export Unconditionally**: All changes will be scoped strictly inside `domain == "examination"` branching. The Traffic workbook builder remains 100% untouched.
2. **Sheet Architecture (10 Cohesive Sheets)**:
   1. `Executive Summary`: High-level executive overview, accurate person vs object counts, review progress, ethical caveats.
   2. `Monitored Persons & Tracks`: Strict filter `entity_type == "person"`. Comprehensive individual intelligence with roster matching, coverage %, associated event counts, and tracking quality notes.
   3. `Examination Events`: Complete event register with priority, evidence strength, confidence breakdown, supporting objects, and proctor review verdicts.
   4. `Evidence Timeline`: Chronological event and evidence observation log mapped to video timestamps (`MM:SS.ms`).
   5. `Detection Records`: Filterable detection samples clearly separating person boxes from object detections with timecode.
   6. `Event Review History`: Dedicated human-in-the-loop audit sheet with proctor verdicts, timestamps, and reviewer notes.
   7. `Seating Zones`: Clear seating zone layout or explicit informational status when unconstrained.
   8. `Processing Metadata`: Model provenance, ONNX runtime, hardware info, and known ethical limitations.
   9. `Data Dictionary`: Exhaustive glossary of every field, unit, and safe legal/academic interpretation.
   10. `Export Metadata`: Cryptographic SHA-256 seal, export mode, generation timestamp, and non-repudiation signature.
3. **Align Preview API**: Update `/api/v1/jobs/{job_id}/export/excel/preview` in `apps/api/main.py` so that `tracks` returns true person count (3), exposes `object_tracks` (6), `events` (4), and returns the exact 10 planned sheets.
4. **Remove Unrelated Research Leakage**: Do not call `_build_research_experiments` for examination jobs.

---
*End of Audit. Proceeding to Implementation.*
