# VisionRadar Phase 5.7: Job #990 Multi-Tier Forensic Reconciliation Audit

**Document Reference**: `reports/phase5/PHASE57_JOB990_RECONCILIATION.md`  
**Evaluation Target**: Job #990 (`exam cheat.mp4`, Video ID 552)  
**Lead Investigators**: Principal Computer Vision Engineer, Data Integrity Engineer, Forensic Reporting Architect  
**Audit Timestamp**: 2026-10-22 06:15:00 UTC  

---

## 1. Executive Summary & Purpose

This audit establishes the empirical ground truth for **Processing Job #990**, tracing all data representations from raw database records, API endpoints, and frontend components to forensic HTML audit dossiers and Excel workbooks.

The objective is to resolve reported inconsistencies in:
1. **Person track vs. Object track accounting**
2. **Role assignment semantics (`UNKNOWN` vs. `STUDENT` vs. `INVIGILATOR`)**
3. **Event counts vs. frame detection counts vs. review status breakdown**
4. **Endpoint contract misalignment and frontend state aggregation**

---

## 2. Job #990 Source Video & Processing Provenance

| Field | Persisted Database Value | Provenance & Validation |
| :--- | :--- | :--- |
| **Job ID** | `990` | Primary key in `processing_jobs` table |
| **Analysis Domain** | `examination` | Validated modular pipeline separation |
| **Processing Status** | `SUCCEEDED` | Completed normally without pipeline crash |
| **Execution Stage** | `Completed` | Final worker stage reached |
| **Progress %** | `100.0%` | Monotonically finished |
| **Processing Started** | `2026-10-22 05:55:23 UTC` | Accurate wall-clock start |
| **Processing Completed** | `2026-10-22 05:55:53 UTC` | Execution duration = 30.0s |
| **Source Video ID** | `552` | Foreign key referencing `videos` table |
| **Source Video Filename**| `exam cheat.mp4` | Uploaded test sequence |
| **Container Duration** | `14.967s` (~15.0s) | Derived from container metadata |
| **Framerate (FPS)** | `30.0 FPS` | Standard video framerate |
| **Total Frames** | `449 frames` | Frame range: 0 to 448 |
| **Resolution** | `480 x 854` (portrait) | Standard mobile/classroom recording |
| **Video SHA-256 Hash** | `ab350d9f3f392af250d0cdeb31abae89fb1633e791a75ec626eae2b94831b0e2` | Verified byte integrity |

---

## 3. Persisted Database Inventory (Ground Truth)

### 3.1 Examination Track Records (`examination_tracks`)

Total rows persisted in database for Job #990: **9 rows**.

#### A. Human Examinee Person Tracks (`entity_type == 'person'`) — 3 Tracks

| Track ID | Entity Type | Frame Span | Duration (s) | Trajectory Points | Continuity Ratio | Assigned Role | Role Source | Seating Zone | Physical Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | `person` | 0 – 448 | 14.97s | 449 pts | 100.0% | `UNKNOWN` | `DEFAULT` | `None` | Continuous foreground examinee seated at desk throughout entire session |
| **#2** | `person` | 342 – 448 | 3.57s | 104 pts | 97.2% | `UNKNOWN` | `DEFAULT` | `None` | Late-entering examinee / peer leaning into left desk vicinity |
| **#3** | `person` | 447 – 448 | 0.07s | 2 pts | 100.0% | `UNKNOWN` | `DEFAULT` | `None` | Transient peripheral appearance at extreme frame boundary (x2=479/480px) right before video termination |

- **Total Unique Human Person Tracks**: **3**
- **Student-Role Tracks**: **0** (no automated assumptions made)
- **Invigilator-Role Tracks**: **0**
- **Unknown-Role Tracks**: **3** (strictly preserved as `UNKNOWN`)

#### B. Associated Physical Object Tracks (`entity_type != 'person'`) — 6 Tracks

| Track ID | Entity Type | Frame Span | Trajectory Points | Associated Person Zone | Object Class |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | `object_book` | 309 – 362 | 27 pts | `person_1` | Book / physical reading material |
| **#2** | `object_book` | 315 – 448 | 67 pts | `person_1` | Book / notes sheet |
| **#3** | `object_book` | 330 – 448 | 60 pts | `person_1` | Book / examination paper |
| **#4** | `object_cell phone` | 396 – 443 | 24 pts | `person_1` | Handheld electronic device (phone / calculator) |
| **#5** | `object_cell phone` | 420 – 448 | 15 pts | `person_1` | Handheld electronic device (phone / calculator) |
| **#6** | `object_cell phone` | 426 – 448 | 12 pts | `person_1` | Handheld electronic device (phone / calculator) |

- **Total Object Tracks**: **6**
- **Collision Note**: Object tracks share integer track IDs (`#1`, `#2`, `#3`) with Person tracks, requiring explicit `entity_type == 'person'` filtering during all track queries.

---

### 3.2 Examination Event Records (`examination_events`)

Total candidate event rows persisted in database for Job #990: **4 rows**.

| Event Identifier | Behavioral Category | Person Track | Start Time | End Time | Duration | Priority | Review Status | Reviewer Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `evt_phone_5c454adc` | `POSSIBLE_PHONE_INTERACTION` | `#1` | 13.20s | 14.93s | 1.73s | AMBER | **ACCEPTED** | "Verified by proctor" |
| `evt_phone_2a661411` | `POSSIBLE_PHONE_INTERACTION` | `#2` | 13.20s | 14.93s | 1.73s | AMBER | **PENDING** | `None` (Awaiting review) |
| `evt_prox_0525a41c` | `PEER_PROXIMITY` | `#1` | 11.37s | 14.93s | 3.57s | AMBER | **ACCEPTED** | "" (Confirmed by proctor) |
| `evt_trans_363f60ea` | `POSSIBLE_OBJECT_TRANSFER` | `#1` | 11.63s | 11.63s | 0.10s | AMBER | **PENDING** | `None` (Awaiting review) |

#### Review Status Breakdown
- **ACCEPTED**: **2** (`evt_phone_5c454adc`, `evt_prox_0525a41c`)
- **PENDING**: **2** (`evt_phone_2a661411`, `evt_trans_363f60ea`)
- **REJECTED**: **0**
- **INCONCLUSIVE**: **0**
- **Mathematical Sum**: $2 + 2 + 0 + 0 = \mathbf{4}$ (Exact 100% reconciliation with total events).

---

## 4. Multi-Tier Discrepancy Matrix

The table below compares the values reported across the five core application surfaces for Job #990:

| Metric | Database Ground Truth | Backend API (`/results`, `/preview`) | Frontend UI (`WorkbenchView`) | HTML Forensic Report (`/audit-report/html`) | Excel Workbook Export (`.xlsx`) | Reconciled Status / Root Cause |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Total Monitored Persons** | **3** | `3` (`/preview`) / `9` (`/results.tracks`) | **9** *(or 1–2 in live stream)* | **3** | **3** | **DISCREPANCY IN UI**: Frontend displayed `9` because `/results.tracks` returned all 9 database track rows without separating person tracks from object tracks. |
| **Confirmed Students** | **0** | `0` | **0** | **0** | **0** | **RECONCILED**: Correctly 0. System does not assume persons are enrolled students without roster verification. |
| **Authorized Staff** | **0** | `0` | **0** | **0** | **0** | **RECONCILED**: Correctly 0. |
| **Unknown Roles** | **3** | `3` (`/preview`) | **9** | **3** | **3** | **DISCREPANCY IN UI**: Frontend counted all tracks missing `STUDENT`/`STAFF` roles as unknown persons, including 6 books and phones. |
| **Associated Physical Objects** | **6** | `6` (`/preview`) | `0` (Unlisted / bundled) | `0` (In disclosure) | **6** (Tab 2, Section 2) | **PARTIAL**: Object tracks were not exposed as a distinct entity card in the UI overview. |
| **Candidate Event Records** | **4** | **4** | **4** | **4** | **4** | **RECONCILED**: Exactly 4 candidate events across all tiers. |
| **Distinct Candidate Event IDs** | **4** | **4** | **4** | **4** | **4** | **RECONCILED**: Identical UUID hashes preserved. |
| **Pending Review Events** | **2** | **2** | **2** | **2** | **2** | **RECONCILED**: Exactly 2 pending items. |
| **Accepted Review Events** | **2** | **2** | **2** | **2** | **2** | **RECONCILED**: Exactly 2 accepted items. |
| **Rejected Review Events** | **0** | **0** | **0** | **0** | **0** | **RECONCILED**: Exactly 0 rejected items. |
| **Inconclusive Review Events** | **0** | **0** | **0** | **0** | **0** | **RECONCILED**: Exactly 0 inconclusive items. |
| **Endpoint: `/examination/tracks`** | Exists in DB | **404 Not Found** | Cannot query | N/A | Queries DB directly | **DEFECT**: Missing dedicated REST endpoint `GET /api/v1/jobs/{job_id}/examination/tracks`. |

---

## 5. Root Cause Analysis

### Root Cause 1: Lack of Entity Separation in `/api/v1/jobs/{job_id}/examination/results`
- **Location**: `apps/api/main.py` lines 1662–1734
- **Mechanism**: The query `db.query(ExaminationTrackRecord).filter(job_id == job_id).all()` selects all rows in `examination_tracks`. Both human candidates (`entity_type == 'person'`) and evidence objects (`entity_type == 'object_book'`, `'object_cell phone'`) are bundled into a single `"tracks"` array.
- **Frontend Effect**: `WorkbenchView.tsx` set `examTracks = exData.tracks` (9 elements), then computed:
  ```tsx
  totalPersons = examTracks.length; // evaluates to 9 instead of 3
  unknownCount = examTracks.filter(t => !t.assigned_role || t.assigned_role === 'UNKNOWN').length; // evaluates to 9
  ```

### Root Cause 2: Missing Dedicated Tracks Endpoint
- **Location**: `apps/api/main.py`
- **Mechanism**: While `PATCH /api/v1/jobs/{job_id}/examination/tracks/{track_id}/role` was implemented, `GET /api/v1/jobs/{job_id}/examination/tracks` was never exposed. Any caller requesting detailed track metadata received HTTP 404.

### Root Cause 3: Track ID Namespace Overlap Between Persons and Objects
- **Location**: `packages/visionradar/examination/pipeline.py` lines 325 and 345
- **Mechanism**: `PersonTracker` allocates sequential IDs starting at 1 (`#1`, `#2`, `#3`). `ObjectTracker` also allocates sequential IDs starting at 1 (`#1`, `#2`, `#3`, `#4`, `#5`, `#6`). In the database and frontend, there is both a Person `#1` and an Object `#1` (book).
- **Frontend Effect**: When a proctor clicks to inspect Person `#1`, lookups like `tracks.find(t => t.track_id === 1)` can inadvertently grab the book object track if `entity_type` is not explicitly checked.

### Root Cause 4: Transient Boundary Track (#3) Accounting
- **Location**: Video frame 447–448
- **Mechanism**: Person Track #3 lasted for exactly 2 frames (0.07s) right as the clip cut off, with zero candidate events.
- **Reporting Need**: Forensic reports must distinguish "Sustained Monitored Examinees" (duration $\ge 1.0$s) from "Transient / Peripheral Visual Detections" so proctors understand whether 3 full examinees were present or 2 examinees plus 1 fleeting edge detection.

---

## 6. Corrective Remediation Action Plan

1. **Update API Response (`apps/api/main.py`)**:
   - Refactor `GET /api/v1/jobs/{job_id}/examination/results` to return clearly partitioned entities:
     - `person_tracks`: strictly `entity_type == 'person'` (3 tracks for Job #990).
     - `object_tracks`: strictly `entity_type != 'person'` (6 tracks for Job #990).
     - `summary`: pre-computed counts (`total_persons: 3`, `students: 0`, `staff: 0`, `unknown: 3`, `objects: 6`).
     - Retain `tracks` as backwards-compatible alias containing only person tracks or typed entities.
   - Implement `GET /api/v1/jobs/{job_id}/examination/tracks` returning comprehensive track inspection metadata.
2. **Update Frontend UI (`WorkbenchView.tsx`, `ExaminationComponents.tsx`)**:
   - Partition tracks so `totalPersons` derives strictly from `t.entity_type === 'person'`.
   - Update `ExaminationOverviewCard` to display both `3 Monitored Persons` and `6 Associated Objects`.
   - Update `ExaminationSidebarInspector`'s Person tab to show `Assigned Role`, `Role Source`, and provide proctor controls to update the role to `STUDENT`, `INVIGILATOR`, or `AUTHORIZED_STAFF`.
3. **Synchronize Forensic Audit Dossier & Excel Export**:
   - Reconcile classification headers and definitions so HTML report, Excel workbook, API, and UI cards present identical, verified counts.
   - Explicitly separate sustained examinee tracks from brief boundary appearances ($< 1.0$s).
4. **Preserve Database & Review History**:
   - Job #990's 2 ACCEPTED and 2 PENDING events will be preserved without modification.
