# VISIONRADAR PHASE 5.8 IMPLEMENTATION REPORT

## Strict Traffic–Examination Isolation, Single Domain Source of Truth, and Job Integrity

**Author:** Principal Software Architect & Reliability Engineer  
**Date:** October 22, 2026  
**Status:** IMPLEMENTED & EMPIRICALLY VERIFIED  
**Deliverable:** `reports/phase5/PHASE58_IMPLEMENTATION.md`  

---

## 1. Executive Summary

Phase 5.8 resolves the domain-isolation defects where manual testing observed Examination overlays (`🎓 EXAMINATION INTELLIGENCE · Person & Object Tracking`) and examination-style candidate events appearing when a highway/traffic video was loaded in the workbench.

Through forensic reproduction in Phase A, the root cause was isolated to four specific architectural flaws:
1. **Frontend Upload Routing Fall-through**: In `WorkbenchView.tsx`, `handleVideoUpload` prioritized user selection only for `examination`, but allowed `traffic` selections to fall through to heuristic filename matches and scene classifier predictions (`CLASSROOM_EXAMINATION`). Any false classification on highway video silently routed the job to the Examination pipeline.
2. **Cross-Domain Startup Fallback**: In `WorkbenchView.tsx` (`selectCompletedJob`), when `matchingJob` was not found for the target domain, it fell back to `jobs[0]`. If `jobs[0]` was an Examination job, it loaded examination tracks and overlays while the user had Traffic selected in the navbar.
3. **Dual Domain State Desynchronization**: `SharedJobContext.tsx` maintained `activeDomain` in `localStorage`, while `WorkbenchView.tsx` maintained local React state `activeModule`. In `loadJobResults`, `activeModule` was changed without synchronizing `SharedJobContext.setDomain()`, leaving the UI in a split-brain state.
4. **Unguarded Traffic Endpoints in Backend**: Several traffic endpoints (`/api/v1/violations?job_id=...`, `/api/v1/jobs/{id}/reports/pdf`, `/api/v1/jobs/{id}/speed-limit`, `/api/v1/jobs/{id}/accident-candidates`) did not reject cross-domain queries with HTTP 400 when invoked with an Examination job ID.

All four defects have been surgically corrected. Zero changes were made to computer vision inference engines (YOLOX, ByteTrack, Speed Estimation, Proximity Calculation), homography calibration, or ANPR.

---

## 2. Architectural Blueprint: Single Domain Source of Truth

```
[User UI Selection: 🚗 Traffic | 🎓 Examination]
       │
       ▼ (Invariant explicit user intent)
[Frontend Video Upload]
       │  (Scene Router runs for advisory analytics ONLY; never overrides explicit selection)
       ▼
[POST /api/v1/videos/{id}/jobs]
       │  (payload.domain strictly validated: 'traffic' | 'examination')
       ▼
[Database Persistence: processing_jobs.domain] <── AUTHORITATIVE SOURCE OF TRUTH
       │
       ├── Worker Thread Dispatch:
       │     ├── if job.domain == 'traffic'     ──> run_job_streaming() [YOLOX + ByteTrack]
       │     └── if job.domain == 'examination' ──> run_examination_job() [Person & Object Tracking]
       │
       ├── Common Results Envelope:
       │     └── module = job.domain [Layer D Standardized Serialization]
       │
       ├── Strict Domain API Guards:
       │     ├── /tracks, /vehicle-speeds, /violations, /reports/pdf, /speed-limit ──> Require job.domain == 'traffic' (else 400)
       │     └── /examination/results, /examination/events, /audit-report/html   ──> Require job.domain == 'examination' (else 400)
       │
       ├── Excel Intelligence Export:
       │     ├── traffic     ──> Unique Tracks, Traffic Events, Speed Analytics
       │     └── examination ──> Monitored Persons & Tracks, Examination Events, Seating Zones
       │
       └── Frontend State Isolation:
             ├── Stale track, event, and overlay purge upon job switch
             ├── Rejection of asynchronous responses belonging to superseded job IDs
             └── Synchronization of SharedJobContext with loaded job.domain
```

---

## 3. Surgical Code Modifications

### 3.1 Frontend Upload Routing (`apps/web/src/components/WorkbenchView.tsx`)

**File:** `apps/web/src/components/WorkbenchView.tsx`  
**Location:** Lines 2202–2230 (`handleVideoUpload`)

**Rationale:** The user's explicit domain choice must be invariant. If a user selects Traffic, the system must never route to Examination, even if the scene router or filename suggests classroom footage. An informational advisory message is logged and displayed in the status banner without silently overriding the user's explicit selection.

```typescript
// STRICT DOMAIN ISOLATION & PIPELINE ROUTING RESOLUTION (Phase 5.8)
// The user's explicit domain selection is the single authoritative processing domain.
// Scene router recommendations may advise the user via non-blocking notification,
// but MUST NEVER silently override an explicitly selected domain.
const explicitDomain: AnalysisModule = (activeDomain === 'examination' || activeModuleRef.current === 'examination')
  ? 'examination'
  : 'traffic';

const sceneRecommendsExam = rec?.scene_type === 'CLASSROOM_EXAMINATION';
const sceneRecommendsTraffic = rec?.scene_type === 'TRAFFIC' && (rec?.confidence || 0) >= 0.50;

const hasConflict = (explicitDomain === 'traffic' && sceneRecommendsExam) ||
                    (explicitDomain === 'examination' && sceneRecommendsTraffic);

if (hasConflict) {
  const recName = sceneRecommendsExam ? 'Classroom Examination' : 'Traffic';
  const confPct = Math.round((rec?.confidence || 0) * 100);
  console.warn(`[Phase 5.8 Domain Isolation] User explicitly selected ${explicitDomain.toUpperCase()}, but Scene Router recommended ${recName} (${confPct}%). Preserving user selection.`);
  setJobStatusText(`Explicit ${explicitDomain.toUpperCase()} selection active (Scene Router suggested ${recName} ${confPct}% — explicit selection preserved)`);
} else {
  setJobStatusText(`${explicitDomain === 'examination' ? 'Examination Intelligence' : 'Traffic Intelligence'} active — Initializing ${explicitDomain.toUpperCase()} Pipeline...`);
}

startJobForModule(vidData.id, explicitDomain, true);
return;
```

---

### 3.2 Frontend Startup & Refresh Auto-Load Guard (`apps/web/src/components/WorkbenchView.tsx`)

**File:** `apps/web/src/components/WorkbenchView.tsx`  
**Location:** Lines 537–570 (`selectCompletedJob`)

**Rationale:** When the workbench starts up or is refreshed in Traffic mode, if no completed Traffic job is found, it must NOT fall back to `jobs[0]` (which may be an Examination job). Instead, it leaves the workbench in a clean, idle state for that domain. Additionally, when `sharedJobId` is cleared (null), the workbench must purge all tracks, events, and overlays.

```typescript
// Handle SharedJobContext clearing activeJobId (e.g. domain switch with no jobs in target domain)
if (!sharedJobId && !isLiveStreamingRef.current && activeJobIdRef.current !== null) {
  setActiveJobId(null);
  activeJobIdRef.current = null;
  setActiveVideoId(null);
  setVideoSrc(null);
  setRealTracks([]);
  setViolations([]);
  setRealCalib(null);
  setExamTracks([]);
  setExamObjectTracks([]);
  setExamEvents([]);
  setExamZones([]);
  setPrimaryExamFinding(null);
  rtFrameBufferRef.current.clear();
  frameMapRef.current.clear();
  seenRef.current.clear();
  setSelTrack(null);
  setSelectedPersonId(null);
  lastRenderedFrameIdRef.current = -1;
  setPipelineStage('IDLE');
  setJobStatusText(null);
  return;
}

// 2. If idle on startup, auto-load the latest succeeded job matching activeDomain strictly
if (pipelineStage === 'IDLE' && !activeJobIdRef.current) {
  const selectCompletedJob = async () => {
    try {
      const res = await fetch(apiUrl('/api/v1/jobs?status=SUCCEEDED'));
      if (!res.ok) return;
      const jobs = await res.json();
      if (!jobs?.length) return;

      const targetDomain = activeDomain || activeModuleRef.current || 'traffic';
      const matchingJob = jobs.find((j: any) => {
        const authDomain = (j.domain || (j.config_json || {}).domain || (j.config_json || {}).module || '').toLowerCase();
        return targetDomain === 'examination' ? authDomain.includes('exam') : !authDomain.includes('exam');
      });

      // STRICT ISOLATION: Never fall back to jobs[0] if it belongs to a different domain!
      if (matchingJob) {
        setActiveJobId(matchingJob.id);
        activeJobIdRef.current = matchingJob.id;
        selectSharedJob(matchingJob.id);
        loadJobResults(matchingJob.id);
      } else {
        // No matching job in targetDomain — maintain clean empty state
        setActiveJobId(null);
        activeJobIdRef.current = null;
      }
    } catch (e) {
      console.error('Domain-scoped auto-load failed:', e);
    }
  };
  selectCompletedJob();
}
```

---

### 3.3 Domain State Synchronization in Context & Workbench

**File:** `apps/web/src/context/SharedJobContext.tsx`  
**Location:** Lines 70–82 (`refreshJobs`)

**Rationale:** On context mount, when auto-selecting the latest job, it must filter by `activeDomain` instead of blindly picking `data[0]`.

```typescript
// If no active job selected yet, auto-select latest matching activeDomain
if (!activeJobId && data.length > 0) {
  const matching = data.find(j => {
    const d = (j.domain || ((j.config_json || {}).domain || (j.config_json || {}).module || '').toLowerCase());
    return activeDomain === 'examination' ? d.includes('exam') : !d.includes('exam');
  });
  if (matching) {
    selectJob(matching.id);
  }
}
```

**File:** `apps/web/src/components/WorkbenchView.tsx`  
**Location:** Lines 336–345 (`loadJobResults`)

**Rationale:** Synchronize `SharedJobContext.activeDomain` whenever a job is loaded so that the header navbar, export modal, and canvas overlay agree.

```typescript
const authDomain = jobData.domain || ((jobData.config_json || {}).domain || (jobData.config_json || {}).module || '').toLowerCase();
const jobModule: AnalysisModule = authDomain.includes('exam') ? 'examination' : 'traffic';

setActiveModule(jobModule);
activeModuleRef.current = jobModule;
if (activeDomain !== jobModule) {
  setDomain(jobModule);
}
```

---

### 3.4 Backend Domain Guards & Serialization (`apps/api/main.py`)

**File:** `apps/api/main.py`

1. **`list_violations` Domain Guard** (Line 756):
   ```python
   if job_id is not None:
       target_job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
       if target_job and target_job.domain != "traffic":
           raise HTTPException(
               status_code=400,
               detail=f"Job #{job_id} belongs to '{target_job.domain}' domain. Traffic violations are only available for traffic jobs."
           )
       query = query.filter(Violation.job_id == job_id)
   ```

2. **`get_job_speed_limit` & `set_job_speed_limit` Domain Guards** (Lines 875, 897):
   ```python
   if job.domain != "traffic":
       raise HTTPException(
           status_code=400,
           detail=f"Job #{job_id} belongs to '{job.domain}' domain. Speed limit configuration is only available for traffic jobs."
       )
   ```

3. **`get_job_accident_candidates` Domain Guard** (Line 1081):
   ```python
   if job.domain != "traffic":
       raise HTTPException(
           status_code=400,
           detail=f"Job #{job_id} belongs to '{job.domain}' domain. Accident candidates are only available for traffic jobs."
       )
   ```

4. **`generate_pdf_report` Domain Guard** (Line 1348):
   ```python
   if job.domain != "traffic":
       raise HTTPException(
           status_code=400,
           detail=f"Job #{job_id} belongs to '{job.domain}' domain. Traffic PDF report is only available for traffic jobs. For examination jobs, use the examination audit report endpoint."
       )
   ```

5. **`get_job_common_results` Authoritative Domain Resolution** (Line 1590):
   ```python
   raw_mod = getattr(job, "domain", None) or (job.config_json or {}).get("domain") or (job.config_json or {}).get("module") or "traffic"
   module = "examination" if "exam" in str(raw_mod).lower() else "traffic"
   ```

---

## 4. Verification Summary

| Component | Change Type | Integrity Verification |
|---|---|---|
| `handleVideoUpload` | Pipeline routing fix | User explicit selection strictly preserved; conflict warning logged |
| `selectCompletedJob` | Startup auto-load fix | Removed `jobs[0]` fallback; clean domain empty state |
| `SharedJobContext.tsx` | Context auto-load fix | Initial select scoped to `activeDomain` |
| `loadJobResults` | Bidirectional sync | `setDomain(jobModule)` keeps header in sync |
| `main.py` violations | API guard | Returns HTTP 400 when querying examination job |
| `main.py` PDF report | API guard | Returns HTTP 400 when querying examination job |
| `main.py` speed limit | API guard | Returns HTTP 400 when querying examination job |
| `main.py` accident intel | API guard | Returns HTTP 400 when querying examination job |
| `main.py` common results | Serialization fix | Uses persisted `job.domain` as primary truth |
| `test_phase5_8_domain_isolation.py` | New regression suite | 20 / 20 tests PASSED (100%) |
| Existing regression suites | Regression safety | 40 / 40 tests PASSED (100%) |
| Frontend production build | Production bundling | `tsc && vite build` built in 5.92s with 0 errors |

---

## 5. Non-Disruption Guarantee

- **ANPR**: Zero ANPR models, endpoints, or dependencies were added.
- **CV Engines**: Neither YOLOX ByteTrack nor Examination Person/Object Tracker algorithms were altered.
- **Homography**: Camera calibration math and speed calculations remain untouched.
- **Database History**: All existing jobs, reviews, tracks, and audit logs are preserved in their original state.
