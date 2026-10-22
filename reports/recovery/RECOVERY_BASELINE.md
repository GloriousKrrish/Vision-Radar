# RECOVERY_BASELINE: VisionRadar Emergency Recovery

## 1. Executive Summary & Source-Control State

- **Date:** 2026-10-22
- **Trigger:** Manual inspection identified regressions after Phase 5.8:
  1. Traffic workbench behavior broken / frozen on loading screen.
  2. UI compressed into a tiny central box on modern high-DPI displays (e.g. 5480x2772).
  3. Video playback failing or stuck on "Loading Real Traffic Video...".
  4. Cross-domain confusion between Traffic and Examination modules when loading jobs or switching domains.
  5. Slow API responses due to unbounded 990-job queries blocking the FastAPI event loop.

### Source Control State:
- **Branch:** `main` (commit `805ae47` HEAD)
- **Uncommitted modified files:**
  - `apps/api/main.py`
  - `apps/web/src/components/WorkbenchView.tsx`
  - `apps/web/src/context/SharedJobContext.tsx`
- **Untracked files:**
  - `reports/phase5/PHASE58_BASELINE_AUDIT.md`
  - `reports/phase5/PHASE58_FINAL_VERIFICATION.md`
  - `reports/phase5/PHASE58_IMPLEMENTATION.md`
  - `reports/phase5/PHASE58_REGRESSION_RESULTS.md`
  - `scripts/reproduce_phase58_scenarios.py`
  - `tests/test_phase5_8_domain_isolation.py`

---

## 2. Recent Changes Relevant to Regressions (Phase 5.8 Audit)

1. **`WorkbenchView.tsx`:**
   - Introduced domain synchronization hook `if (activeDomain !== jobModule) { setDomain(jobModule); }` in `loadJobResults`.
   - Introduced an active job clearance hook `if (!sharedJobId && !isLiveStreamingRef.current && activeJobIdRef.current !== null)` which aggressively purged active state whenever `sharedJobId` became null.
   - Initialized `isRealVideoRef` to `true` with no active video source, causing the canvas render loop to paint a blank `#0F172A` box with text `"Loading Real Traffic Video... Select or upload a real video file..."`.
   - When no job was loaded or video was empty, synthetic demo mode was inhibited.
2. **`SharedJobContext.tsx`:**
   - `setDomain` searched `jobList` for a matching job in the new domain.
   - When `jobList` was empty or still loading asynchronously from `/api/v1/jobs`, `matchingJob` was `undefined`.
   - As a result, `setDomain` called `setActiveJobId(null); setActiveJob(null);`, causing `WorkbenchView` to clear the current job and video playback instantly.
3. **`apps/api/main.py`:**
   - `/api/v1/jobs` executed `db.query(ProcessingJob).order_by(ProcessingJob.created_at.desc()).all()` with **no query limit**.
   - With 990 jobs in the database, this transferred **44.5 Megabytes of JSON**, taking over **10.35 seconds** and synchronously blocking the FastAPI event loop.
4. **Layout & Responsive Sizing (`index.css` & `WorkbenchView.tsx`):**
   - `apps/web/src/index.css`: `main { max-width: 1400px; margin: auto; }` locked the application into a 1400px fixed width. On high-resolution displays (like the user's 5480x2772 screen), this consumed under 25% of the viewport width.
   - `WorkbenchView.tsx`: Canvas had fixed internal dimensions `width={800} height={450}` without fluid responsive sizing (`width: 100%`, `maxHeight: 70vh`, `aspectRatio: 16/9`), causing the video display to appear tiny and compressed.
   - The `.grid` layout had fixed `minmax(0, 1fr) 320px` which constrained sidebars and tables.

---

## 3. Confirmed Regression Points

| Component | Failure Mode | Root Cause |
| :--- | :--- | :--- |
| **Video Playback / Persistence** | Stuck on "Loading Real Traffic Video..." | `isRealVideoRef` was true while `videoSrc` was null or player was waiting for metadata; no fallback to sample video or interactive upload/demo mode. Race condition in `SharedJobContext.setDomain` wiped `activeJobId`. |
| **Workbench Layout** | UI compressed into a tiny central box | `main { max-width: 1400px }` and fixed canvas dimensions without fluid responsive container scaling on high-resolution displays. |
| **Domain Isolation** | Cross-domain confusion on direct URL load (`?job_id=990`) | `SharedJobContext` and `WorkbenchView` had competing domain/job synchronization effects with uncoordinated state wipes. |
| **API Performance** | 10.35s hang on page load | Unbounded `GET /api/v1/jobs` serializing 44.5 MB of JSON for 990 jobs on every page refresh. |

---

## 4. Known-Good Behavior to Restore

1. **Traffic Module:**
   - Vehicle bounding boxes (Car, Bus, Truck, Motorcycle) rendered in green/cyan radar HUD.
   - Monocular road calibration quadrilateral (`P1-P4`) and real-world metric speed estimation (`XX km/h`).
   - Speeding violations log with evidence snapshots and threshold tolerance adjustments.
   - Traffic vehicle detection log and historical tracking tables.
   - Seamless loading of default sample video (`Traffic1.mp4`) or synthetic demo mode if no video is active.
2. **Examination Module:**
   - Person and object tracking with role assignments (Candidate, Invigilator, Staff).
   - Seating zone overlays (`📍 DESK_01`, `📍 PODIUM_STAFF`).
   - Candidate observational events (`POSSIBLE_PHONE_INTERACTION`, `PEER_PROXIMITY`).
   - Human proctor review controls and audit report exports.
3. **Responsive Workbench:**
   - Fluid desktop layout expanding across widescreen viewports (up to 2560px/4K/5K) without horizontal overflow.
   - Canvas dynamically scaling to container width while maintaining 16:9 aspect ratio and exact coordinate mapping.
   - Sidebar inspector readable and usable alongside the video canvas.

---

## 5. Safe Recovery Strategy (Non-Destructive)

1. **Backend Optimization & Protection:**
   - Add a default `limit: int = 50` parameter to `GET /api/v1/jobs` (with optional `domain` and `status` query filters) to eliminate the 10.35s event loop stall while maintaining complete backwards compatibility.
   - Ensure domain validation guards on domain-specific endpoints (`/violations`, `/speed-limit`, `/accident-candidates`, `/pdf`, `/examination/*`) remain strict.
2. **Responsive CSS Restoration (`index.css` & `WorkbenchView.tsx`):**
   - Update `main` container in `index.css` from `max-width: 1400px` to `width: 100%; max-width: min(98vw, 2400px); margin: 0 auto; padding: 16px;`.
   - Update canvas styling to `width: 100%; height: auto; aspect-ratio: 16 / 9; max-height: 72vh; object-fit: contain;` while preserving coordinate transformations for click/hover handlers (`clientX` to canvas `W/H` ratio).
   - Expand grid layout with responsive column rules (`minmax(0, 1fr) clamp(320px, 24vw, 420px)`).
3. **Video Persistence & State Stability (`SharedJobContext.tsx` & `WorkbenchView.tsx`):**
   - Prevent `SharedJobContext.setDomain` from setting `activeJobId = null` if `jobList` is still loading or if a job is currently being fetched.
   - In `WorkbenchView.tsx`, ensure that loading an existing job correctly initializes `videoSrc` from `/api/v1/videos/{video_id}/stream`.
   - If no video is loaded in Traffic mode, display an actionable empty state with a 1-click "Load Sample Traffic Video (Traffic1.mp4)" button or instant Demo Mode toggle, preventing the frozen black loading screen.
4. **Empirical Verification:**
   - Test switching between Traffic (Job #983) and Examination (Job #990).
   - Verify responsive rendering across standard and high-DPI viewports.
   - Run complete Python test suites and frontend production build.
