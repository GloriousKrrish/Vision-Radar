# VisionRadar Phase 5.2 — Domain Routing Root Cause Analysis
## Strict Domain Isolation, Pipeline Routing, and Cross-Workbench Consistency

**Date:** October 21, 2026  
**Investigator:** Principal Software Architect & Senior Computer Vision Pipeline Engineer  
**Status:** ROOT CAUSE IDENTIFIED AND REPRODUCED  

---

### 1. The Reported Defect

When an examination video (e.g., `exam cheat.mp4`, `Exam 2.mp4`) is uploaded to VisionRadar:
1. The platform initially treats the video as belonging to the **Traffic Intelligence** domain.
2. The video canvas renders traffic-specific bounding box tags: `CAR`, `TRUCK`, and `UNCALIBRATED MOTION`.
3. The right-hand sidebar and tables render vehicle velocity metrics, traffic congestion charts, and vehicle speed tables.
4. The user is forced to manually click the **"🎓 Examination"** domain toggle to view classroom proctoring results.

---

### 2. End-to-End Execution Trace

Tracing an uploaded video from file selection to canvas rendering revealed the exact sequence of events:

```
[User selects file: Exam 2.mp4]
         │
         ▼
[handleFileUpload] in apps/web/src/components/WorkbenchView.tsx
         │
         ├── 1. POST /api/v1/projects/1/videos (Uploads file to backend)
         │
         ├── 2. POST /api/v1/videos/{video_id}/classify (Invokes SceneClassifier)
         │       │
         │       ▼
         │   [packages/visionradar/scene_router/classifier.py]
         │   - Samples 6 temporal frames.
         │   - Evaluates roadway cues vs classroom cues.
         │   - Straight perspective lines (desks, boards) score roadway_composite: 0.488.
         │   - Cues are borderline / inconsistent -> scene_type: AMBIGUOUS, confidence: 0.37.
         │   - requires_user_confirmation: True.
         │   - Line 142 fallback: recommended = TRAFFIC if avg_roadway >= avg_classroom.
         │   - Returns: { scene_type: "AMBIGUOUS", recommended_module: "traffic", requires_user_confirmation: true }
         │
         ├── 3. [WorkbenchView.tsx Line 1954-1955]
         │       const targetModule: AnalysisModule = rec?.recommended_module || 'traffic';
         │       startJobForModule(vidData.id, targetModule, true);
         │       --> COMPLETELY IGNORES rec.requires_user_confirmation!
         │       --> COMPLETELY IGNORES user's activeModule state!
         │       --> Blindly triggers startJobForModule with 'traffic'!
         │
         ├── 4. [startJobForModule] POST /api/v1/videos/{video_id}/jobs
         │       Payload: { config_json: { module: "traffic", processing_mode: "balanced" } }
         │
         ├── 5. [apps/api/main.py Line 366]
         │       module = (payload.config_json or {}).get("module", "traffic")
         │       if module == "examination": ...
         │       else: run_job_streaming(job.id, loop)  <-- TRAFFIC PIPELINE DISPATCHED!
         │
         ├── 6. [visionradar/worker/job_worker.py: run_job_streaming]
         │       - Instantiates YOLOX-Nano vehicle detector (COCO classes 2: car, 7: truck, 5: bus, 3: motorcycle).
         │       - Instantiates vehicle ByteTrack tracker.
         │       - Projects detections onto road homography plane.
         │       - Emits WebSocket frame_result with vehicle_class: "Car", "Truck".
         │
         └── 7. [WorkbenchView.tsx WebSocket onmessage]
                 - Receives type="frame_result" with vehicle tracks.
                 - Pushes into rtFrameBufferRef.
                 - Canvas renders "CAR", "TRUCK", and "UNCALIBRATED MOTION" over students!
```

---

### 3. Isolated Root Causes

#### Root Cause 1: Premature Auto-Dispatch in `handleFileUpload` Ignoring User Intent & Confirmation
In `apps/web/src/components/WorkbenchView.tsx`:
- Line 1954 sets `const targetModule: AnalysisModule = rec?.recommended_module || 'traffic';`.
- If the user had already clicked `🎓 Examination` in the UI before uploading, `handleFileUpload` stomped on `activeModule` and overrode it with `rec?.recommended_module`.
- If `rec.requires_user_confirmation` was `true`, `handleFileUpload` still auto-dispatched the job immediately without awaiting user confirmation.

#### Root Cause 2: Scene Router Ambiguity Falling Back to Traffic
In `packages/visionradar/scene_router/classifier.py`:
- In `_analyze_single_frame`, perspective lines from classroom desks and walls scored high on `road_lines_score`.
- When classification was `AMBIGUOUS` or confidence was low (< 0.65), lines 142 and 166 set:
  `recommended = AnalysisModule.TRAFFIC if avg_roadway >= avg_classroom else AnalysisModule.EXAMINATION`
  Instead of defaulting to `AnalysisModule.NONE` and strictly requiring user choice, it assigned `traffic`.

#### Root Cause 3: Backend Job Creation Missing Authoritative `domain` Column and Lenient Dispatch
In `packages/visionradar/models/entities.py` and `apps/api/main.py`:
- `ProcessingJob` had no explicit, dedicated `domain` column. The domain was tucked away inside an unvalidated JSON blob `config_json["module"]`.
- In `apps/api/main.py`, line 366:
  `module = (payload.config_json or {}).get("module", "traffic")`
  Any unrecognized, empty, or `"none"` module silently defaulted to `"traffic"`. There was no HTTP 400 validation for invalid domains.
- `JobResponse` did not return a top-level `domain` attribute, forcing frontend clients to parse `config_json` or guess.

#### Root Cause 4: Cache & Frame Buffer Cross-Domain Contamination
In `WorkbenchView.tsx`:
- When switching `activeModule` manually or changing `activeJobId`, `rtFrameBufferRef` and `frameMapRef` were not fully purged of vehicle tracks.
- Stale WebSocket messages from the prior traffic stream continued to arrive and populate the active frame buffer.

---

### 4. Required Corrective Actions

1. **Explicit Domain Contract (Phase B):**
   - Add authoritative `domain` attribute (`traffic` | `examination`) to `ProcessingJob` model and database schema.
   - Update `JobCreate` and `JobResponse` schemas to expose `domain`.
   - Reject invalid or missing domains at backend ingestion with HTTP 400.
   - Pipeline dispatcher must strictly read `job.domain`.

2. **Pre-Upload & Upload Domain Selection (Phase C):**
   - In `WorkbenchView.tsx`, allow the user to select the Domain (Traffic or Examination) **prior to upload**.
   - If the user has already selected `Examination`, preserve that choice and dispatch `examination` without auto-reverting to `traffic`.
   - If the Scene Router returns `requires_user_confirmation`, pause and display the recommendation banner, allowing the user to confirm before creating the job.

3. **Strict Cache & Stream Isolation (Phase D & E):**
   - Clear `rtFrameBufferRef`, `frameMapRef`, `realTracks`, `examTracks`, `violations`, and `examEvents` immediately upon job or domain transition.
   - Close active WebSocket and reject any incoming frame messages whose `job_id` does not match `activeJobIdRef.current`.
   - Canvas overlay must strictly inspect `activeJob.domain` before rendering either vehicle labels or person/exam overlays.

4. **Cross-Workbench Consistency & Verification (Phase F, G & H):**
   - SharedJobContext must read `job.domain` directly from the authoritative backend job record.
   - Excel export must enforce domain compatibility from `job.domain`.
   - Automated tests covering domain routing, upload dispatch, invalid domain rejection, and cache isolation.
