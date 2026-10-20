# Known Limitations and Unverified Capabilities

**Initiative**: Universal Video Intelligence Platform  
**Date**: October 19, 2026  
**Auditing Standard**: Empirical Transparency & Evidence-First Reporting  

---

## 1. Ground Truth & Representative Footage Disclosure

> [!IMPORTANT]
> **No Representative Real-World Classroom/Exam Video in Repository**:  
> The repository contains physical test videos for traffic analysis (`Traffic1.mp4`), but no pre-recorded, multi-camera, real-world classroom examination footage was present.  
> As a result, the Examination Intelligence Module has been empirically verified using:
> 1. Real traffic video negative testing (verifying zero false triggers of classroom rules on traffic footage).
> 2. Synthetic, controlled video test patterns with injected moving person centroids and object bounding boxes.
> 3. Standard COCO class extraction on unit test frames.  
> **Claim Status**: Detection accuracy on dense, real-world university examination halls remains **UNVERIFIED on representative field footage** until benchmarked with production classroom datasets.

---

## 2. Model & CV Limitations

### A. Small Handheld Object Occlusion
- YOLOX-Nano is an ultra-lightweight 0.91M parameter network.
- When mobile phones or paper slips are smaller than ~20x20 pixels (e.g., student seated in the rear row of a large lecture hall), detection recall naturally degrades.
- Hand occlusion (student covering a phone with palm) prevents visual identification by standard COCO bounding box heads.

### B. Seating Zone Perspective Distortion
- Seating zones are evaluated using 2D image plane polygons.
- In classrooms with steep stadium seating or extreme perspective foreshortening, students seated in back rows appear smaller and can overlap with adjacent zone boundaries.
- Future enhancement: Introduce 3D classroom homography calibration similar to the traffic module's metric homography matrix.

### C. Live Stream Ingestion (Scope Boundary)
- In accordance with Section 8 of the initiative specification, live stream ingestion (RTSP/WebRTC) is architected through timestamped frame interfaces and bounded queues, but is **not active in this release**.
- Only uploaded video files (MP4/WebM) are currently processed.

---

## 3. Rollback Instructions

In the unlikely event that this additive architecture must be completely rolled back, execute the following commands:

```bash
# 1. Discard working tree additions
git restore apps/api/main.py apps/web/src/types.ts apps/web/src/components/WorkbenchView.tsx packages/visionradar/models/entities.py

# 2. Remove added package files
rm -rf packages/visionradar/scene_router/
rm -rf packages/visionradar/examination/
rm -rf packages/visionradar/models/envelope.py
rm -rf apps/web/src/components/ExaminationComponents.tsx
rm -rf tests/test_scene_router.py tests/test_examination_module.py tests/test_universal_pipeline_integration.py
rm -rf reports/universal_intelligence/

# 3. Clean database (additive tables only)
python -c "from visionradar.models.database import engine; from sqlalchemy import text; \
with engine.connect() as conn: \
    conn.execute(text('DROP TABLE IF EXISTS scene_classifications')); \
    conn.execute(text('DROP TABLE IF EXISTS examination_events')); \
    conn.execute(text('DROP TABLE IF EXISTS examination_tracks')); \
    conn.execute(text('DROP TABLE IF EXISTS examination_zones')); \
    conn.commit()"
```
This safely removes all added components without touching any traffic tables, traffic models, or existing user data.
