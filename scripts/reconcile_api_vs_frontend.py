import os
import sys
import json
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from apps.api.main import app
from visionradar.models.database import get_db
from visionradar.models.entities import ProcessingJob, Video, Track

client = TestClient(app)

def reconcile():
    db = next(get_db())
    job = db.query(ProcessingJob).filter(ProcessingJob.id == 64).first()
    if not job:
        job = db.query(ProcessingJob).filter(ProcessingJob.status == "SUCCEEDED").order_by(ProcessingJob.id.desc()).first()
    if not job:
        print("ERROR: No succeeded job found")
        return

    video = db.query(Video).filter(Video.id == job.video_id).first()
    
    print("==================================================")
    print("VISIONRADAR -- API <-> FRONTEND RECONCILIATION AUDIT")
    print("==================================================")
    print(f"Job ID: {job.id} | Video ID: {job.video_id} | Filename: {video.filename if video else 'N/A'}")
    print(f"Resolution: {video.width if video else 1920}x{video.height if video else 1080} @ {video.fps if video else 29.97:.2f} FPS")

    # Evaluation Target Frame: Frame 150 (t = 5.00s)
    target_frame = 150
    fps = video.fps if video else 29.97
    target_time = target_frame / fps

    print(f"\nTarget Evaluation Frame: {target_frame} (Timestamp: {target_time:.2f}s)")

    # 1. API RESPONSE FOR FRAME 150
    api_res = client.get(f"/api/v1/jobs/{job.id}/frames/{target_frame}")
    print(f"\n--- 1. CANONICAL API RESPONSE (GET /api/v1/jobs/{job.id}/frames/{target_frame}) ---")
    print(f"HTTP Status: {api_res.status_code}")
    api_data = api_res.json() if api_res.status_code == 200 else {}
    print(f"API Returned Frame Index: {api_data.get('frame_index')}")
    print(f"API Returned Timestamp: {api_data.get('timestamp_s')}s")
    print(f"API Returned Detections Count: {len(api_data.get('detections', []))}")
    
    api_det_map = {}
    for d in api_data.get('detections', []):
        tid = d['track_id']
        api_det_map[tid] = d
        print(f"  [API Track #{tid:2d}] Class: {d['class_name']} | Conf: {d['confidence']:.4f} | bbox_xyxy: {d['bbox_xyxy']} | anchor_xy: {d['anchor_xy']}")

    # 2. FRONTEND RECONCILIATION (Emulating WorkbenchView.tsx logic for active frames)
    tracks = db.query(Track).filter(Track.job_id == job.id).all()
    print("\n--- 2. FRONTEND MATCHING RECONCILIATION (WorkbenchView.tsx logic) ---")
    
    W, H = 800, 450
    vidW = video.width if video else 1920
    vidH = video.height if video else 1080
    scaleX = W / vidW
    scaleY = H / vidH
    print(f"Video Source: {vidW}x{vidH} | Canvas Display: {W}x{H} | ScaleX: {scaleX:.4f} | ScaleY: {scaleY:.4f}")

    eval_cases = [(12, 120), (15, 200)]
    for tid, f_target in eval_cases:
        trk = db.query(Track).filter(Track.job_id == job.id, Track.track_id == tid).first()
        print(f"\n>>> RECONCILIATION FOR TRACK #{tid} AT REQUESTED FRAME {f_target} <<<")
        if not trk:
            print(f"  Track #{tid}: NOT FOUND in Job #{job.id}")
            continue

        # Get API response for f_target
        f_api_res = client.get(f"/api/v1/jobs/{job.id}/frames/{f_target}")
        f_api_data = f_api_res.json() if f_api_res.status_code == 200 else {}
        api_dets = f_api_data.get("detections", [])
        api_match = next((d for d in api_dets if d['track_id'] == tid), None)

        traj = trk.trajectory_json or []
        matched_pt = next((pt for pt in traj if abs(pt.get("frame_index", -999) - f_target) <= 6), None)

        if matched_pt:
            fe_frame = matched_pt.get("frame_index")
            fe_raw_bbox = matched_pt.get("bbox") or [0, 0, 0, 0]
            rx, ry, rw, rh = fe_raw_bbox
            if rw > rx:
                rw = rw - rx
            if rh > ry:
                rh = rh - ry
            
            fe_bbox_xyxy = [round(rx, 1), round(ry, 1), round(rx + rw, 1), round(ry + rh, 1)]
            bx = rx * scaleX
            by = ry * scaleY
            bw = max(15, rw * scaleX)
            bh = max(12, rh * scaleY)
            fe_rendered_canvas_bbox = [round(bx, 1), round(by, 1), round(bx + bw, 1), round(by + bh, 1)]

            print(f"  Frontend Displayed Frame: {f_target}")
            print(f"  Frontend Selected Trajectory Point Frame: {fe_frame} (Diff: {fe_frame - f_target:+d} frames)")
            print(f"  Frontend Calculated Source BBox (xyxy): {fe_bbox_xyxy}")
            print(f"  Frontend Rendered Canvas BBox (800x450): {fe_rendered_canvas_bbox}")
        else:
            print(f"  Frontend Match: NONE for frame {f_target}")

        if api_match:
            api_bbox = api_match['bbox_xyxy']
            print(f"  API Returned Frame Index: {f_api_data.get('frame_index')}")
            print(f"  API Returned Source BBox (xyxy): {api_bbox}")

            if matched_pt:
                f_diff = fe_frame - f_target
                b_diff = [round(fe_bbox_xyxy[i] - api_bbox[i], 1) for i in range(4)]
                print(f"  --> FRAME DIFFERENCE (fe_frame - requested_frame): {f_diff:+d} frames")
                print(f"  --> BBOX DIFFERENCE (fe_bbox - api_bbox): {b_diff}")
        else:
            print(f"  API Match: NONE returned for Track #{tid} at frame {f_target}")

    # 3. TEST BACKEND ±3 FRAME MATCHING IN API
    print("\n--- 3. TEST BACKEND ±3 FRAME MATCHING IN GET /frames/{frame_index} ---")
    test_frame = 300
    api_res_300 = client.get(f"/api/v1/jobs/{job.id}/frames/{test_frame}")
    print(f"GET /api/v1/jobs/{job.id}/frames/{test_frame} Status: {api_res_300.status_code}")
    d_300 = api_res_300.json() if api_res_300.status_code == 200 else {}
    print(f"Requested Frame: {test_frame}")
    print(f"Returned Frame Field in JSON: {d_300.get('frame_index')}")
    print(f"Returned Timestamp Field in JSON: {d_300.get('timestamp_s')}s")
    for trk_det in d_300.get('detections', []):
        print(f"  - Track #{trk_det['track_id']} ({trk_det['class_name']}): BBox={trk_det['bbox_xyxy']}")

if __name__ == "__main__":
    reconcile()
