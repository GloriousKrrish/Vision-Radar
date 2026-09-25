import os
import sys
import json
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from apps.api.main import app
from visionradar.models.database import get_db
from visionradar.models.entities import ProcessingJob, Video, Track

client = TestClient(app)

def run_reconciliation():
    db = next(get_db())
    job = db.query(ProcessingJob).filter(ProcessingJob.id == 64).first()
    if not job:
        job = db.query(ProcessingJob).filter(ProcessingJob.status == "SUCCEEDED").order_by(ProcessingJob.id.desc()).first()
    
    if not job:
        print("ERROR: No suitable job found")
        return

    video = db.query(Video).filter(Video.id == job.video_id).first()
    tracks = db.query(Track).filter(Track.job_id == job.id).all()

    # Index track trajectory points strictly by frame_index (Matching new frontend WorkbenchView logic)
    frame_map = {} # frame_index -> list of detections
    for trk in tracks:
        traj = trk.trajectory_json or []
        for pt in traj:
            f = pt.get("frame_index")
            if f is not None:
                if f not in frame_map:
                    frame_map[f] = []
                raw_bbox = pt.get("bbox")
                if raw_bbox and len(raw_bbox) >= 4:
                    rx, ry, rw, rh = raw_bbox
                    if rw > rx:
                        rw = rw - rx
                    if rh > ry:
                        rh = rh - ry
                    bbox_xyxy = [round(rx, 1), round(ry, 1), round(rx + rw, 1), round(ry + rh, 1)]
                    frame_map[f].append({
                        "track_id": trk.track_id,
                        "class_name": trk.vehicle_class,
                        "bbox_xyxy": bbox_xyxy
                    })

    eval_frames = [120, 150, 194, 200, 201, 202]

    print("=========================================================================================")
    print("PHASE 3.2.3A EXACT FRAME RENDERING RECONCILIATION TABLE")
    print(f"Job ID: {job.id} | Video: {video.filename if video else 'Traffic1.mp4'}")
    print("=========================================================================================")
    print(f"| {'Req Frame':<9} | {'Render Frame':<12} | {'Track ID':<8} | {'Frontend BBox (xyxy)':<24} | {'Canonical API BBox':<24} | {'Difference':<15} |")
    print("|-----------|--------------|----------|--------------------------|--------------------------|-----------------|")

    markdown_rows = []

    for f_req in eval_frames:
        # Get canonical API response
        res = client.get(f"/api/v1/jobs/{job.id}/frames/{f_req}")
        api_data = res.json() if res.status_code == 200 else {}
        api_dets = {d['track_id']: d['bbox_xyxy'] for d in api_data.get('detections', [])}

        fe_dets = frame_map.get(f_req, [])
        if not fe_dets and not api_dets:
            row_str = f"| {f_req:<9} | {f_req:<12} | {'N/A':<8} | {'[No Detections]':<24} | {'[No Detections]':<24} | {'0.0':<15} |"
            print(row_str)
            markdown_rows.append(row_str)
            continue

        all_tids = sorted(list(set(list(api_dets.keys()) + [d['track_id'] for d in fe_dets])))
        for tid in all_tids:
            fe_match = next((d for d in fe_dets if d['track_id'] == tid), None)
            api_bbox = api_dets.get(tid)

            if fe_match:
                fe_bbox = fe_match['bbox_xyxy']
                f_render = str(f_req)
            else:
                fe_bbox = None
                f_render = "NONE"

            if fe_bbox and api_bbox:
                diff = [round(fe_bbox[i] - api_bbox[i], 1) for i in range(4)]
                diff_str = str(diff)
            elif not fe_bbox and not api_bbox:
                diff_str = "0.0"
            else:
                diff_str = "MISMATCH"

            fe_bbox_str = str(fe_bbox) if fe_bbox else "NONE"
            api_bbox_str = str(api_bbox) if api_bbox else "NONE"

            row_str = f"| {f_req:<9} | {f_render:<12} | {tid:<8} | {fe_bbox_str:<24} | {api_bbox_str:<24} | {diff_str:<15} |"
            print(row_str)
            markdown_rows.append(row_str)

    print("=========================================================================================")

if __name__ == "__main__":
    run_reconciliation()
