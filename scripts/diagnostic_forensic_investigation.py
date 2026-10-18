import sqlite3
import json
import os
import hashlib
import cv2

def main():
    conn = sqlite3.connect('data/visionradar.db')
    c = conn.cursor()

    print("=== NON-TRAFFIC1 JOBS IN DATABASE ===")
    query = """
    SELECT j.id, j.video_id, v.filename, v.storage_path, v.width, v.height, v.fps, j.calibration_id, j.status, j.error_message, j.telemetry_json
    FROM processing_jobs j
    JOIN videos v ON j.video_id = v.id
    WHERE v.filename NOT IN ('Traffic1.mp4')
    ORDER BY j.id DESC
    LIMIT 25
    """
    rows = c.execute(query).fetchall()
    for r in rows:
        jid, vid, fname, spath, w, h, fps, cal_id, status, err, tel = r
        tel_data = json.loads(tel) if tel else {}
        frames = tel_data.get('frames_processed', 0)
        tracks = tel_data.get('total_tracks', 0)
        err_snippet = (err[:80] + '...') if err else 'None'
        print(f"Job #{jid:3d} | Vid #{vid:3d} ({fname:20s} {w}x{h}@{fps:.1f}fps) | Calib={str(cal_id):5s} | Status={status:9s} | Frames={frames:3d} | Tracks={tracks:2d} | Err={err_snippet}")

    print("\n=== VIDEO FILE INSPECTION ===")
    test_files = [
        "Traffic1.mp4",
        "data/videos/synthetic_highway.mp4",
        "DATASETS/Vehicle_Tracking/AICC22-Custom/AICC22-Custom_Original-video.mp4",
        "DATASETS/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4"
    ]
    for p in test_files:
        if not os.path.exists(p):
            print(f"File NOT FOUND: {p}")
            continue
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        cap = cv2.VideoCapture(p)
        opened = cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        h_dim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) if opened else 0
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]) if opened else ""
        cap.release()
        print(f"Path: {p:55s} | Size: {size:8d} | {w}x{h_dim} @ {fps:.2f}fps ({fc} frames) | FourCC: {fourcc_str} | SHA256: {h[:16]}...")

if __name__ == '__main__':
    main()
