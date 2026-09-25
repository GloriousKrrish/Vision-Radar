import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from apps.api.main import app

client = TestClient(app)

def test_api():
    print("==================================================")
    print("VISIONRADAR — PHASE 3.2.3 FRAME API FORENSIC TEST")
    print("==================================================")
    res = client.get("/api/v1/jobs/1/frames/300")
    print(f"HTTP Status: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print("Response JSON:")
        print(f"Job ID: {data.get('job_id')}")
        print(f"Frame Index: {data.get('frame_index')}")
        print(f"Timestamp: {data.get('timestamp_s')}s")
        print(f"Source Resolution: {data.get('source_width')}x{data.get('source_height')}")
        print(f"Detections Count: {len(data.get('detections', []))}")
        for d in data.get('detections', []):
            print(f"  - Track #{d.get('track_id')} | Class: {d.get('class_name')} | Conf: {d.get('confidence')} | BBox: {d.get('bbox_xyxy')}")
    else:
        print(f"Response Error: {res.text}")

if __name__ == "__main__":
    test_api()
