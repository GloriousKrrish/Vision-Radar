import pytest
from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["cv_engine"] == "active"

def test_system_status():
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "total_projects" in data
    assert "total_videos" in data

def test_create_project_and_upload_video():
    # 1. Create project
    res_p = client.post("/api/v1/projects", json={"name": "Test Highway Site", "description": "Benchmark project"})
    assert res_p.status_code == 201
    p_data = res_p.json()
    p_id = p_data["id"]

    # 2. List projects
    res_list = client.get("/api/v1/projects")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1

    # 3. Upload synthetic video file
    with open("data/videos/synthetic_highway.mp4", "rb") as f:
        res_v = client.post(
            f"/api/v1/projects/{p_id}/videos",
            files={"file": ("synthetic_highway.mp4", f, "video/mp4")}
        )
    assert res_v.status_code == 200
    v_data = res_v.json()
    v_id = v_data["id"]

    # 4. Create Calibration
    calib_payload = {
        "image_points": [[330.0, 160.0], [470.0, 160.0], [748.0, 435.0], [51.0, 435.0]],
        "world_points": [[-1.0, 150.0], [13.0, 150.0], [13.0, 0.5], [-1.0, 0.5]],
        "camera_height_m": 9.0,
        "pitch_deg": 14.0
    }
    res_c = client.post(f"/api/v1/videos/{v_id}/calibrations", json=calib_payload)
    assert res_c.status_code == 200
    c_data = res_c.json()

    # 5. Start Job
    res_j = client.post(f"/api/v1/videos/{v_id}/jobs", json={"calibration_id": c_data["id"]})
    assert res_j.status_code == 200
    j_data = res_j.json()
    assert j_data["status"] in ["QUEUED", "RUNNING", "SUCCEEDED"]
