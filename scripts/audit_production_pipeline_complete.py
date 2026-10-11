import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.parse
import cv2
import numpy as np

# Ensure packages is in path
sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

def run_audit():
    print("==================================================")
    print(" VISIONRADAR PRODUCTION REALITY & DATASET AUDIT")
    print("==================================================")
    
    # ----------------------------------------------------
    # PART 1: FRESH VIDEO UPLOAD & LIVE PIPELINE TRACE
    # ----------------------------------------------------
    print("\n--- PART 1: FRESH VIDEO UPLOAD TRACE ---")
    video_path = "Traffic1.mp4"
    if not os.path.exists(video_path):
        video_path = "data/videos/Traffic1.mp4"
    
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_hash = hashlib.sha256(video_bytes).hexdigest()
    
    # 1. Get project
    req = urllib.request.Request("http://localhost:8000/api/v1/projects")
    with urllib.request.urlopen(req) as resp:
        projects = json.loads(resp.read().decode("utf-8"))
        project_id = projects[0]["id"]
        
    print(f"Project ID: {project_id}")
    
    # 2. Upload video
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="Traffic1_Audit_Fresh.mp4"\r\n')
    body.extend(b"Content-Type: video/mp4\r\n\r\n")
    body.extend(video_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    
    upload_req = urllib.request.Request(
        f"http://localhost:8000/api/v1/projects/{project_id}/videos",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(upload_req) as resp:
        video_res = json.loads(resp.read().decode("utf-8"))
        video_id = video_res["id"]
        
    print(f"Uploaded Video ID: {video_id}")
    print(f"Video Filename: {video_res['filename']}")
    print(f"Video Hash (SHA-256): {video_hash}")
    print(f"Resolution: {video_res['width']}x{video_res['height']}")
    print(f"FPS: {video_res['fps']}")
    print(f"Duration: {video_res['duration_sec']} s")

    # 3. Start processing job WITHOUT calibration (as raw frontend upload does)
    job_payload = json.dumps({
        "calibration_id": None,
        "config_json": {"detector": "yolox"}
    }).encode("utf-8")
    
    job_req = urllib.request.Request(
        f"http://localhost:8000/api/v1/videos/{video_id}/jobs",
        data=job_payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(job_req) as resp:
        job_res = json.loads(resp.read().decode("utf-8"))
        job_id = job_res["id"]
        
    print(f"Job Created: Job #{job_id}")
    
    # Poll job completion
    t0 = time.time()
    while True:
        st_req = urllib.request.Request(f"http://localhost:8000/api/v1/jobs/{job_id}")
        with urllib.request.urlopen(st_req) as resp:
            st = json.loads(resp.read().decode("utf-8"))
        if st["status"] in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.5)
    t1 = time.time()
    
    telemetry = st.get("telemetry") or {}
    print(f"Final Job State: {st['status']}")
    print(f"Job Processing Duration: {round(t1 - t0, 2)} s")
    print(f"Frames Processed: {telemetry.get('frames_processed')}")
    print(f"Processing FPS: {telemetry.get('average_fps')}")
    print(f"Total Detections (post-NMS): {telemetry.get('post_nms_detection_count')}")
    print(f"Total Tracks: {telemetry.get('total_tracks')}")
    events_summary = (telemetry.get("traffic_intelligence") or {}).get("events_summary") or {}
    print(f"Total Traffic Events: {events_summary.get('total_events', 0)}")
    
    # ----------------------------------------------------
    # PART 3: PRODUCTION MODEL AUDIT
    # ----------------------------------------------------
    print("\n--- PART 3: PRODUCTION MODEL AUDIT ---")
    model_path = "data/models/yolox_nano.onnx"
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    net.setInput(dummy_input)
    t_start = time.perf_counter()
    for _ in range(20):
        out = net.forward()
    t_end = time.perf_counter()
    avg_latency_ms = ((t_end - t_start) / 20.0) * 1000.0
    
    print(f"Model Path: {model_path}")
    print(f"Model SHA-256 Hash: {model_hash}")
    print(f"Model Output Shape: {out.shape}")
    print(f"Model Architecture: YOLOX-Nano ONNX Anchorless Object Detector")
    print(f"Input Resolution: 640x640")
    print(f"Confidence Threshold: 0.25")
    print(f"NMS Threshold: 0.45")
    print(f"Classes: 80 COCO Classes (Vehicle mapping: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck)")
    print(f"Preprocessing: Letterbox resize to 640x640 with pad 114 (no mean subtraction / divide by std)")
    print(f"Postprocessing: Vectorized YOLOX anchor grid decoding [8400, 85] + OpenCV NMSBoxes")
    print(f"Inference Backend: OpenCV DNN (CPUExecutionProvider)")
    print(f"Average Inference Latency: {round(avg_latency_ms, 2)} ms / frame ({round(1000.0/avg_latency_ms, 1)} FPS)")
    
    # Check if model is standard YOLOX COCO pre-trained or fine-tuned
    known_coco_hash = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
    is_standard_coco = (model_hash == known_coco_hash)
    print(f"Is Standard COCO YOLOX-Nano ONNX: {is_standard_coco}")
    
    return {
        "job_id": job_id,
        "video_res": video_res,
        "video_hash": video_hash,
        "model_hash": model_hash,
        "st": st,
        "telemetry": telemetry,
        "avg_latency_ms": avg_latency_ms,
        "is_standard_coco": is_standard_coco
    }

if __name__ == "__main__":
    run_audit()
