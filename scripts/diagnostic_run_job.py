import asyncio
import json
import time
import websockets
import urllib.request
import os

SERVER_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000"

async def test_job_completion():
    sample_video_path = os.path.abspath("data/videos/synthetic_highway.mp4")
    print(f"[TEST] Uploading '{sample_video_path}'...")
    
    with open(sample_video_path, 'rb') as f:
        file_bytes = f.read()

    boundary = '----WebKitFormBoundaryDiagnosticTest123'
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="diagnostic_test.mp4"\r\n'
        f'Content-Type: video/mp4\r\n\r\n'
    ).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    upload_req = urllib.request.Request(
        f"{SERVER_URL}/api/v1/projects/1/videos",
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )

    t0 = time.time()
    with urllib.request.urlopen(upload_req) as resp:
        vid_data = json.loads(resp.read().decode('utf-8'))
        video_id = vid_data['id']
        print(f"[TEST] Uploaded Video ID: {video_id} ({vid_data['width']}x{vid_data['height']} @ {vid_data['fps']} FPS)")

    # Create Job
    job_req = urllib.request.Request(
        f"{SERVER_URL}/api/v1/videos/{video_id}/jobs",
        data=json.dumps({"calibration_id": 1}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(job_req) as resp:
        job_data = json.loads(resp.read().decode('utf-8'))
        job_id = job_data['id']
        print(f"[TEST] Created Job ID: #{job_id}")

    # Connect WebSocket
    ws_endpoint = f"{WS_URL}/api/v1/jobs/{job_id}/stream"
    print(f"[TEST] Connecting to WebSocket {ws_endpoint}...")

    frames_received = 0
    statuses_received = []
    completed_received = False
    error_received = None

    try:
        async with websockets.connect(ws_endpoint) as ws:
            while True:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                except asyncio.TimeoutError:
                    print(f"[TEST WARNING] 60-second timeout waiting on WebSocket! Checking DB...")
                    req = urllib.request.urlopen(f"{SERVER_URL}/api/v1/jobs/{job_id}")
                    j_status = json.loads(req.read().decode('utf-8'))
                    print(f"[TEST DB STATUS] status={j_status.get('status')}, stage={j_status.get('stage')}, progress={j_status.get('progress_pct')}")
                    break

                msg = json.loads(raw_msg)
                mtype = msg.get("type")
                if mtype == "frame_result":
                    frames_received += 1
                    if frames_received % 50 == 0 or frames_received == 1:
                        print(f"[WS] Frame #{msg.get('frame_index')} received (total: {frames_received})")
                elif mtype == "status":
                    statuses_received.append(msg)
                    print(f"[WS STATUS] stage={msg.get('stage')}, progress={msg.get('progress')}%, msg={msg.get('message')}")
                elif mtype == "completed":
                    completed_received = True
                    print(f"[WS COMPLETED] Received 'completed' message! Tracks: {msg.get('total_tracks')}, RTF: {msg.get('real_time_factor')}")
                    break
                elif mtype == "error":
                    error_received = msg.get('message')
                    print(f"[WS ERROR] Received error: {error_received}")
                    break
                elif mtype == "ping":
                    print(f"[WS PING] Keepalive ping received")
                else:
                    print(f"[WS OTHER] {mtype}: {msg}")
    except Exception as e:
        print(f"[TEST WS EXCEPTION] {e}")

    t_total = time.time() - t0
    print(f"[TEST FINISHED in {t_total:.2f}s]")
    print(f"  Frames: {frames_received}")
    print(f"  Completed received: {completed_received}")
    print(f"  Error: {error_received}")

    # Check final DB state
    req = urllib.request.urlopen(f"{SERVER_URL}/api/v1/jobs/{job_id}")
    j_final = json.loads(req.read().decode('utf-8'))
    print(f"  Final DB State: status={j_final.get('status')}, stage={j_final.get('stage')}, progress={j_final.get('progress_pct')}, completed_at={j_final.get('completed_at')}")

if __name__ == "__main__":
    asyncio.run(test_job_completion())
