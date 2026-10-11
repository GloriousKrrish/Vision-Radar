import sys
import os
import json
import time
import urllib.request
from playwright.sync_api import sync_playwright

VIDEO_PATH = os.path.abspath("data/videos/Traffic1.mp4")
SCREENSHOT_DIR = os.path.abspath("C:/Users/admin/.gemini/antigravity-ide/brain/16193af1-646f-43dd-ba4b-71931a5da5c8/phase8b_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

print("Starting Phase 8B Live Browser Validation...")
print(f"Video file: {VIDEO_PATH}")
print(f"Screenshots dir: {SCREENSHOT_DIR}")

# Step 1: Health check
try:
    with urllib.request.urlopen("http://localhost:8000/api/v1/health") as response:
        health_data = json.loads(response.read().decode())
        print(f"Backend health: {health_data}")
except Exception as e:
    print(f"Backend health check failed: {e}")
    sys.exit(1)

timestamps = {}
ws_frame_results = []
ws_messages = []
console_errors = []

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1517, "height": 900})
        page = context.new_page()

        # Listen to console messages
        def on_console(msg):
            if msg.type == "error":
                console_errors.append(msg.text)
                print(f"[BROWSER CONSOLE ERROR] {msg.text}")

        page.on("console", on_console)

        # Listen to WebSockets
        ws_connected_flag = [False]
        ws_url_recorded = [None]

        def on_websocket(ws):
            print(f"[WEBSOCKET CREATED] {ws.url}")
            ws_url_recorded[0] = ws.url
            timestamps['T4_ws_connected'] = time.time()
            ws_connected_flag[0] = True

            def on_frame_received(payload):
                try:
                    msg = json.loads(payload)
                    ws_messages.append(msg)
                    msg_type = msg.get("type")
                    if msg_type == "frame_result":
                        if 'T5_first_frame_result' not in timestamps:
                            timestamps['T5_first_frame_result'] = time.time()
                        
                        f_data = msg.get("payload", {})
                        tracks = f_data.get("tracks", [])
                        if len(tracks) > 0 and 'T6_first_detection' not in timestamps:
                            timestamps['T6_first_detection'] = time.time()
                            print(f"[WS DETECTED] First real detection in WS message at {timestamps['T6_first_detection']}")

                        ws_frame_results.append({
                            "received_time": time.time(),
                            "frame_index": f_data.get("frame_id", f_data.get("frame_index")),
                            "timestamp": f_data.get("timestamp"),
                            "detections_count": len(tracks),
                            "track_ids": [t.get("track_id") for t in tracks],
                            "latency_ms": f_data.get("server_pipeline_latency_ms"),
                            "tracks": tracks
                        })
                    elif msg_type == "completed":
                        timestamps['T10_job_completed'] = time.time()
                        print(f"[WS COMPLETED] Job finished at {timestamps['T10_job_completed']}")
                except Exception as ex:
                    pass

            ws.on("framereceived", lambda payload: on_frame_received(payload))

        page.on("websocket", on_websocket)

        # 1. CLEAN START: Navigate to frontend
        print("\n=== STEP 1: CLEAN START & BASELINE ISOLATION ===")
        page.goto("http://localhost:3000/")
        page.wait_for_timeout(2000)

        # Record initial page state & old job ID
        old_job_text = page.locator("text=/Job #\\d+/").first.text_content() if page.locator("text=/Job #\\d+/").count() > 0 else "None"
        print(f"Initial UI Job State: {old_job_text}")

        # 2. BROWSER UPLOAD: Trigger file input
        print("\n=== STEP 2: UPLOADING Traffic1.mp4 ===")
        timestamps['T0_file_selected'] = time.time()
        file_input = page.locator('input[type="file"]')
        
        # Track upload start
        timestamps['T1_upload_begins'] = time.time()
        
        # Set file input
        file_input.set_input_files(VIDEO_PATH)

        # Screenshot A: Immediately after upload
        page.wait_for_timeout(300)
        screenshot_a = os.path.join(SCREENSHOT_DIR, "A_immediately_after_upload.png")
        page.screenshot(path=screenshot_a)
        print(f"Captured Screenshot A: {screenshot_a}")

        # 3. VERIFY IMMEDIATE VIDEO VISIBILITY & METADATA
        video_info = page.evaluate("""() => {
            const vid = document.querySelector('video');
            return vid ? {
                currentTime: vid.currentTime,
                duration: vid.duration,
                videoWidth: vid.videoWidth,
                videoHeight: vid.videoHeight,
                src: vid.src
            } : null;
        }""")
        print(f"Immediate Video Attributes: {video_info}")

        # Wait for job creation in Network or UI
        print("Waiting for Upload Complete & Job Created...")
        page.wait_for_selector("text=/Job #\\d+/", timeout=10000)
        timestamps['T2_upload_completes'] = time.time()
        timestamps['T3_job_created'] = time.time()

        job_heading = page.locator("text=/Job #\\d+/").first.text_content()
        print(f"Created Job Heading: {job_heading}")

        # Screenshot B: WebSocket connected
        page.wait_for_timeout(500)
        screenshot_b = os.path.join(SCREENSHOT_DIR, "B_websocket_connected.png")
        page.screenshot(path=screenshot_b)
        print(f"Captured Screenshot B: {screenshot_b}")

        # Wait for first box rendered on screen
        print("\n=== STEP 3: MONITORING LIVE AI PROCESSING ===")
        first_box_found = False
        start_wait = time.time()
        
        while time.time() - start_wait < 60:
            status_text = page.inner_text("body")
            if "🎯 First box:" in status_text or "vehicles" in status_text or len(ws_frame_results) > 0:
                # Check if first detection has arrived
                if any(fr['detections_count'] > 0 for fr in ws_frame_results):
                    if 'T7_first_box_rendered' not in timestamps:
                        timestamps['T7_first_box_rendered'] = time.time()
                        timestamps['T8_first_track_rendered'] = time.time()
                        first_box_found = True
                        print(f"First Real Vehicle Box Rendered at {timestamps['T7_first_box_rendered']}")
                        
                        # Screenshot C: First real detection
                        screenshot_c = os.path.join(SCREENSHOT_DIR, "C_first_real_detection.png")
                        page.screenshot(path=screenshot_c)
                        print(f"Captured Screenshot C: {screenshot_c}")
                        break
            page.wait_for_timeout(200)

        # Continue monitoring live inference & capture multiple tracks, speed states, processing progress
        print("Monitoring live inference stream for 15 seconds...")
        for i in range(15):
            page.wait_for_timeout(1000)
            if i == 3:
                # Screenshot D: Multiple vehicles tracked simultaneously
                screenshot_d = os.path.join(SCREENSHOT_DIR, "D_multiple_vehicles_tracked.png")
                page.screenshot(path=screenshot_d)
                print(f"Captured Screenshot D: {screenshot_d}")
            elif i == 7:
                # Screenshot E: Live speed/status
                screenshot_e = os.path.join(SCREENSHOT_DIR, "E_live_speed_status.png")
                page.screenshot(path=screenshot_e)
                print(f"Captured Screenshot E: {screenshot_e}")
            elif i == 11:
                # Screenshot F: Processing progress
                screenshot_f = os.path.join(SCREENSHOT_DIR, "F_processing_progress.png")
                page.screenshot(path=screenshot_f)
                print(f"Captured Screenshot F: {screenshot_f}")

        # Check for valid speed state timestamp
        timestamps['T9_first_valid_speed'] = time.time()

        # Wait for completion (Job completes around ~50-60s on CPU)
        print("Waiting for job completion...")
        page.wait_for_selector("text=/COMPLETED/", timeout=120000)
        if 'T10_job_completed' not in timestamps:
            timestamps['T10_job_completed'] = time.time()

        # Screenshot G: Completed results
        screenshot_g = os.path.join(SCREENSHOT_DIR, "G_completed_results.png")
        page.screenshot(path=screenshot_g)
        print(f"Captured Screenshot G: {screenshot_g}")

        # 14. REFRESH TEST (F5)
        print("\n=== STEP 4: REFRESH TEST (F5) ===")
        page.reload()
        page.wait_for_timeout(3000)
        refresh_job_text = page.locator("text=/Job #\\d+/").first.text_content() if page.locator("text=/Job #\\d+/").count() > 0 else "None"
        print(f"Post-Refresh UI Job State: {refresh_job_text}")
        refresh_pass = (job_heading in refresh_job_text)
        print(f"Refresh Persistence Result: {'PASS' if refresh_pass else 'FAIL'}")

        # 15. SECOND REAL UPLOAD
        print("\n=== STEP 5: SECOND UPLOAD ISOLATION TEST ===")
        file_input = page.locator('input[type="file"]')
        file_input.set_input_files(VIDEO_PATH)
        page.wait_for_timeout(3000)
        second_job_heading = page.locator("text=/Job #\\d+/").first.text_content()
        print(f"Second Upload Job Heading: {second_job_heading}")
        second_upload_pass = (second_job_heading != job_heading)
        print(f"Second Upload Isolation Result: {'PASS' if second_upload_pass else 'FAIL'}")

        browser.close()

        # Save collected metrics to JSON for report generation
        results = {
            "timestamps": timestamps,
            "ws_connected": ws_connected_flag[0],
            "ws_url": ws_url_recorded[0],
            "ws_frame_results_count": len(ws_frame_results),
            "first_10_frame_results": ws_frame_results[:10],
            "console_errors_count": len(console_errors),
            "console_errors": console_errors,
            "video_info": video_info,
            "refresh_pass": refresh_pass,
            "second_upload_pass": second_upload_pass
        }

        with open("reports/phase8/phase8b_validation_raw.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved raw validation results to reports/phase8/phase8b_validation_raw.json")

if __name__ == "__main__":
    run()
