import sys, os, glob
sys.path.insert(0, 'packages')
import requests, json, time

BASE = 'http://localhost:8000/api/v1'

# Get project
r = requests.get(f'{BASE}/projects')
proj_id = r.json()[0]['id']
print('Project ID:', proj_id)

# Upload Traffic1.mp4
print('Uploading Traffic1.mp4...')
with open('Traffic1.mp4', 'rb') as f:
    upl = requests.post(f'{BASE}/projects/{proj_id}/videos',
                        files={'file': ('Traffic1.mp4', f, 'video/mp4')}, timeout=120)
vid_id = upl.json()['id']
print('Uploaded video ID:', vid_id, '| Shape from DB:', upl.json())

# Create job
job_r = requests.post(f'{BASE}/videos/{vid_id}/jobs', json={'calibration_id': None})
job_id = job_r.json()['id']
print('Job ID:', job_id)

# Poll until complete
for i in range(200):
    time.sleep(3)
    jr = requests.get(f'{BASE}/jobs/{job_id}').json()
    status = jr.get('status')
    stage = jr.get('stage')
    pct = jr.get('progress_pct', 0)
    print(f'  [{i*3:3d}s] {status} | {stage} | {pct:.0f}%')
    if status in ('SUCCEEDED', 'FAILED'):
        t = jr.get('telemetry') or {}
        print('=== TELEMETRY ===')
        print(json.dumps({
            'total_tracks': t.get('total_tracks'),
            'vehicle_detection_count': t.get('vehicle_detection_count'),
            'post_nms_detection_count': t.get('post_nms_detection_count'),
            'frames_processed': t.get('frames_processed'),
            'average_fps': t.get('average_fps'),
            'video_resolution': t.get('video_resolution'),
            'video_fps': t.get('video_fps'),
        }, indent=2))
        if status == 'FAILED':
            err = jr.get('error_message', '')
            print('ERROR:', err[:800])
        break
