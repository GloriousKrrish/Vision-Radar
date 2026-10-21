import sqlite3
import os

conn = sqlite3.connect('data/visionradar.db')
cur = conn.cursor()
cur.execute('SELECT id, filename, storage_path, width, height, duration_sec FROM videos')
rows = cur.fetchall()

exam_vids = []
for r in rows:
    vid_id, fn, sp, w, h, dur = r
    fn_lower = fn.lower()
    if any(k in fn_lower for k in ['exam', 'cheat', 'class', 'student', 'test']):
        exists = os.path.exists(sp)
        exam_vids.append((vid_id, fn, sp, w, h, dur, exists))

print(f"Total videos in DB: {len(rows)}")
print(f"Matching exam/cheat/class videos ({len(exam_vids)}):")
for v in exam_vids:
    print(v)
