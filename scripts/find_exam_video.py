import sqlite3

conn = sqlite3.connect('data/visionradar.db')
cur = conn.cursor()
cur.execute("SELECT id, filename, storage_path, duration_sec, fps, width, height FROM videos WHERE LOWER(filename) LIKE '%exam%' OR LOWER(filename) LIKE '%cheat%' ORDER BY id DESC")
rows = cur.fetchall()
print(f"Found {len(rows)} videos matching exam/cheat:")
for r in rows:
    print(r)

cur.execute("SELECT id, video_id, status, config_json, stage FROM processing_jobs ORDER BY id DESC LIMIT 10")
jobs = cur.fetchall()
print("\nRecent jobs:")
for j in jobs:
    print(j)
