import sqlite3

conn = sqlite3.connect('data/visionradar.db')
cur = conn.cursor()
cur.execute('PRAGMA table_info(examination_tracks)')
cols = [r[1] for r in cur.fetchall()]
print('Existing examination_tracks columns:', cols)

if 'assigned_role' not in cols:
    cur.execute("ALTER TABLE examination_tracks ADD COLUMN assigned_role VARCHAR(32) DEFAULT 'UNKNOWN'")
    print('Added assigned_role column')
if 'role_source' not in cols:
    cur.execute("ALTER TABLE examination_tracks ADD COLUMN role_source VARCHAR(32) DEFAULT 'DEFAULT'")
    print('Added role_source column')

# Also check examination_events table columns
cur.execute('PRAGMA table_info(examination_events)')
event_cols = [r[1] for r in cur.fetchall()]
print('Existing examination_events columns:', event_cols)
if 'assigned_role' not in event_cols:
    cur.execute("ALTER TABLE examination_events ADD COLUMN assigned_role VARCHAR(32) DEFAULT 'UNKNOWN'")
    print('Added assigned_role column to examination_events')

conn.commit()
conn.close()
print("Migration completed successfully.")
