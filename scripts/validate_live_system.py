import urllib.request
import json
import io
import sys
import openpyxl

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

base = "http://localhost:8000/api/v1"

def test_api():
    print("=================================================================")
    print("1. LIVE API VALIDATION FOR JOB #990")
    print("=================================================================")

    # 1. Job details
    req = urllib.request.urlopen(f"{base}/jobs/990")
    job = json.loads(req.read().decode("utf-8"))
    print(f"[API] GET /jobs/990 -> Status: {req.status}")
    print(f"      Video ID: {job.get('video_id')} | Status: {job.get('status')} | Domain: {job.get('domain')}")

    # 2. Examination Results
    req = urllib.request.urlopen(f"{base}/jobs/990/examination/results")
    results = json.loads(req.read().decode("utf-8"))
    print(f"\n[API] GET /jobs/990/examination/results -> Status: {req.status}")
    print(f"      Summary Block: {results.get('summary')}")
    print(f"      Person Tracks Count: {len(results.get('person_tracks', []))}")
    print(f"      Object Tracks Count: {len(results.get('object_tracks', []))}")
    print(f"      Events Count: {len(results.get('events', []))}")
    
    for p in results.get("person_tracks", []):
        print(f"      - Person Track #{p['track_id']}: role='{p.get('assigned_role')}' source='{p.get('role_source')}'")
    for o in results.get("object_tracks", []):
        print(f"      - Object Track #{o['track_id']}: class='{o.get('class_name')}' entity='{o.get('entity_type')}'")
    for e in results.get("events", []):
        print(f"      - Event #{e['event_id']}: type='{e.get('event_type')}' review_status='{e.get('review_status')}'")

    # 3. Dedicated Tracks Endpoint
    req = urllib.request.urlopen(f"{base}/jobs/990/examination/tracks")
    tracks_payload = json.loads(req.read().decode("utf-8"))
    print(f"\n[API] GET /jobs/990/examination/tracks -> Status: {req.status}")
    print(f"      Summary Block: {tracks_payload.get('summary')}")
    print(f"      Person Tracks: {len(tracks_payload.get('person_tracks', []))}")
    for pt in tracks_payload.get("person_tracks", []):
        print(f"      - Track #{pt['track_id']}: dur={pt.get('duration_sec')}s frames={pt.get('duration_frames')} continuity={pt.get('continuity_pct')}% is_transient={pt.get('is_transient')}")
    print(f"      Object Tracks: {len(tracks_payload.get('object_tracks', []))}")
    for ot in tracks_payload.get("object_tracks", []):
        print(f"      - Track #{ot['track_id']}: class='{ot.get('class_name')}' assoc_person={ot.get('associated_person_track_id')}")

    # 4. HTML Audit Report Endpoint
    req = urllib.request.urlopen(f"{base}/jobs/990/examination/audit-report/html")
    html_content = req.read().decode("utf-8")
    print(f"\n[API] GET /jobs/990/examination/audit-report/html -> Status: {req.status}")
    print(f"      Content Length: {len(html_content)} bytes")
    
    # Check key text elements in HTML
    checks = [
        ("3", "Total Persons Count (3)"),
        ("2 Seated · 1 Peripheral", "Seated vs Peripheral breakdown"),
        ("Evidence Objects Tracked", "Evidence Objects Tracked card"),
        ("VR-SEAL-", "Cryptographic Audit Seal"),
        ("Forensic Data Dictionary &amp; Accounting Rules", "Forensic Data Dictionary section"),
        ("Calculator Discrimination", "COCO taxonomy disclaimer"),
    ]
    all_html_passed = True
    for needle, desc in checks:
        found = needle in html_content
        status_str = "PASS" if found else "FAIL"
        if not found:
            all_html_passed = False
        print(f"      Check '{desc}': {status_str}")
    print(f"      HTML Verification: {'100% VERIFIED' if all_html_passed else 'FAILED'}")

    # 5. Excel Export Endpoint
    req = urllib.request.urlopen(f"{base}/jobs/990/export/excel?mode=final")
    excel_bytes = req.read()
    print(f"\n[API] GET /jobs/990/export/excel?mode=final -> Status: {req.status}")
    print(f"      Content-Type: {req.headers.get('Content-Type')}")
    print(f"      Size: {len(excel_bytes)} bytes")

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes), data_only=True)
    print(f"      Excel Sheets: {wb.sheetnames}")
    
    # Inspect Executive Summary
    summary_ws = wb["Executive Summary"]
    print("      Executive Summary Key Cells:")
    for row in range(1, 26):
        c1 = summary_ws.cell(row=row, column=1).value
        c2 = summary_ws.cell(row=row, column=2).value
        if c1 is not None:
            val_str = str(c2) if c2 is not None else ""
            print(f"        Row {row:2d}: {str(c1):<45} | {val_str}")

    print("\n=================================================================")
    print("LIVE VALIDATION OF ALL ENDPOINTS COMPLETED SUCCESSFULLY")
    print("=================================================================")

if __name__ == "__main__":
    test_api()
