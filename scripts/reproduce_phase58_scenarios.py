import sys
import json
import urllib.request
import urllib.parse
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Video, Track, ExaminationTrackRecord, Violation, ExaminationEventRecord

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

base = "http://localhost:8000/api/v1"

def http_get(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except Exception as e:
        return 0, str(e)

def http_post(url, payload):
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except Exception as e:
        return 0, str(e)

def audit_scenarios():
    print("==================================================================")
    print("PHASE 5.8: DOMAIN ISOLATION & SCENARIO BASELINE AUDIT")
    print("==================================================================")
    db = SessionLocal()

    # Find a sample traffic job and sample examination job
    traffic_jobs = db.query(ProcessingJob).filter_by(domain="traffic", status="SUCCEEDED").order_by(ProcessingJob.id.desc()).all()
    exam_jobs = db.query(ProcessingJob).filter_by(domain="examination", status="SUCCEEDED").order_by(ProcessingJob.id.desc()).all()

    print(f"Total Succeeded Traffic Jobs in DB: {len(traffic_jobs)}")
    if traffic_jobs:
        tj = traffic_jobs[0]
        print(f"  Latest Traffic Job: ID={tj.id}, VideoID={tj.video_id}, Domain={tj.domain}")

    print(f"Total Succeeded Examination Jobs in DB: {len(exam_jobs)}")
    if exam_jobs:
        ej = exam_jobs[0]
        print(f"  Latest Examination Job: ID={ej.id}, VideoID={ej.video_id}, Domain={ej.domain}")

    # SCENARIOS AUDIT
    print("\n--- SCENARIO 3 & 4: CROSS-DOMAIN API QUERY GUARDS ---")
    if traffic_jobs and exam_jobs:
        tj_id = traffic_jobs[0].id
        ej_id = exam_jobs[0].id

        # 1. Query examination endpoints on a Traffic Job
        status, res = http_get(f"{base}/jobs/{tj_id}/examination/results")
        print(f"Query /jobs/{tj_id} (Traffic) -> /examination/results: Status {status}")
        print(f"  Response: {res}")

        status, res = http_get(f"{base}/jobs/{tj_id}/examination/tracks")
        print(f"Query /jobs/{tj_id} (Traffic) -> /examination/tracks: Status {status}")
        print(f"  Response: {res}")

        # 2. Query traffic endpoints on an Examination Job
        status, res = http_get(f"{base}/jobs/{ej_id}/tracks")
        print(f"Query /jobs/{ej_id} (Exam) -> /tracks (Traffic): Status {status}")
        print(f"  Response: {res}")

        status, res = http_get(f"{base}/violations?job_id={ej_id}")
        print(f"Query /violations?job_id={ej_id} (Exam): Status {status}")
        print(f"  Response: {res}")

        # 3. Query reports across domains
        status, res = http_get(f"{base}/jobs/{tj_id}/examination/audit-report/html")
        print(f"Query /jobs/{tj_id} (Traffic) -> /examination/audit-report/html: Status {status}")
        print(f"  Response snippet: {str(res)[:120]}")

        # 4. Excel export domain resolution
        status, res = http_get(f"{base}/jobs/{tj_id}/export/excel/preview")
        print(f"Query /jobs/{tj_id} (Traffic) -> /export/excel/preview: Status {status}")
        if isinstance(res, dict):
            print(f"  Preview Domain: {res.get('analysis_domain')}")

        status, res = http_get(f"{base}/jobs/{ej_id}/export/excel/preview")
        print(f"Query /jobs/{ej_id} (Exam) -> /export/excel/preview: Status {status}")
        if isinstance(res, dict):
            print(f"  Preview Domain: {res.get('analysis_domain')}")

    print("\n--- SCENARIO 8 & 9: JOBS LIST DOMAIN FILTERING ---")
    status, all_jobs = http_get(f"{base}/jobs?status=SUCCEEDED")
    print(f"Query /jobs?status=SUCCEEDED -> Status {status}, Count={len(all_jobs) if isinstance(all_jobs, list) else 0}")
    if isinstance(all_jobs, list) and len(all_jobs) > 0:
        first_j = all_jobs[0]
        print(f"  First returned job: ID={first_j.get('id')}, Domain={first_j.get('domain')}")

    db.close()

if __name__ == "__main__":
    audit_scenarios()
