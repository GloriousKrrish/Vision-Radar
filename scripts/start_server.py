import os
import sys
import subprocess
import uvicorn

# Ensure packages/ and root are in PYTHONPATH
root_dir = os.path.abspath(".")
packages_dir = os.path.abspath("packages")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if packages_dir not in sys.path:
    sys.path.insert(0, packages_dir)

existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{root_dir};{packages_dir};{existing_pythonpath}"

def main():
    print("==================================================")
    print("   VISIONRADAR — TRAFFIC INTELLIGENCE PLATFORM   ")
    print("==================================================")
    print("Starting FastAPI Backend Server on http://localhost:8000...")
    print("Open API Docs available at: http://localhost:8000/docs")

    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()

