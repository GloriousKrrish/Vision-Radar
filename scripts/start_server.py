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

import argparse

def main():
    parser = argparse.ArgumentParser(description="VisionRadar Backend Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print("==================================================")
    print("   VISIONRADAR — TRAFFIC INTELLIGENCE PLATFORM   ")
    print("==================================================")
    print(f"Starting FastAPI Backend Server on http://localhost:{args.port}...")
    print(f"Open API Docs available at: http://localhost:{args.port}/docs")

    uvicorn.run("apps.api.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()

