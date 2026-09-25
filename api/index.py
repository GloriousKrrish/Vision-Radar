import os
import sys

# Ensure root directory and packages directory are in Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "packages"))

from apps.api.main import app

# For Vercel Serverless Function entrypoint
