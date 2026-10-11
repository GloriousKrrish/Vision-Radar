"""
VisionRadar Stream Router
Defines streaming/WebSocket endpoints for job progress and live inference.
"""

from fastapi import APIRouter

router = APIRouter(prefix="", tags=["Streaming"])
