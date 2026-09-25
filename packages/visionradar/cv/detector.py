"""
Legacy compatibility wrapper re-exporting Detection, BaseDetector, LightweightDetector, and YOLOXDetector.
"""
from visionradar.cv.detection.base import Detection, BaseDetector
from visionradar.cv.detection.motion import LightweightMotionDetector as LightweightDetector
from visionradar.cv.detection.motion import LightweightMotionDetector
from visionradar.cv.detection.yolox import YOLOXDetector
from visionradar.cv.detection.registry import get_detector

__all__ = [
    "Detection",
    "BaseDetector",
    "LightweightDetector",
    "LightweightMotionDetector",
    "YOLOXDetector",
    "get_detector"
]
