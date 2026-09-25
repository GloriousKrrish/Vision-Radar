from typing import Dict, Type, Any
from visionradar.cv.detection.base import BaseDetector
from visionradar.cv.detection.motion import LightweightMotionDetector
from visionradar.cv.detection.yolox import YOLOXDetector

DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
    "mog2": LightweightMotionDetector,
    "motion": LightweightMotionDetector,
    "lightweight": LightweightMotionDetector,
    "yolox": YOLOXDetector,
    "yolox-nano": YOLOXDetector,
    "yolo": YOLOXDetector
}

def get_detector(name: str = "yolox", **kwargs) -> BaseDetector:
    """
    Factory function to instantiate object detectors from registry by name.
    """
    key = name.lower().strip()
    detector_cls = DETECTOR_REGISTRY.get(key, YOLOXDetector)
    if detector_cls == LightweightMotionDetector:
        valid_keys = ("history", "var_threshold", "detect_shadows")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return detector_cls(**filtered_kwargs)
    return detector_cls(**kwargs)
