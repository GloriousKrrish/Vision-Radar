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
    "yolo": YOLOXDetector,
    "yolox-nano-baseline": YOLOXDetector,
    "yolox-baseline": YOLOXDetector,
    "yolox-nano-optimized": YOLOXDetector,
    "yolox-fast": YOLOXDetector,
    "yolox-highres": YOLOXDetector
}

def get_detector(name: str = "yolox", **kwargs) -> BaseDetector:
    """
    Factory function to instantiate object detectors from registry by name.
    Preserves baseline configuration while offering optimized and high-resolution profiles.
    """
    key = name.lower().strip()
    detector_cls = DETECTOR_REGISTRY.get(key, YOLOXDetector)
    if detector_cls == LightweightMotionDetector:
        valid_keys = ("history", "var_threshold", "detect_shadows")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return detector_cls(**filtered_kwargs)

    # Configurable profile resolution
    if key in ("yolox-nano-baseline", "yolox-baseline"):
        kwargs.setdefault("input_size", (640, 640))
        kwargs.setdefault("class_specific_nms", False)
    elif key in ("yolox-nano-optimized", "yolox-fast", "yolox-nano-416", "yolox", "yolox-nano", "yolo"):
        kwargs.setdefault("input_size", (416, 416))
        kwargs.setdefault("class_specific_nms", True)
    elif key in ("yolox-highres", "yolox-640"):
        kwargs.setdefault("input_size", (640, 640))
        kwargs.setdefault("class_specific_nms", True)

    return detector_cls(**kwargs)
