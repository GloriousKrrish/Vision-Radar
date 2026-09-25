from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class Detection:
    """
    Typed Detection result container.
    """
    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: Tuple[float, float, float, float],  # (x1, y1, x2, y2)
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = float(confidence)
        self.bbox = bbox
        self.frame_index = frame_index
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None
        }

class BaseDetector(ABC):
    """
    Abstract interface for object detectors.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.35) -> List[Detection]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass
