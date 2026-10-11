"""
VisionRadar Evaluation Schemas (Task 1 Compliant).

Defines canonical internal schemas for ground-truth detections and ground-truth trajectories.
Read-only schema; does not alter original dataset files.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class GroundTruthDetection:
    """
    Canonical representation of a single ground-truth object detection.
    """
    dataset: str
    split: str
    image_id: str
    frame_index: int
    class_name: str
    class_id: int
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] in absolute pixel coordinates
    confidence: Optional[float] = None
    source_annotation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "image_id": self.image_id,
            "frame_index": self.frame_index,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox_xyxy": [round(c, 2) for c in self.bbox_xyxy],
            "confidence": self.confidence,
            "source_annotation": self.source_annotation
        }


@dataclass
class GroundTruthTrack:
    """
    Canonical representation of a single ground-truth vehicle track point.
    """
    dataset: str
    sequence_id: str
    frame_index: int
    gt_track_id: int
    class_name: str
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] in absolute pixel coordinates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sequence_id": self.sequence_id,
            "frame_index": self.frame_index,
            "gt_track_id": self.gt_track_id,
            "class_name": self.class_name,
            "bbox_xyxy": [round(c, 2) for c in self.bbox_xyxy]
        }
