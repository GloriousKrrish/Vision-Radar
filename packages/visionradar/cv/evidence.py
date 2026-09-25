import cv2
import os
import numpy as np
from typing import Dict, Any, Optional

class EvidenceWriter:
    """
    Saves candidate violation keyframe snapshots and generates auditable evidence packages.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_evidence_package(
        self,
        violation_id: str,
        frame_img: np.ndarray,
        bbox: tuple,
        metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Saves full event frame image and cropped vehicle snapshot image.
        """
        v_dir = os.path.join(self.output_dir, violation_id)
        os.makedirs(v_dir, exist_ok=True)

        full_frame_path = os.path.join(v_dir, "event_frame.jpg")
        crop_path = os.path.join(v_dir, "vehicle_crop.jpg")

        cv2.imwrite(full_frame_path, frame_img)

        # Crop bounding box region
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame_img.shape[:2]
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10)
        y2 = min(h, y2 + 10)

        crop_img = frame_img[y1:y2, x1:x2]
        if crop_img.size > 0:
            cv2.imwrite(crop_path, crop_img)
        else:
            cv2.imwrite(crop_path, frame_img)

        return {
            "full_frame": full_frame_path,
            "crop_frame": crop_path,
            "dir": v_dir
        }
