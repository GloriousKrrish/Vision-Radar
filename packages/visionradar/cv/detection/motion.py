import numpy as np
import cv2
from typing import List, Dict, Any
from visionradar.cv.detection.base import BaseDetector, Detection

class LightweightMotionDetector(BaseDetector):
    """
    CPU-friendly baseline motion detector using MOG2 background subtraction and adaptive contours.
    Serves as the MOG2 baseline for experiment comparisons.
    """

    def __init__(self, model_name: str = "MOG2-Motion-Baseline"):
        self.model_name = model_name
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.35) -> List[Detection]:
        detections = []
        if image is None or image.size == 0:
            return detections

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]

        horizon_y = int(h * 0.20)
        roi = image[horizon_y:, :]
        gray_roi = gray[horizon_y:, :]

        fg_mask = self.bg_subtractor.apply(roi)
        _, thresh_fg = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        _, thresh_dark = cv2.threshold(gray_roi, 130, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_or(thresh_fg, thresh_dark)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            min_area = max(100.0, (w * h) * 0.001)
            max_area = (w * h) * 0.25

            if min_area < area < max_area:
                x, y_roi, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = bw / float(bh)
                if 0.4 <= aspect_ratio <= 3.5:
                    y = y_roi + horizon_y
                    cls_name = "Car" if bw < 70 else ("Truck" if bw > 140 else "SUV")
                    conf = min(0.99, max(0.65, 0.85 + (area / max_area) * 0.15))
                    if conf >= confidence_threshold:
                        detections.append(Detection(
                            class_id=0 if cls_name == "Car" else (1 if cls_name == "SUV" else 2),
                            class_name=cls_name,
                            confidence=conf,
                            bbox=(float(x), float(y), float(x + bw), float(y + bh))
                        ))
        return detections

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "version": "1.0.0",
            "framework": "OpenCV-MOG2",
            "license": "Apache-2.0",
            "device": "CPU",
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        }
