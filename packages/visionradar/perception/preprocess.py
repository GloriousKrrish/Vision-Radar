"""
VisionRadar Perception Engine — Preprocessing Module

Provides letterbox resizing, padding, normalization, and tensor layout formatting
for YOLOX-Nano ONNX model inference. Pure function.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def letterbox_preprocess(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: int = 114
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Preprocesses raw BGR image for YOLOX-Nano model input.
    
    Args:
        image: raw BGR numpy array (H, W, C)
        target_size: (target_width, target_height)
        pad_color: letterbox border fill color (default 114 for YOLOX)

    Returns:
        blob: float32 numpy array layout (1, 3, target_height, target_width)
        scale: min ratio used to resize image
        pad: (pad_left, pad_top) pixel offsets
    """
    if image is None or image.size == 0:
        target_w, target_h = target_size
        dummy_blob = np.zeros((1, 3, target_h, target_w), dtype=np.float32)
        return dummy_blob, 1.0, (0, 0)

    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale factor and target resized dimensions
    scale = min(target_w / float(w), target_h / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))

    # Resize image
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Pre-allocate letterbox canvas with pad_color
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    pad_left = int(round((target_w - nw) / 2.0))
    pad_top = int(round((target_h - nh) / 2.0))
    padded[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

    # Format as HWC -> CHW float32 blob (1, 3, H, W)
    blob = cv2.dnn.blobFromImage(padded, 1.0, (target_w, target_h), (0, 0, 0), swapRB=False, crop=False)

    return blob, scale, (pad_left, pad_top)
