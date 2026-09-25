"""
VisionRadar Perception Engine — Inference Engine Wrapper

Wraps ONNX model loading, session warmup, thread configuration, and forward execution.
Exposes infer() method and reports exact inference time.
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import Tuple, Optional, Any

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    ONNX inference engine wrapper utilizing OpenCV DNN or ONNX Runtime.
    Performs 3-5 warmup iterations on startup to eliminate cold-start spikes.
    Pins thread count explicitly for deterministic latency.
    """
    def __init__(
        self,
        model_path: str = "data/models/yolox_nano.onnx",
        input_size: Tuple[int, int] = (640, 640),
        intra_op_threads: int = 8,
        inter_op_threads: int = 1
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self.net = None
        self.backend_name = "Unknown"
        self.warmup_completed = False
        self.last_inference_ms = 0.0

        self._load_and_warmup()

    def _load_and_warmup(self):
        if not os.path.exists(self.model_path):
            if os.path.exists("data/models/yolox_nano.onnx"):
                self.model_path = "data/models/yolox_nano.onnx"

        assert os.path.exists(self.model_path), f"ONNX model file not found at {self.model_path}"

        # Configure OpenCV DNN thread count
        cv2.setNumThreads(self.intra_op_threads)
        logger.info(f"[InferenceEngine] OpenCV DNN thread count set to {self.intra_op_threads}")

        t0_load = time.perf_counter()
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        t1_load = time.perf_counter()
        logger.info(f"[InferenceEngine] Loaded ONNX model from {self.model_path} in {(t1_load - t0_load)*1000:.1f} ms")

        self.backend_name = f"OpenCV-DNN (CPU, {self.intra_op_threads} threads)"

        # Warmup execution: 5 dummy iterations to prime internal buffers
        target_w, target_h = self.input_size
        dummy_blob = np.zeros((1, 3, target_h, target_w), dtype=np.float32)

        t0_warmup = time.perf_counter()
        for i in range(5):
            self.net.setInput(dummy_blob)
            _ = self.net.forward()
        t1_warmup = time.perf_counter()

        self.warmup_completed = True
        logger.info(f"[InferenceEngine] Warmup completed (5 iterations) in {(t1_warmup - t0_warmup)*1000:.1f} ms")

    def infer(self, blob: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Executes model inference on preprocessed blob tensor.
        
        Returns:
            raw_output: raw prediction numpy array
            inference_ms: duration in milliseconds
        """
        if self.net is None:
            raise RuntimeError("InferenceEngine net is not initialized")

        t0 = time.perf_counter()
        self.net.setInput(blob)
        outputs = self.net.forward()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        self.last_inference_ms = elapsed_ms
        return outputs, elapsed_ms
