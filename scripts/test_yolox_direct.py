"""
Diagnostic script to test YOLOX detector directly on Traffic1.mp4 frames.
"""
import os
import sys
import hashlib
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detection.yolox import YOLOXDetector, COCO_VEHICLE_CLASSES

VIDEO_PATH = "Traffic1.mp4" if os.path.exists("Traffic1.mp4") else "data/videos/Traffic1.mp4"
MODEL_PATH = "data/models/yolox_nano.onnx"

def test_yolox():
    print("=== YOLOX DIRECT DETECTOR TEST ===")
    assert os.path.exists(VIDEO_PATH), f"Video file not found at {VIDEO_PATH}"
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"

    with open(MODEL_PATH, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"Video path: {VIDEO_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model hash: {model_hash}")

    detector = YOLOXDetector(model_path=MODEL_PATH, confidence_threshold=0.25, input_size=(640, 640))

    print(f"Input resolution: {detector.input_size}")
    print(f"Backend: {detector.backend_name}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print(f"NMS threshold: {detector.nms_threshold}")

    decoder = VideoDecoder(VIDEO_PATH)
    frames = list(decoder.decode_frames())
    print(f"Total video frames: {len(frames)}")

    # Test on frame 10 (known to have vehicles)
    frame_idx, timestamp, frame = frames[10]
    print(f"\nTesting on frame index {frame_idx} (timestamp {timestamp:.2f}s, shape {frame.shape}):")

    outputs, cand_indices, post_nms_dets, vehicle_dets = detector.detect_verbose(frame, confidence_threshold=0.25)
    print(f"Raw output shape: {outputs.shape}")
    print(f"Raw candidate prediction count (score >= 0.25): {len(cand_indices)}")
    print(f"Post-NMS detection count: {len(post_nms_dets)}")
    print(f"Vehicle detection count: {len(vehicle_dets)}")

    classes_detected = [d.class_name for d in vehicle_dets]
    print(f"Classes detected: {classes_detected}")

    for idx, d in enumerate(vehicle_dets):
        print(f"  Det #{idx+1}: class={d.class_name}, conf={d.confidence:.3f}, bbox={[round(c, 1) for c in d.bbox]}")

    assert len(vehicle_dets) > 0, "YOLOX produced zero vehicle detections on a known traffic frame!"
    print("\n[PASS] YOLOX direct detector test successful!")

if __name__ == "__main__":
    test_yolox()
