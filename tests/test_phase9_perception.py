import os
import pytest
import numpy as np
import cv2

from visionradar.cv.detection.yolox import YOLOXDetector, COCO_VEHICLE_CLASSES
from visionradar.cv.tracker import ByteTrackTracker

MODEL_PATH = "data/models/yolox_nano.onnx"

def test_production_model_existence():
    assert os.path.exists(MODEL_PATH), "Production YOLOX ONNX model must exist at data/models/yolox_nano.onnx"

def test_yolox_detector_baseline_init():
    detector = YOLOXDetector(model_path=MODEL_PATH, confidence_threshold=0.25, input_size=(640, 640))
    assert detector.valid_onnx_detector is True
    assert detector.confidence_threshold == 0.25
    assert detector.input_size == (640, 640)
    assert detector.model_name == "YOLOX-Nano-ONNX"

def test_coco_class_mapping():
    assert COCO_VEHICLE_CLASSES[2] == "Car"
    assert COCO_VEHICLE_CLASSES[3] == "Motorcycle"
    assert COCO_VEHICLE_CLASSES[5] == "Bus"
    assert COCO_VEHICLE_CLASSES[7] == "Truck"

def test_dummy_image_detection():
    detector = YOLOXDetector(model_path=MODEL_PATH)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (100, 100), (300, 300), (255, 255, 255), -1)
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list)

def test_threshold_ablation_experiment_safety():
    detector = YOLOXDetector(model_path=MODEL_PATH)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for conf in [0.15, 0.25, 0.50]:
        dets = detector.detect(dummy_frame, confidence_threshold=conf)
        assert isinstance(dets, list)

def test_resolution_ablation_experiment_safety():
    for res in [(640, 640), (512, 512), (416, 416)]:
        det = YOLOXDetector(model_path=MODEL_PATH, input_size=res)
        assert det.input_size == res
        dummy = np.zeros((res[1], res[0], 3), dtype=np.uint8)
        dets = det.detect(dummy)
        assert isinstance(dets, list)

def test_tracker_continuity_safety():
    tracker = ByteTrackTracker()
    assert tracker.active_tracks == {}
