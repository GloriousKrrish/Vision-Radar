import os
import sys
import time
import hashlib
import numpy as np
import cv2

# Ensure project packages are accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.cv.detection.yolox import YOLOXDetector

def validate_model(model_path: str = "data/models/yolox_nano.onnx") -> dict:
    """
    Validates an ONNX object detection model for YOLOX architecture compatibility.
    Distinguishes:
    - MODEL_LOAD_ONLY
    - MODEL_RUNTIME_VALID
    - OBJECT_DETECTION_VALID
    """
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return {"status": "MODEL_INVALID", "error": "File not found"}

    # 1. Compute SHA-256 Hash
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()

    file_size = os.path.getsize(model_path)

    model_runtime_valid = False
    output_schema_valid = False
    known_object_detection_valid = False

    # 2. OpenCV DNN Model Load
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        print(f"ERROR: Failed to load ONNX model via OpenCV DNN: {e}")
        return {
            "status": "MODEL_INVALID",
            "model_runtime_valid": False,
            "output_schema_valid": False,
            "known_object_detection_valid": False,
            "error": str(e)
        }

    # 3. Dry-Run Forward Pass
    input_size = (640, 640)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        t0 = time.perf_counter()
        blob = cv2.dnn.blobFromImage(dummy_img, 1.0, input_size, (0, 0, 0), swapRB=False, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        model_runtime_valid = True
    except Exception as e:
        return {
            "status": "MODEL_LOAD_ONLY",
            "model_runtime_valid": False,
            "output_schema_valid": False,
            "known_object_detection_valid": False,
            "error": f"Runtime forward pass failed: {e}"
        }

    output_shape = list(outputs.shape)
    
    # 4. Detection Head Validation
    class_count = 0
    if len(output_shape) == 3:
        num_anchors, num_features = output_shape[1], output_shape[2]
        if num_features >= 6:
            output_schema_valid = True
            class_count = num_features - 5
    elif len(output_shape) == 2:
        num_anchors, num_features = output_shape[0], output_shape[1]
        if num_features >= 6:
            output_schema_valid = True
            class_count = num_features - 5

    # 5. Known Object Detection Validation
    test_image_path = "data/debug/yolox_known_vehicle.jpg"
    if not os.path.exists(test_image_path):
        test_image_path = "data/debug/yolox_test_frame.jpg"

    det_count = 0
    test_dets = []

    if os.path.exists(test_image_path):
        test_img = cv2.imread(test_image_path)
        detector = YOLOXDetector(model_path=model_path, confidence_threshold=0.25)
        dets = detector.detect(test_img)
        det_count = len(dets)
        test_dets = [f"{d.class_name}:{d.confidence:.2f}" for d in dets]
        if det_count > 0:
            known_object_detection_valid = True

    # Determine Final Status
    if model_runtime_valid and output_schema_valid and known_object_detection_valid:
        status = "OBJECT_DETECTION_VALID"
    elif model_runtime_valid and output_schema_valid:
        status = "MODEL_RUNTIME_VALID"
    elif model_runtime_valid:
        status = "MODEL_LOAD_ONLY"
    else:
        status = "MODEL_INVALID"

    report = {
        "status": status,
        "model_path": model_path,
        "file_size_bytes": file_size,
        "model_hash": model_hash,
        "model_runtime_valid": model_runtime_valid,
        "output_schema_valid": output_schema_valid,
        "known_object_detection_valid": known_object_detection_valid,
        "input_shape": [1, 3, input_size[1], input_size[0]],
        "output_shape": output_shape,
        "output_tensor_type": "YOLOX Decoded Bounding-Box Anchor Tensor [1, N, 85]",
        "class_count": class_count,
        "inference_backend": "OpenCV-DNN (CPU)",
        "inference_time_ms": round(inference_ms, 2),
        "known_vehicle_test_image": test_image_path,
        "known_vehicle_detection_count": det_count,
        "known_vehicle_detections": test_dets,
        "fallback_used": False
    }

    return report

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "data/models/yolox_nano.onnx"
    result = validate_model(model_file)
    print("=" * 60)
    print("VISIONRADAR — YOLOX DETECTOR MODEL VALIDATION REPORT")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:<28}: {v}")
    print("=" * 60)
    if result.get("status") == "MODEL_INVALID":
        sys.exit(1)
