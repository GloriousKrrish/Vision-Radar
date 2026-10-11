import time
import cv2
import numpy as np
import onnxruntime as ort

model_path = "data/models/yolox_nano.onnx"
dummy_416 = np.zeros((1, 3, 416, 416), dtype=np.float32)
dummy_640 = np.zeros((1, 3, 640, 640), dtype=np.float32)

print("=== 1. OpenCV DNN @ 640x640 (Status Quo) ===")
net_640 = cv2.dnn.readNetFromONNX(model_path)
net_640.setInput(dummy_640)
_ = net_640.forward()
times_cv_640 = []
for _ in range(20):
    t0 = time.perf_counter()
    net_640.setInput(dummy_640)
    _ = net_640.forward()
    times_cv_640.append((time.perf_counter() - t0) * 1000.0)
print(f"OpenCV DNN (640x640): Mean={np.mean(times_cv_640):.2f}ms, p95={np.percentile(times_cv_640, 95):.2f}ms")

print("\n=== 2. OpenCV DNN @ native 416x416 ===")
net_416 = cv2.dnn.readNetFromONNX(model_path)
net_416.setInput(dummy_416)
_ = net_416.forward()
times_cv_416 = []
for _ in range(20):
    t0 = time.perf_counter()
    net_416.setInput(dummy_416)
    _ = net_416.forward()
    times_cv_416.append((time.perf_counter() - t0) * 1000.0)
print(f"OpenCV DNN (416x416): Mean={np.mean(times_cv_416):.2f}ms, p95={np.percentile(times_cv_416, 95):.2f}ms")

print("\n=== 3. ONNX Runtime @ native 416x416 ===")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
_ = session.run(None, {input_name: dummy_416})
times_ort = []
for _ in range(20):
    t0 = time.perf_counter()
    _ = session.run(None, {input_name: dummy_416})
    times_ort.append((time.perf_counter() - t0) * 1000.0)
print(f"ONNX Runtime (416x416): Mean={np.mean(times_ort):.2f}ms, p95={np.percentile(times_ort, 95):.2f}ms")

speedup_ort = np.mean(times_cv_640) / np.mean(times_ort)
print(f"\nTOTAL SPEEDUP (OpenCV 640 -> ORT 416): {speedup_ort:.2f}x FASTER!")
