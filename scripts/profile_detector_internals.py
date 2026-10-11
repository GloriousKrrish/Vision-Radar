import time
import os
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "packages")

import cv2
import numpy as np
from visionradar.cv.detection.yolox import YOLOXDetector

cap = cv2.VideoCapture("data/videos/Traffic1.mp4")
ret, frame = cap.read()
cap.release()

detector = YOLOXDetector(model_path="data/models/yolox_nano.onnx")

# Measure individual components
times_resize = []
times_pad = []
times_blob = []
times_forward = []
times_decode = []
times_nms = []
times_total = []

for _ in range(30):
    t_start = time.perf_counter()

    h, w = frame.shape[:2]
    target_w, target_h = detector.input_size
    scale = min(target_w / float(w), target_h / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))

    t0 = time.perf_counter()
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    times_resize.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_left = int(round((target_w - nw) / 2.0))
    pad_top = int(round((target_h - nh) / 2.0))
    padded[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    times_pad.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    blob = cv2.dnn.blobFromImage(padded, 1.0, detector.input_size, (0, 0, 0), swapRB=False, crop=False)
    times_blob.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    detector.net.setInput(blob)
    outputs = detector.net.forward()
    times_forward.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    preds = outputs[0] if len(outputs.shape) == 3 else outputs
    cx = (preds[:, 0] + detector.grids[:, 0]) * detector.strides
    cy = (preds[:, 1] + detector.grids[:, 1]) * detector.strides
    bw = np.exp(np.clip(preds[:, 2], -10.0, 10.0)) * detector.strides
    bh = np.exp(np.clip(preds[:, 3], -10.0, 10.0)) * detector.strides
    obj_conf = preds[:, 4]
    cls_conf = preds[:, 5:]
    class_ids = np.argmax(cls_conf, axis=1)
    class_max_scores = np.max(cls_conf, axis=1)
    total_scores = obj_conf * class_max_scores
    cand_indices = np.where(total_scores >= 0.25)[0]
    times_decode.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    boxes, confidences, final_class_ids = [], [], []
    for idx in cand_indices:
        cid = int(class_ids[idx])
        c_x, c_y = float(cx[idx]), float(cy[idx])
        b_w, b_h = float(bw[idx]), float(bh[idx])
        x1 = (c_x - pad_left - b_w / 2.0) / scale
        y1 = (c_y - pad_top - b_h / 2.0) / scale
        boxes.append([int(x1), int(y1), int(b_w / scale), int(b_h / scale)])
        confidences.append(float(total_scores[idx]))
        final_class_ids.append(cid)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    times_nms.append((time.perf_counter() - t0) * 1000.0)

    times_total.append((time.perf_counter() - t_start) * 1000.0)

print("--- DETECTOR INTERNAL TIMING (1080p frame) ---")
print(f"1. cv2.resize:      Mean={np.mean(times_resize):.2f}ms")
print(f"2. np.full + pad:   Mean={np.mean(times_pad):.2f}ms")
print(f"3. blobFromImage:   Mean={np.mean(times_blob):.2f}ms")
print(f"4. net.forward():   Mean={np.mean(times_forward):.2f}ms")
print(f"5. Grid decode:     Mean={np.mean(times_decode):.2f}ms")
print(f"6. Unpack + NMS:    Mean={np.mean(times_nms):.2f}ms")
print(f"TOTAL DETECT TIME:  Mean={np.mean(times_total):.2f}ms")
