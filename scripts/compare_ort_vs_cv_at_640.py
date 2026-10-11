import os
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "packages")

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from visionradar.cv.detection.yolox import YOLOXDetector, COCO_VEHICLE_CLASSES

video_path = "data/videos/Traffic1.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

# 1. OpenCV DNN at 640x640
det_cv = YOLOXDetector(model_path="data/models/yolox_nano.onnx", input_size=(640, 640))
dets_cv = det_cv.detect(frame)
print(f"=== OpenCV DNN (640x640): {len(dets_cv)} Detections ===")
for d in dets_cv:
    print(f"  Class: {d.class_name}, Conf: {d.confidence:.4f}, BBox: {[round(c, 1) for c in d.bbox]}")

# 2. ONNX Runtime at 640x640
model = onnx.load("data/models/yolox_nano.onnx")
model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 640
model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 640
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess = ort.InferenceSession(model.SerializeToString(), sess_options, providers=['CPUExecutionProvider'])

h, w = frame.shape[:2]
target_w, target_h = (640, 640)
scale = min(target_w / float(w), target_h / float(h))
nw, nh = int(round(w * scale)), int(round(h * scale))
resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
pad_left = int(round((target_w - nw) / 2.0))
pad_top = int(round((target_h - nh) / 2.0))
padded[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

blob = cv2.dnn.blobFromImage(padded, 1.0, (640, 640), (0, 0, 0), swapRB=False)
outputs = sess.run(None, {'images': blob})
preds = outputs[0][0]

strides = [8, 16, 32]
grid_pts, stride_pts = [], []
for s in strides:
    gh, gw = target_h // s, target_w // s
    for y in range(gh):
        for x in range(gw):
            grid_pts.append((x, y))
            stride_pts.append(s)
grids = np.array(grid_pts, dtype=np.float32)
strides_arr = np.array(stride_pts, dtype=np.float32)

cx = (preds[:, 0] + grids[:, 0]) * strides_arr
cy = (preds[:, 1] + grids[:, 1]) * strides_arr
bw = np.exp(np.clip(preds[:, 2], -10.0, 10.0)) * strides_arr
bh = np.exp(np.clip(preds[:, 3], -10.0, 10.0)) * strides_arr

obj_conf = preds[:, 4]
cls_conf = preds[:, 5:]
class_ids = np.argmax(cls_conf, axis=1)
total_scores = obj_conf * np.max(cls_conf, axis=1)

conf_thresh = 0.25
cand_indices = np.where(total_scores >= conf_thresh)[0]
boxes, confidences, final_class_ids = [], [], []

for idx in cand_indices:
    cid = int(class_ids[idx])
    if cid in COCO_VEHICLE_CLASSES:
        c_x, c_y = float(cx[idx]), float(cy[idx])
        b_w, b_h = float(bw[idx]), float(bh[idx])
        x1 = (c_x - pad_left - b_w / 2.0) / scale
        y1 = (c_y - pad_top - b_h / 2.0) / scale
        boxes.append([int(x1), int(y1), int(b_w / scale), int(b_h / scale)])
        confidences.append(float(total_scores[idx]))
        final_class_ids.append(cid)

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.45)
print(f"\n=== ONNX Runtime (640x640): {len(indices)} Detections ===")
for idx in indices.flatten():
    bx, by, bw_i, bh_i = boxes[idx]
    cid = final_class_ids[idx]
    print(f"  Class: {COCO_VEHICLE_CLASSES.get(cid)}, Conf: {confidences[idx]:.4f}, BBox: {[bx, by, bx+bw_i, by+bh_i]}")
