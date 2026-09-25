import os
import time
import cv2
import numpy as np
import hashlib
from typing import Dict, Any, List, Tuple

COCO_VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def generate_grids_and_strides(input_size: Tuple[int, int] = (640, 640)):
    strides = [8, 16, 32]
    grid_pts = []
    stride_pts = []
    in_h, in_w = input_size[1], input_size[0]

    for s in strides:
        g_h, g_w = in_h // s, in_w // s
        for y in range(g_h):
            for x in range(g_w):
                grid_pts.append((x, y))
                stride_pts.append(s)

    return np.array(grid_pts, dtype=np.float32), np.array(stride_pts, dtype=np.float32)

def letterbox_preprocess(img: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / float(w), target_h / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    
    dw = (target_w - nw) / 2.0
    dh = (target_h - nh) / 2.0
    top, left = int(round(dh)), int(round(dw))
    
    padded[top:top+nh, left:left+nw] = resized
    return padded, scale, left, top

def run_forensic_debug(image_path: str, model_path: str = "data/models/yolox_nano.onnx") -> Dict[str, Any]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]

    # Preprocessing
    padded, scale, pad_left, pad_top = letterbox_preprocess(img, (640, 640))
    blob = cv2.dnn.blobFromImage(padded, 1.0, (640, 640), (0, 0, 0), swapRB=False, crop=False)

    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    net.setInput(blob)
    outputs = net.forward()
    preds = outputs[0] if len(outputs.shape) == 3 else outputs

    grids, strides = generate_grids_and_strides((640, 640))

    raw_prediction_count = preds.shape[0]
    finite_mask = np.isfinite(preds)
    finite_count = int(np.sum(np.all(finite_mask, axis=1)))

    obj_conf = preds[:, 4]
    cls_conf = preds[:, 5:]

    class_ids = np.argmax(cls_conf, axis=1)
    class_max_scores = np.max(cls_conf, axis=1)
    combined_scores = obj_conf * class_max_scores

    # Confidence sweeps
    sweeps = {}
    for th in [0.01, 0.05, 0.10, 0.20, 0.25, 0.35]:
        above = np.where(combined_scores >= th)[0]
        sweeps[f"above_{th:.2f}"] = len(above)

    # Filter vehicle classes at 0.25
    veh_candidates = []
    boxes, confs, cids = [], [], []

    # Vectorized decoding
    cx = (preds[:, 0] + grids[:, 0]) * strides
    cy = (preds[:, 1] + grids[:, 1]) * strides
    bw = np.exp(np.clip(preds[:, 2], -10.0, 10.0)) * strides
    bh = np.exp(np.clip(preds[:, 3], -10.0, 10.0)) * strides

    for idx in range(raw_prediction_count):
        cid = int(class_ids[idx])
        score = float(combined_scores[idx])
        if cid in COCO_VEHICLE_CLASSES and score >= 0.25:
            veh_candidates.append(idx)
            # Map back from letterbox to original image coordinates
            c_x, c_y = cx[idx], cy[idx]
            b_w, b_h = bw[idx], bh[idx]

            x1 = (c_x - pad_left - b_w / 2.0) / scale
            y1 = (c_y - pad_top - b_h / 2.0) / scale
            w_orig = b_w / scale
            h_orig = b_h / scale

            boxes.append([int(x1), int(y1), int(w_orig), int(h_orig)])
            confs.append(score)
            cids.append(cid)

    # NMS Forensic
    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45)
    post_nms_count = len(indices) if len(indices) > 0 else 0

    final_detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            bx, by, bw_i, bh_i = boxes[idx]
            cid = cids[idx]
            final_detections.append({
                "class_id": cid,
                "class_name": COCO_VEHICLE_CLASSES[cid],
                "confidence": round(confs[idx], 4),
                "bbox": [max(0, bx), max(0, by), min(orig_w, bx + bw_i), min(orig_h, by + bh_i)]
            })

    report = {
        "image_path": image_path,
        "image_dims": [orig_w, orig_h],
        "raw_prediction_count": raw_prediction_count,
        "finite_value_count": finite_count,
        "objectness": {
            "min": float(np.min(obj_conf)),
            "max": float(np.max(obj_conf)),
            "mean": float(np.mean(obj_conf))
        },
        "class_confidence": {
            "min": float(np.min(class_max_scores)),
            "max": float(np.max(class_max_scores)),
            "mean": float(np.mean(class_max_scores))
        },
        "combined_confidence": {
            "min": float(np.min(combined_scores)),
            "max": float(np.max(combined_scores)),
            "mean": float(np.mean(combined_scores))
        },
        "confidence_sweeps": sweeps,
        "pre_nms_vehicle_candidates": len(veh_candidates),
        "post_nms_count": post_nms_count,
        "final_detection_count": len(final_detections),
        "detections": final_detections
    }
    return report

if __name__ == "__main__":
    os.makedirs("data/debug", exist_ok=True)
    report = run_forensic_debug("data/debug/yolox_test_frame.jpg")
    import json
    print(json.dumps(report, indent=2))
