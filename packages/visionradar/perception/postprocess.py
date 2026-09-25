"""
VisionRadar Perception Engine — Postprocessing Module

Pure function for YOLOX anchor grid decoding, score filtering, class mapping,
and deterministic NMS suppression to produce immutable VehicleDetection objects.
"""

import uuid
import numpy as np
import cv2
from typing import List, Tuple, Dict
from visionradar.perception.schemas import VehicleDetection, VehicleClass, BoundingBox

# COCO Vehicle Class ID to VehicleClass enum mapping
COCO_CLASS_MAPPING = {
    2: VehicleClass.CAR,
    3: VehicleClass.MOTORCYCLE,
    5: VehicleClass.BUS,
    7: VehicleClass.TRUCK
}


def _generate_yolox_grids(input_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-generates YOLOX grid coordinates and strides for 80x80, 40x40, 20x20 feature maps."""
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


# Pre-compute static 640x640 grids once
STATIC_GRIDS, STATIC_STRIDES = _generate_yolox_grids((640, 640))


def postprocess_yolox_output(
    raw_outputs: np.ndarray,
    frame_id: int,
    source_shape: Tuple[int, int],
    input_size: Tuple[int, int] = (640, 640),
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    scale: float = 1.0,
    pad: Tuple[int, int] = (0, 0)
) -> List[VehicleDetection]:
    """
    Decodes raw YOLOX model output tensor into VehicleDetection instances.

    Args:
        raw_outputs: raw model output array (1, 8400, 85) or (8400, 85)
        frame_id: current monotonic frame identifier
        source_shape: (src_width, src_height) of original image
        input_size: (target_width, target_height)
        confidence_threshold: score filtering threshold (0.25)
        nms_threshold: IoU NMS threshold (0.45)
        scale: min resize ratio used in letterbox
        pad: (pad_left, pad_top) pixel padding in letterbox

    Returns:
        List[VehicleDetection]: immutable detection instances
    """
    if raw_outputs is None or raw_outputs.size == 0:
        return []

    preds = raw_outputs[0] if len(raw_outputs.shape) == 3 else raw_outputs
    if len(preds.shape) != 2 or preds.shape[1] < 6:
        return []

    src_w, src_h = source_shape
    pad_left, pad_top = pad

    # Anchor Grid Decoding
    if preds.shape[0] == len(STATIC_GRIDS) and input_size == (640, 640):
        grids, strides = STATIC_GRIDS, STATIC_STRIDES
    else:
        grids, strides = _generate_yolox_grids(input_size)

    if preds.shape[0] == len(grids):
        cx = (preds[:, 0] + grids[:, 0]) * strides
        cy = (preds[:, 1] + grids[:, 1]) * strides
        bw = np.exp(np.clip(preds[:, 2], -10.0, 10.0)) * strides
        bh = np.exp(np.clip(preds[:, 3], -10.0, 10.0)) * strides
    else:
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

    obj_conf = preds[:, 4]
    cls_conf = preds[:, 5:]
    class_ids = np.argmax(cls_conf, axis=1)
    class_max_scores = np.max(cls_conf, axis=1)
    total_scores = obj_conf * class_max_scores

    # Candidate filtering
    cand_indices = np.where(total_scores >= confidence_threshold)[0]
    if len(cand_indices) == 0:
        return []

    boxes, confidences, final_class_ids, class_confs = [], [], [], []

    for idx in cand_indices:
        cid = int(class_ids[idx])
        if cid in COCO_CLASS_MAPPING:
            c_x, c_y = float(cx[idx]), float(cy[idx])
            b_w, b_h = float(bw[idx]), float(bh[idx])

            # Un-pad and un-scale back to original source image space
            x1 = (c_x - pad_left - b_w / 2.0) / scale
            y1 = (c_y - pad_top - b_h / 2.0) / scale
            bw_s = b_w / scale
            bh_s = b_h / scale

            boxes.append([int(round(x1)), int(round(y1)), int(round(bw_s)), int(round(bh_s))])
            confidences.append(float(total_scores[idx]))
            final_class_ids.append(cid)
            class_confs.append(float(class_max_scores[idx]))

    if not boxes:
        return []

    # Deterministic NMS with cv2.dnn.NMSBoxes (include boundary scores)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold - 1e-6, nms_threshold)
    detections: List[VehicleDetection] = []

    if len(indices) > 0:
        flattened_indices = indices.flatten()
        # Sort indices deterministically by confidence desc, then idx asc for tie-breaking
        sorted_indices = sorted(flattened_indices, key=lambda i: (-confidences[i], i))

        for idx in sorted_indices:
            bx, by, bw_i, bh_i = boxes[idx]
            cid = final_class_ids[idx]
            vclass = COCO_CLASS_MAPPING.get(cid, VehicleClass.CAR)

            # Clamp coordinates to original image boundaries
            rx1 = float(max(0, bx))
            ry1 = float(max(0, by))
            rx2 = float(min(src_w, bx + bw_i))
            ry2 = float(min(src_h, by + bh_i))

            detections.append(VehicleDetection(
                detection_id=uuid.uuid4().hex,
                frame_id=frame_id,
                bbox=BoundingBox(x1=rx1, y1=ry1, x2=rx2, y2=ry2),
                confidence=confidences[idx],
                vehicle_class=vclass,
                class_confidence=class_confs[idx]
            ))

    return detections
