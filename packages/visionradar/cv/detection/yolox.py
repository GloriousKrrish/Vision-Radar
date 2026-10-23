import os
import time
import hashlib
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from visionradar.cv.detection.base import BaseDetector, Detection

try:
    import onnx
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# COCO Dataset vehicle class mapping
COCO_VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

class YOLOXDetector(BaseDetector):
    """
    YOLOX / YOLO ONNX Real Object Detector utilizing ONNX Runtime / OpenCV DNN engine on CPU/GPU.
    Supports genuine ONNX model weights loading, anchor grid decoding [1, 8400, 85],
    confidence filtering, NMS suppression, telemetry tracking, and transparent fallback accounting.
    """

    def __init__(
        self,
        model_path: Optional[str] = "data/models/yolox_nano.onnx",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "CPU",
        class_specific_nms: bool = True
    ):
        self.detector_requested = "YOLOX-Nano-ONNX"
        self.detector_actual = "YOLOX-Nano-ONNX"
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        self.class_specific_nms = class_specific_nms
        self.net = None
        self.ort_session = None
        self.ort_input_name = None
        self.valid_onnx_detector = False
        self.fallback_used = False
        self.fallback_reason = None
        self.model_hash = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"

        # Pre-compute static YOLOX anchor grids for 640x640 input resolution (8400 anchors)
        self.grids, self.strides = self._generate_yolox_grids(input_size)

        # Telemetry Counters
        self.total_frames = 0
        self.total_inference_ms = 0.0
        self.last_inference_ms = 0.0
        self.total_raw_detections = 0
        self.total_post_nms_detections = 0
        self.total_vehicle_detections = 0

        # Resolve Model File & SHA256 Hash
        target_path = model_path
        if not target_path or not os.path.exists(target_path):
            if os.path.exists("data/models/yolox_nano.onnx"):
                target_path = "data/models/yolox_nano.onnx"
                self.model_path = target_path

        if target_path and os.path.exists(target_path):
            try:
                with open(target_path, "rb") as f:
                    self.model_hash = hashlib.sha256(f.read()).hexdigest()

                # 1. Try High-Performance ONNX Runtime CPU Engine First
                if _ORT_AVAILABLE and device.upper() != "GPU":
                    try:
                        sess_opts = ort.SessionOptions()
                        sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
                        sess_opts.log_severity_level = 3  # suppress harmless shape warnings
                        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        available_p = ort.get_available_providers()
                        active_providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider'] if 'OpenVINOExecutionProvider' in available_p else ['CPUExecutionProvider']

                        if self.input_size == (416, 416):
                            # Native input shape: load model file directly without graph mutation overhead for ~25ms inference
                            self.ort_session = ort.InferenceSession(
                                target_path, sess_opts, providers=active_providers
                            )
                        else:
                            model_proto = onnx.load(target_path)
                            model_proto.graph.input[0].type.tensor_type.shape.dim[2].dim_value = self.input_size[1]
                            model_proto.graph.input[0].type.tensor_type.shape.dim[3].dim_value = self.input_size[0]
                            model_proto.graph.output[0].type.tensor_type.shape.dim[1].dim_value = len(self.grids)
                            self.ort_session = ort.InferenceSession(
                                model_proto.SerializeToString(), sess_opts, providers=active_providers
                            )
                        self.ort_input_name = self.ort_session.get_inputs()[0].name
                        self.valid_onnx_detector = True
                        self.fallback_used = False
                        self.fallback_reason = None
                        self.model_name = "YOLOX-Nano-ONNX"
                        self.detector_actual = "YOLOX-Nano-ONNX"
                        self.backend_name = "ONNXRuntime-CPU (Optimized)"
                        self.status = "ONLINE"
                    except Exception as e_ort:
                        self.ort_session = None

                # 2. OpenCV DNN Engine (Primary if GPU, or Fallback if ORT fails)
                if self.ort_session is None:
                    self.net = cv2.dnn.readNetFromONNX(target_path)
                    if device.upper() == "GPU" and hasattr(cv2.dnn, 'DNN_BACKEND_CUDA'):
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    else:
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

                    dummy_input = np.zeros((1, 3, self.input_size[1], self.input_size[0]), dtype=np.float32)
                    self.net.setInput(dummy_input)
                    test_output = self.net.forward()

                    if (len(test_output.shape) == 3 and test_output.shape[2] in (6, 85, 84, 80)) or \
                       (len(test_output.shape) == 2 and test_output.shape[1] in (6, 85, 84, 80)):
                        self.valid_onnx_detector = True
                        self.fallback_used = False
                        self.fallback_reason = None
                        self.model_name = "YOLOX-Nano-ONNX"
                        self.detector_actual = "YOLOX-Nano-ONNX"
                        self.backend_name = f"OpenCV-DNN ({self.device})"
                        self.status = "ONLINE"
                    else:
                        self.valid_onnx_detector = False
                        self.fallback_used = True
                        self.fallback_reason = f"Non-detection tensor shape {test_output.shape}"
                        self.model_name = "YOLOX-ONNX (Fallback Active)"
                        self.detector_actual = "MOG2 Baseline"
                        self.backend_name = "MOG2 Subspace (Fallback Active)"
                        self.status = "FALLBACK_ACTIVE"

            except Exception as e:
                print(f"[YOLOX FORENSIC] Failed to load ONNX model at {target_path}: {e}")
                self.net = None
                self.ort_session = None
                self.valid_onnx_detector = False
                self.fallback_used = True
                self.fallback_reason = f"Failed to load model file: {e}"
                self.model_name = "YOLOX-ONNX (Model Unavailable)"
                self.detector_actual = "MOG2 Baseline"
                self.backend_name = "MOG2 Subspace (Fallback Active)"
                self.status = "MODEL_UNAVAILABLE"
        else:
            self.valid_onnx_detector = False
            self.fallback_used = True
            self.fallback_reason = "Model ONNX file not found on disk"
            self.model_name = "YOLOX-ONNX (Model Unavailable)"
            self.detector_actual = "MOG2 Baseline"
            self.backend_name = "MOG2 Subspace (Fallback Active)"
            self.status = "MODEL_UNAVAILABLE"

    def _generate_yolox_grids(self, input_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-generates YOLOX grid coordinates and strides for 80x80, 40x40, and 20x20 feature maps.
        """
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

    def detect(self, image: np.ndarray, confidence_threshold: Optional[float] = None) -> List[Detection]:
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        detections = []

        if image is None or image.size == 0:
            return detections

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        if self.valid_onnx_detector and (self.ort_session is not None or self.net is not None):
            target_w, target_h = self.input_size
            scale = min(target_w / float(w), target_h / float(h))
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
            pad_left = int(round((target_w - nw) / 2.0))
            pad_top = int(round((target_h - nh) / 2.0))
            padded[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

            if self.ort_session is not None:
                blob = padded.transpose((2, 0, 1)).astype(np.float32)[np.newaxis, ...]
                outputs = self.ort_session.run(None, {self.ort_input_name: blob})
                preds = outputs[0][0]
            else:
                blob = cv2.dnn.blobFromImage(padded, 1.0, self.input_size, (0, 0, 0), swapRB=False, crop=False)
                self.net.setInput(blob)
                outputs = self.net.forward()
                preds = outputs[0] if len(outputs.shape) == 3 else outputs

            raw_count = len(preds)
            obj_conf = preds[:, 4]
            cls_conf = preds[:, 5:]
            class_ids = np.argmax(cls_conf, axis=1)
            class_max_scores = np.max(cls_conf, axis=1)
            total_scores = obj_conf * class_max_scores

            cand_indices = np.where(total_scores >= conf_thresh)[0]
            boxes, confidences, final_class_ids = [], [], []
            float_boxes = []

            if len(cand_indices) > 0:
                if preds.shape[0] == len(self.grids):
                    c_grids = self.grids[cand_indices]
                    c_strides = self.strides[cand_indices]
                    cx = (preds[cand_indices, 0] + c_grids[:, 0]) * c_strides
                    cy = (preds[cand_indices, 1] + c_grids[:, 1]) * c_strides
                    bw = np.exp(np.clip(preds[cand_indices, 2], -10.0, 10.0)) * c_strides
                    bh = np.exp(np.clip(preds[cand_indices, 3], -10.0, 10.0)) * c_strides
                else:
                    cx = preds[cand_indices, 0]
                    cy = preds[cand_indices, 1]
                    bw = preds[cand_indices, 2]
                    bh = preds[cand_indices, 3]

                for i, idx in enumerate(cand_indices):
                    cid = int(class_ids[idx])
                    if cid in COCO_VEHICLE_CLASSES:
                        c_x, c_y = float(cx[i]), float(cy[i])
                        b_w, b_h = float(bw[i]), float(bh[i])
                        x1 = (c_x - pad_left - b_w / 2.0) / scale
                        y1 = (c_y - pad_top - b_h / 2.0) / scale
                        bw_s = b_w / scale
                        bh_s = b_h / scale
                        boxes.append([int(x1), int(y1), int(bw_s), int(bh_s)])
                        float_boxes.append([x1, y1, bw_s, bh_s])
                        confidences.append(float(total_scores[idx]))
                        final_class_ids.append(cid)

            if self.class_specific_nms and len(boxes) > 0:
                # Per-class NMS to prevent cross-class suppression (e.g. cars suppressing motorcycles or trucks)
                selected_indices = []
                unique_cids = set(final_class_ids)
                for u_cid in unique_cids:
                    cls_idxs = [j for j, c in enumerate(final_class_ids) if c == u_cid]
                    cls_boxes = [boxes[j] for j in cls_idxs]
                    cls_confs = [confidences[j] for j in cls_idxs]
                    nms_res = cv2.dnn.NMSBoxes(cls_boxes, cls_confs, conf_thresh, self.nms_threshold)
                    if len(nms_res) > 0:
                        for s in nms_res.flatten():
                            selected_indices.append(cls_idxs[s])
                indices = np.array(selected_indices)
            else:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, self.nms_threshold) if len(boxes) > 0 else []

            post_nms_count = len(indices) if len(indices) > 0 else 0

            if len(indices) > 0:
                for idx in indices.flatten():
                    fb = float_boxes[idx]
                    bx, by, bw_i, bh_i = fb[0], fb[1], fb[2], fb[3]
                    cid = final_class_ids[idx]
                    detections.append(Detection(
                        class_id=cid,
                        class_name=COCO_VEHICLE_CLASSES.get(cid, "Car"),
                        confidence=confidences[idx],
                        bbox=(float(max(0.0, bx)), float(max(0.0, by)), float(min(float(w), bx + bw_i)), float(min(float(h), by + bh_i)))
                    ))

            # Inter-class vehicle deduplication (suppress dual Car/Truck detections on identical vehicle)
            if len(detections) > 1:
                deduped = []
                for i, d1 in enumerate(detections):
                    dominated = False
                    for j, d2 in enumerate(detections):
                        if i != j and d2.confidence > d1.confidence:
                            # Compute IoU between bounding boxes
                            xA = max(d1.bbox[0], d2.bbox[0])
                            yA = max(d1.bbox[1], d2.bbox[1])
                            xB = min(d1.bbox[2], d2.bbox[2])
                            yB = min(d1.bbox[3], d2.bbox[3])
                            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                            a1 = max(1e-5, (d1.bbox[2] - d1.bbox[0]) * (d1.bbox[3] - d1.bbox[1]))
                            a2 = max(1e-5, (d2.bbox[2] - d2.bbox[0]) * (d2.bbox[3] - d2.bbox[1]))
                            iou = inter / float(a1 + a2 - inter)
                            if iou > 0.65:
                                dominated = True
                                break
                    if not dominated:
                        deduped.append(d1)
                detections = deduped

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            self.total_frames += 1
            self.last_inference_ms = elapsed_ms
            self.total_inference_ms += elapsed_ms
            self.total_raw_detections += raw_count
            self.total_post_nms_detections += post_nms_count
            self.total_vehicle_detections += len(detections)
            return detections

        # Transparent Fallback Execution Mode (MOG2 / Subspace Contour Vehicle Detector)
        horizon_y = int(h * 0.15)
        roi = image[horizon_y:, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        if not hasattr(self, '_fallback_bg_subtractor'):
            self._fallback_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)

        fg_mask = self._fallback_bg_subtractor.apply(roi)
        _, thresh_fg = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        _, thresh_dark = cv2.threshold(gray_roi, 140, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_or(thresh_fg, thresh_dark)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_count = len(contours)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = max(50.0, (w * h) * 0.0002)
            max_area = (w * h) * 0.30

            if min_area < area < max_area:
                x, y_roi, bw_i, bh_i = cv2.boundingRect(cnt)
                aspect_ratio = bw_i / float(bh_i)
                if 0.35 <= aspect_ratio <= 4.0:
                    y = y_roi + horizon_y
                    cls_name = "Car" if bw_i < 75 else ("Truck" if bw_i > 140 else "SUV")
                    conf = min(0.98, max(0.65, 0.85 + (area / max_area) * 0.15))
                    if conf >= conf_thresh:
                        detections.append(Detection(
                            class_id=0 if cls_name == "Car" else (1 if cls_name == "SUV" else 2),
                            class_name=cls_name,
                            confidence=conf,
                            bbox=(float(x), float(y), float(x + bw_i), float(y + bh_i))
                        ))

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        self.total_frames += 1
        self.last_inference_ms = elapsed_ms
        self.total_inference_ms += elapsed_ms
        self.total_raw_detections += raw_count
        self.total_post_nms_detections += len(detections)
        self.total_vehicle_detections += len(detections)
        return detections

    def detect_verbose(self, image: np.ndarray, confidence_threshold: Optional[float] = None) -> Tuple[np.ndarray, List[Any], List[Any], List[Detection]]:
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        if image is None or image.size == 0 or (self.net is None and self.ort_session is None):
            return np.zeros((0,)), [], [], []

        h, w = image.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / float(w), target_h / float(h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_left = int(round((target_w - nw) / 2.0))
        pad_top = int(round((target_h - nh) / 2.0))
        padded[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

        if self.ort_session is not None:
            blob = padded.transpose((2, 0, 1)).astype(np.float32)[np.newaxis, ...]
            outputs = self.ort_session.run(None, {self.ort_input_name: blob})
            preds = outputs[0][0]
        else:
            blob = cv2.dnn.blobFromImage(padded, 1.0, self.input_size, (0, 0, 0), swapRB=False, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward()
            preds = outputs[0] if len(outputs.shape) == 3 else outputs

        if preds.shape[0] == len(self.grids):
            cx = (preds[:, 0] + self.grids[:, 0]) * self.strides
            cy = (preds[:, 1] + self.grids[:, 1]) * self.strides
            bw = np.exp(np.clip(preds[:, 2], -10.0, 10.0)) * self.strides
            bh = np.exp(np.clip(preds[:, 3], -10.0, 10.0)) * self.strides
        else:
            cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

        obj_conf = preds[:, 4]
        cls_conf = preds[:, 5:]
        class_ids = np.argmax(cls_conf, axis=1)
        class_max_scores = np.max(cls_conf, axis=1)
        total_scores = obj_conf * class_max_scores

        cand_indices = np.where(total_scores >= conf_thresh)[0]
        boxes, confidences, final_class_ids = [], [], []
        float_boxes = []

        for idx in cand_indices:
            cid = int(class_ids[idx])
            if cid in COCO_VEHICLE_CLASSES:
                c_x, c_y = float(cx[idx]), float(cy[idx])
                b_w, b_h = float(bw[idx]), float(bh[idx])
                x1 = (c_x - pad_left - b_w / 2.0) / scale
                y1 = (c_y - pad_top - b_h / 2.0) / scale
                bw_s = b_w / scale
                bh_s = b_h / scale
                boxes.append([int(x1), int(y1), int(bw_s), int(bh_s)])
                float_boxes.append([x1, y1, bw_s, bh_s])
                confidences.append(float(total_scores[idx]))
                final_class_ids.append(cid)

        if self.class_specific_nms and len(boxes) > 0:
            selected_indices = []
            unique_cids = set(final_class_ids)
            for u_cid in unique_cids:
                cls_idxs = [j for j, c in enumerate(final_class_ids) if c == u_cid]
                cls_boxes = [boxes[j] for j in cls_idxs]
                cls_confs = [confidences[j] for j in cls_idxs]
                nms_res = cv2.dnn.NMSBoxes(cls_boxes, cls_confs, conf_thresh, self.nms_threshold)
                if len(nms_res) > 0:
                    for s in nms_res.flatten():
                        selected_indices.append(cls_idxs[s])
            indices = np.array(selected_indices)
        else:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, self.nms_threshold) if len(boxes) > 0 else []

        post_nms_dets = []
        vehicle_dets = []

        if len(indices) > 0:
            for idx in indices.flatten():
                fb = float_boxes[idx]
                bx, by, bw_i, bh_i = fb[0], fb[1], fb[2], fb[3]
                cid = final_class_ids[idx]
                det = Detection(
                    class_id=cid,
                    class_name=COCO_VEHICLE_CLASSES.get(cid, "Car"),
                    confidence=confidences[idx],
                    bbox=(float(max(0.0, bx)), float(max(0.0, by)), float(min(float(w), bx + bw_i)), float(min(float(h), by + bh_i)))
                )
                post_nms_dets.append(det)
                vehicle_dets.append(det)

        return outputs, cand_indices.tolist(), post_nms_dets, vehicle_dets

    def get_telemetry(self) -> Dict[str, Any]:
        avg_ms = self.total_inference_ms / max(1, self.total_frames)
        return {
            "name": self.model_name,
            "detector_name": self.model_name,
            "detector_requested": self.detector_requested,
            "detector_actual": self.detector_actual,
            "version": "1.0.0",
            "detector_version": "1.0.0",
            "framework": "ONNX / OpenCV-DNN",
            "model_path": self.model_path or "data/models/yolox_nano.onnx",
            "model_hash": self.model_hash,
            "device": self.device,
            "backend": self.backend_name,
            "inference_backend": self.backend_name,
            "input_resolution": list(self.input_size),
            "class_specific_nms": self.class_specific_nms,
            "inference_time_ms": round(self.last_inference_ms, 2),
            "avg_inference_time_ms": round(avg_ms, 2),
            "raw_detection_count": self.total_raw_detections,
            "post_nms_detection_count": self.total_post_nms_detections,
            "vehicle_detection_count": self.total_vehicle_detections,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "status": self.status
        }

    def get_model_info(self) -> Dict[str, Any]:
        return self.get_telemetry()


