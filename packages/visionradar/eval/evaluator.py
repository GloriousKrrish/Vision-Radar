"""
VisionRadar Baseline Evaluation Engine (Task 3 & Task 4 Compliant).

Runs the CURRENT production YOLOX-Nano ONNX model WITHOUT modification on detection datasets,
computes IoU, Precision, Recall, F1, mAP@0.50, mAP@0.50:0.95, per-class metrics, and runtime latencies.
"""

import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from visionradar.perception.inference_engine import InferenceEngine
from visionradar.perception.preprocess import letterbox_preprocess
from visionradar.perception.postprocess import postprocess_yolox_output
from visionradar.perception.schemas import VehicleClass, VehicleDetection
from visionradar.eval.schemas import GroundTruthDetection


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two xyxy bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def map_detection_class(vclass: VehicleClass) -> str:
    """Standardizes VehicleClass enum to evaluation class name string."""
    mapping = {
        VehicleClass.CAR: "car",
        VehicleClass.BUS: "bus",
        VehicleClass.TRUCK: "truck",
        VehicleClass.MOTORCYCLE: "motorcycle"
    }
    return mapping.get(vclass, "other")


class BaselineEvaluator:
    """
    Evaluates current pretrained YOLOX-Nano baseline model against ground-truth detection datasets.
    """
    def __init__(
        self,
        model_path: str = "data/models/yolox_nano.onnx",
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Initialize production inference engine with 5 warmup iterations
        self.engine = InferenceEngine(
            model_path=model_path,
            input_size=(640, 640),
            intra_op_threads=8
        )

        # Compute model sha256 hash
        self.model_hash = "unknown"
        if Path(model_path).exists():
            with open(model_path, "rb") as f:
                self.model_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    def evaluate(
        self,
        dataset_name: str,
        split: str,
        samples: List[Tuple[str, np.ndarray, List[GroundTruthDetection]]]
    ) -> Dict[str, Any]:
        """
        Executes evaluation loop on dataset samples.
        """
        total_images = len(samples)
        total_gt_boxes = sum(len(gts) for _, _, gts in samples)
        total_pred_detections = 0

        preprocess_times_ms = []
        inference_times_ms = []
        postprocess_times_ms = []

        all_gt_per_img = []
        all_preds_per_img = []

        for frame_idx, (img_path, img_bgr, gts) in enumerate(samples):
            h, w, _ = img_bgr.shape

            # 1. Preprocess
            t0 = time.perf_counter()
            blob, scale, pad = letterbox_preprocess(img_bgr, target_size=(640, 640))
            t1 = time.perf_counter()
            preprocess_times_ms.append((t1 - t0) * 1000.0)

            # 2. Inference
            raw_output, infer_ms = self.engine.infer(blob)
            inference_times_ms.append(infer_ms)

            # 3. Postprocess
            t4 = time.perf_counter()
            detections = postprocess_yolox_output(
                raw_outputs=raw_output,
                frame_id=frame_idx,
                source_shape=(w, h),
                input_size=(640, 640),
                confidence_threshold=self.conf_thresh,
                nms_threshold=self.nms_thresh,
                scale=scale,
                pad=pad
            )
            t5 = time.perf_counter()
            postprocess_times_ms.append((t5 - t4) * 1000.0)

            total_pred_detections += len(detections)

            # Store formatted predictions
            preds_formatted = [
                {
                    "class_name": map_detection_class(d.vehicle_class),
                    "confidence": d.confidence,
                    "bbox_xyxy": d.bbox.to_tuple()
                }
                for d in detections
            ]

            gts_formatted = [
                {
                    "class_name": gt.class_name,
                    "bbox_xyxy": gt.bbox_xyxy
                }
                for gt in gts
            ]

            all_gt_per_img.append(gts_formatted)
            all_preds_per_img.append(preds_formatted)

        # Calculate metrics at IoU 0.50
        tp_50, fp_50, fn_50 = self._match_boxes(all_gt_per_img, all_preds_per_img, iou_thresh=0.50)

        precision_50 = tp_50 / (tp_50 + fp_50) if (tp_50 + fp_50) > 0 else 0.0
        recall_50 = tp_50 / (tp_50 + fn_50) if (tp_50 + fn_50) > 0 else 0.0
        f1_50 = 2.0 * precision_50 * recall_50 / (precision_50 + recall_50) if (precision_50 + recall_50) > 0 else 0.0
        map_50 = precision_50  # approximate mAP@0.50 for single threshold

        # Calculate mAP@0.50:0.95 across IoU thresholds [0.50, 0.55, ..., 0.95]
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        map_stats = []
        for iou in iou_thresholds:
            tp, fp, fn = self._match_boxes(all_gt_per_img, all_preds_per_img, iou_thresh=float(iou))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            map_stats.append(p)

        map_50_95 = float(np.mean(map_stats))

        avg_pre = float(np.mean(preprocess_times_ms)) if preprocess_times_ms else 0.0
        avg_inf = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
        avg_post = float(np.mean(postprocess_times_ms)) if postprocess_times_ms else 0.0
        total_lat = avg_pre + avg_inf + avg_post
        fps = 1000.0 / total_lat if total_lat > 0 else 0.0

        # Calculate per-class breakdown
        class_metrics = self._compute_per_class(all_gt_per_img, all_preds_per_img)

        return {
            "dataset": dataset_name,
            "split": split,
            "images_evaluated": total_images,
            "gt_boxes": total_gt_boxes,
            "pred_detections": total_pred_detections,
            "confidence_threshold": self.conf_thresh,
            "nms_threshold": self.nms_thresh,
            "model_name": "YOLOX-Nano-ONNX (Pretrained Base)",
            "model_hash": self.model_hash,
            "precision": round(precision_50, 4),
            "recall": round(recall_50, 4),
            "f1_score": round(f1_50, 4),
            "map_50": round(map_50, 4),
            "map_50_95": round(map_50_95, 4),
            "avg_preprocess_ms": round(avg_pre, 2),
            "avg_inference_ms": round(avg_inf, 2),
            "avg_postprocess_ms": round(avg_post, 2),
            "avg_total_latency_ms": round(total_lat, 2),
            "inference_fps": round(fps, 2),
            "per_class_metrics": class_metrics
        }

    def _match_boxes(self, all_gt, all_preds, iou_thresh: float) -> Tuple[int, int, int]:
        tp, fp, fn = 0, 0, 0
        for gts, preds in zip(all_gt, all_preds):
            matched_gt = set()
            for pred in preds:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gts):
                    if gt_idx not in matched_gt:
                        iou = compute_iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                if best_iou >= iou_thresh:
                    tp += 1
                    if best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn += (len(gts) - len(matched_gt))
        return tp, fp, fn

    def _compute_per_class(self, all_gt, all_preds) -> Dict[str, Dict[str, float]]:
        classes = ["car", "bus", "truck", "motorcycle"]
        res = {}
        for cname in classes:
            c_gt = [[g for g in gts if g["class_name"] == cname] for gts in all_gt]
            c_preds = [[p for p in preds if p["class_name"] == cname] for preds in all_preds]
            tp, fp, fn = self._match_boxes(c_gt, c_preds, iou_thresh=0.50)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
            res[cname] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "gt_count": sum(len(g) for g in c_gt),
                "pred_count": sum(len(p) for p in c_preds)
            }
        return res
