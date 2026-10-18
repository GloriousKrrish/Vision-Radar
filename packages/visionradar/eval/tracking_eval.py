"""
VisionRadar Tracking Evaluation Engine (Phase 20 Compliant).

Provides ground-truth evaluation for multi-object vehicle tracking.
Computes:
- CLEAR-MOT (MOTA, MOTP, ID Switches, Fragmentations, False Positives, Misses)
- ID Metrics (IDF1, IDP, IDR, ID Transfers)
- HOTA (HOTA, DetA, AssA)
- Latency profiling (Mean, p95, Tracker FPS)
"""

import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# NumPy 2.x compatibility patch for motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm
from visionradar.cv.detector import Detection
from visionradar.cv.tracker import ByteTrackTracker, compute_iou
from visionradar.perception.inference_engine import InferenceEngine
from visionradar.perception.preprocess import letterbox_preprocess
from visionradar.perception.postprocess import postprocess_yolox_output


def compute_hota(
    gt_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
    pred_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
    iou_thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Computes HOTA, DetA, AssA according to Luiten et al., IJCV 2021.
    """
    if iou_thresholds is None:
        iou_thresholds = [float(x) for x in np.arange(0.05, 0.99, 0.05)]

    all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))
    hota_list = []
    deta_list = []
    assa_list = []

    for alpha in iou_thresholds:
        # Step 1: Compute per-frame bipartite detection matches
        # matches: list of (frame, pred_id, gt_id)
        matches = []
        n_gt_total = sum(len(gt_by_frame.get(f, [])) for f in all_frames)
        n_pr_total = sum(len(pred_by_frame.get(f, [])) for f in all_frames)

        for f in all_frames:
            gts = gt_by_frame.get(f, [])
            prs = pred_by_frame.get(f, [])
            if not gts or not prs:
                continue

            cost_mat = np.zeros((len(prs), len(gts)), dtype=float)
            for pi, (pid, pbox) in enumerate(prs):
                for gi, (gid, gbox) in enumerate(gts):
                    iou = compute_iou(pbox, gbox)
                    cost_mat[pi, gi] = iou

            # Greedy bipartite matching at threshold alpha
            temp_mat = cost_mat.copy()
            while True:
                if temp_mat.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(temp_mat), temp_mat.shape)
                max_val = temp_mat[max_idx]
                if max_val < alpha:
                    break
                p_i, g_j = max_idx
                matches.append((f, prs[p_i][0], gts[g_j][0]))
                temp_mat[p_i, :] = -1.0
                temp_mat[:, g_j] = -1.0

        n_tp = len(matches)
        n_fp = n_pr_total - n_tp
        n_fn = n_gt_total - n_tp

        denom_deta = n_tp + n_fp + n_fn
        deta = float(n_tp / denom_deta) if denom_deta > 0 else 0.0

        # Step 2: Compute Association Accuracy (AssA)
        if n_tp == 0:
            assa = 0.0
            hota = 0.0
        else:
            # Map of (pred_id, gt_id) occurrences
            pr_counts: Dict[int, int] = {}
            gt_counts: Dict[int, int] = {}
            pair_counts: Dict[Tuple[int, int], int] = {}

            for _, pid, gid in matches:
                pr_counts[pid] = pr_counts.get(pid, 0) + 1
                gt_counts[gid] = gt_counts.get(gid, 0) + 1
                pair = (pid, gid)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            ass_scores = []
            for _, pid, gid in matches:
                tpa = pair_counts[(pid, gid)]
                fpa = pr_counts[pid] - tpa
                fna = gt_counts[gid] - tpa
                denom_a = tpa + fpa + fna
                ass_scores.append(tpa / denom_a if denom_a > 0 else 0.0)

            assa = float(np.mean(ass_scores))
            hota = float(np.sqrt(deta * assa))

        hota_list.append(hota)
        deta_list.append(deta)
        assa_list.append(assa)

    return {
        "HOTA": float(np.mean(hota_list)),
        "DetA": float(np.mean(deta_list)),
        "AssA": float(np.mean(assa_list))
    }


def parse_groundtruth_txt(txt_path: str, frame_offset: int = 0) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    """
    Parses standard MOT format ground truth text file:
    frame, id, left, top, width, height, conf, class, visibility
    Applies frame_offset to align with 0-indexed video frames if dataset is 1-indexed.
    """
    gt_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                frame_idx = int(parts[0]) + frame_offset
                tid = int(parts[1])
                left, top, w, h = map(float, parts[2:6])
                bbox = (left, top, left + w, top + h)
                if frame_idx not in gt_by_frame:
                    gt_by_frame[frame_idx] = []
                gt_by_frame[frame_idx].append((tid, bbox))
    return gt_by_frame


def extract_or_load_detections(
    video_path: str,
    max_frames: int = 500,
    model_path: str = "data/models/yolox_nano.onnx",
    conf_thresh: float = 0.10,
    nms_thresh: float = 0.45,
    cache_path: Optional[str] = None
) -> List[List[Detection]]:
    """
    Runs production YOLOX-Nano inference on video frames to get raw detections.
    Supports file caching for reproducible, instant re-evaluation.
    """
    if cache_path and Path(cache_path).exists():
        import json
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        all_detections = []
        for frame_dets in cached_data:
            all_detections.append([
                Detection(
                    class_id=d["class_id"],
                    class_name=d["class_name"],
                    confidence=float(d["confidence"]),
                    bbox=tuple(d["bbox"])
                )
                for d in frame_dets
            ])
        return all_detections

    engine = InferenceEngine(model_path=model_path, input_size=(640, 640), intra_op_threads=8)
    cap = cv2.VideoCapture(video_path)
    all_detections: List[List[Detection]] = []
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        blob, scale, pad = letterbox_preprocess(frame, (640, 640))
        outputs, _ = engine.infer(blob)
        raw_dets = postprocess_yolox_output(
            outputs,
            frame_id=frame_idx,
            source_shape=(w, h),
            confidence_threshold=conf_thresh,
            nms_threshold=nms_thresh,
            scale=scale,
            pad=pad
        )
        dets = [
            Detection(
                class_id=2,
                class_name=d.vehicle_class.value if hasattr(d.vehicle_class, "value") else str(d.vehicle_class),
                confidence=float(d.confidence),
                bbox=d.bbox.to_tuple()
            )
            for d in raw_dets
        ]
        all_detections.append(dets)
        frame_idx += 1

    cap.release()

    if cache_path:
        import json
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": list(d.bbox)
                }
                for d in frame_dets
            ]
            for frame_dets in all_detections
        ]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

    return all_detections


def run_tracker_on_detections(
    tracker: Any,
    detections_per_frame: List[List[Detection]],
    start_frame_idx: int = 0
) -> Tuple[Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]], Dict[str, float]]:
    """
    Runs tracker on pre-extracted detections, recording predictions and latency.
    """
    preds_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    latencies_ms: List[float] = []

    for idx, dets in enumerate(detections_per_frame):
        frame_idx = start_frame_idx + idx
        timestamp = frame_idx * (1.0 / 30.0)

        t0 = time.perf_counter()
        active_tracks = tracker.update(dets, frame_index=frame_idx, timestamp=timestamp)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        preds_by_frame[frame_idx] = [
            (t.track_id, t.bbox) for t in active_tracks
        ]

    timing_stats = {
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),
        "tracker_fps": float(1000.0 / np.mean(latencies_ms)) if np.mean(latencies_ms) > 0 else 0.0
    }

    return preds_by_frame, timing_stats


def evaluate_tracking_benchmark(
    gt_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
    pred_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Runs full CLEAR-MOT, IDF1, and HOTA evaluation using motmetrics and custom HOTA.
    """
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))

    for f in all_frames:
        gts = gt_by_frame.get(f, [])
        preds = pred_by_frame.get(f, [])

        gt_ids = [g[0] for g in gts]
        pred_ids = [p[0] for p in preds]

        if not gts or not preds:
            cost_mat = np.empty((len(gt_ids), len(pred_ids)))
            cost_mat.fill(np.nan)
        else:
            cost_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
            for gi, (_, gbox) in enumerate(gts):
                for pi, (_, pbox) in enumerate(preds):
                    iou = compute_iou(gbox, pbox)
                    if iou >= iou_threshold:
                        cost_mat[gi, pi] = 1.0 - iou
                    else:
                        cost_mat[gi, pi] = np.nan

        acc.update(gt_ids, pred_ids, cost_mat)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames", "obj_frequencies", "num_matches", "num_switches", "num_false_positives",
            "num_misses", "mota", "motp", "precision", "recall",
            "idtp", "idfp", "idfn", "idf1", "idp", "idr",
            "mostly_tracked", "partially_tracked", "mostly_lost", "num_fragmentations", "num_transfer"
        ],
        name="tracking_results"
    )

    hota_res = compute_hota(gt_by_frame, pred_by_frame)

    res = {
        "num_frames": int(summary["num_frames"].iloc[0]),
        "recall": float(summary["recall"].iloc[0]),
        "precision": float(summary["precision"].iloc[0]),
        "MOTA": float(summary["mota"].iloc[0]),
        "MOTP": float(summary["motp"].iloc[0]),
        "IDF1": float(summary["idf1"].iloc[0]),
        "IDP": float(summary["idp"].iloc[0]),
        "IDR": float(summary["idr"].iloc[0]),
        "id_switches": int(summary["num_switches"].iloc[0]),
        "fragmentations": int(summary["num_fragmentations"].iloc[0]),
        "id_transfers": int(summary["num_transfer"].iloc[0]),
        "mostly_tracked": int(summary["mostly_tracked"].iloc[0]),
        "partially_tracked": int(summary["partially_tracked"].iloc[0]),
        "mostly_lost": int(summary["mostly_lost"].iloc[0]),
        "num_false_positives": int(summary["num_false_positives"].iloc[0]),
        "num_misses": int(summary["num_misses"].iloc[0]),
        "matches": int(summary["num_matches"].iloc[0]),
        "HOTA": hota_res["HOTA"],
        "DetA": hota_res["DetA"],
        "AssA": hota_res["AssA"]
    }

    return res
