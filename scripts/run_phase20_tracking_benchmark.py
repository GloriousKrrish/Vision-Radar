"""
Phase 20: Ground-Truth Tracking Benchmark Runner.

Executes baseline vs. improved ByteTrack tracking evaluation on official ground-truth datasets:
- DATASETS/Vehicle_Tracking/VNTraffic (1920x1080, 500 frames, 16,036 GT boxes)
- DATASETS/Vehicle_Tracking/AICC22-Custom (1280x720, 200 frames, 1,185 GT boxes)

Outputs:
- reports/phase20/tracking_metrics.json
- reports/phase20/benchmark_config.json
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

# NumPy 2.x compatibility patch for motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visionradar.cv.detector import Detection
from visionradar.cv.tracker import ByteTrackTracker, compute_iou
from visionradar.eval.tracking_eval import (
    parse_groundtruth_txt,
    extract_or_load_detections,
    run_tracker_on_detections,
    evaluate_tracking_benchmark
)


class LegacyBaselineTracker:
    """
    Exact emulation of Phase 19 / legacy tracker for fair, controlled baseline:
    - Static bounding boxes on missed frames (no Kalman prediction)
    - Greedy IoU argmax matching (no bipartite linear assignment)
    - Center distance fallback with 120px radius
    - No duplicate track suppression
    - ID switches incremented without track resurrection
    """
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3, high_threshold: float = 0.40):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.next_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.recently_lost_tracks: Dict[int, Dict[str, Any]] = {}
        self.total_id_switches = 0
        self.total_fragmentations = 0

    def _associate_pool(self, track_ids, pool_dets, matched_tracks, matched_dets):
        avail_tids = [tid for tid in track_ids if tid not in matched_tracks]
        avail_didxs = [i for i in range(len(pool_dets)) if i not in matched_dets]
        if not avail_tids or not avail_didxs:
            return

        iou_mat = np.zeros((len(avail_tids), len(avail_didxs)), dtype=np.float32)
        for i, tid in enumerate(avail_tids):
            t_box = self.active_tracks[tid]["bbox"]
            for j, d_idx in enumerate(avail_didxs):
                iou_mat[i, j] = compute_iou(t_box, pool_dets[d_idx].bbox)

        while True:
            if iou_mat.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            max_iou = iou_mat[max_idx]
            if max_iou < self.iou_threshold:
                break
            t_i, d_j = max_idx
            tid = avail_tids[t_i]
            d_idx = avail_didxs[d_j]

            track_obj = self.active_tracks[tid]
            if track_obj["time_since_update"] > 1:
                self.total_fragmentations += 1
            track_obj["bbox"] = pool_dets[d_idx].bbox
            track_obj["time_since_update"] = 0
            track_obj["hits"] += 1
            matched_tracks.add(tid)
            matched_dets.add(d_idx)
            iou_mat[t_i, :] = -1.0
            iou_mat[:, d_j] = -1.0

        # Center distance fallback
        for tid in avail_tids:
            if tid in matched_tracks:
                continue
            t_box = self.active_tracks[tid]["bbox"]
            tc_x, tc_y = (t_box[0] + t_box[2]) / 2.0, (t_box[1] + t_box[3]) / 2.0
            best_d_idx = -1
            min_dist = 120.0
            for d_idx in avail_didxs:
                if d_idx in matched_dets:
                    continue
                det = pool_dets[d_idx]
                dc_x, dc_y = (det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0
                dist = np.hypot(dc_x - tc_x, dc_y - tc_y)
                if dist < min_dist:
                    min_dist = dist
                    best_d_idx = d_idx
            if best_d_idx >= 0:
                track_obj = self.active_tracks[tid]
                if track_obj["time_since_update"] > 1:
                    self.total_fragmentations += 1
                track_obj["bbox"] = pool_dets[best_d_idx].bbox
                track_obj["time_since_update"] = 0
                track_obj["hits"] += 1
                matched_tracks.add(tid)
                matched_dets.add(best_d_idx)

    def update(self, detections: List[Detection], frame_index: int, timestamp: float):
        for t in self.active_tracks.values():
            t["time_since_update"] += 1

        dets_high = [d for d in detections if d.confidence >= self.high_threshold]
        dets_low = [d for d in detections if d.confidence < self.high_threshold]

        track_ids = list(self.active_tracks.keys())
        matched_tracks = set()
        matched_high_dets = set()
        matched_low_dets = set()

        self._associate_pool(track_ids, dets_high, matched_tracks, matched_high_dets)
        self._associate_pool(track_ids, dets_low, matched_tracks, matched_low_dets)

        for j, det in enumerate(dets_high):
            if j not in matched_high_dets:
                for lost_tid, lost_meta in list(self.recently_lost_tracks.items()):
                    ref_frame = lost_meta.get("lost_at_frame", lost_meta["last_frame"])
                    if frame_index - ref_frame <= self.max_age:
                        iou_lost = compute_iou(lost_meta["last_bbox"], det.bbox)
                        if iou_lost >= 0.35:
                            self.total_id_switches += 1
                            del self.recently_lost_tracks[lost_tid]
                            break

                self.active_tracks[self.next_id] = {
                    "track_id": self.next_id,
                    "bbox": det.bbox,
                    "time_since_update": 0,
                    "hits": 1,
                    "last_frame": frame_index
                }
                self.next_id += 1

        stale_ids = [tid for tid, t in self.active_tracks.items() if t["time_since_update"] > self.max_age]
        for tid in stale_ids:
            stale_trk = self.active_tracks[tid]
            self.recently_lost_tracks[tid] = {
                "last_bbox": stale_trk["bbox"],
                "last_frame": stale_trk.get("last_frame", frame_index),
                "lost_at_frame": frame_index
            }
            del self.active_tracks[tid]

        class DummyTrack:
            def __init__(self, tid, bbox):
                self.track_id = tid
                self.bbox = bbox

        return [DummyTrack(t["track_id"], t["bbox"]) for t in self.active_tracks.values()]


def main():
    print("=" * 70)
    print("VISIONRADAR PHASE 20: GROUND-TRUTH TRACKING BENCHMARK")
    print("=" * 70)

    reports_dir = Path("reports/phase20")
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = "data/models/yolox_nano.onnx"
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    datasets = [
        {
            "name": "VNTraffic",
            "video_path": "DATASETS/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4",
            "gt_txt_path": "DATASETS/Vehicle_Tracking/VNTraffic/VNTraffic_GroundTruth.txt",
            "max_frames": 500,
            "start_frame": 0,
            "frame_offset": 0
        },
        {
            "name": "AICC22-Custom",
            "video_path": "DATASETS/Vehicle_Tracking/AICC22-Custom/AICC22-Custom_Original-video.mp4",
            "gt_txt_path": "DATASETS/Vehicle_Tracking/AICC22-Custom/AICC22-Custom_GroundTruth.txt",
            "max_frames": 200,
            "start_frame": 0,
            "frame_offset": -1
        }
    ]

    benchmark_config = {
        "phase": 20,
        "model_path": model_path,
        "model_sha256": model_hash,
        "detector_input_resolution": [640, 640],
        "conf_threshold": 0.10,
        "high_threshold": 0.40,
        "low_threshold": 0.10,
        "iou_threshold": 0.30,
        "max_age": 30,
        "datasets": [d["name"] for d in datasets],
        "evaluation_protocol": "CLEAR-MOT (Bernardin 2008), IDF1 (Ristani 2016), HOTA (Luiten 2021)",
        "matching_iou_threshold": 0.50
    }

    with open(reports_dir / "benchmark_config.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_config, f, indent=2)

    results = {
        "metadata": benchmark_config,
        "datasets": {}
    }

    for d in datasets:
        dname = d["name"]
        print(f"\n--- Evaluating Dataset: {dname} ---")
        gt_by_frame = parse_groundtruth_txt(d["gt_txt_path"], frame_offset=d.get("frame_offset", 0))
        total_gt_boxes = sum(len(boxes) for boxes in gt_by_frame.values())
        print(f"Loaded ground truth: {len(gt_by_frame)} frames, {total_gt_boxes} boxes.")

        print(f"Extracting detections from {d['video_path']} (max {d['max_frames']} frames)...")
        dets_per_frame = extract_or_load_detections(
            d["video_path"],
            max_frames=d["max_frames"],
            model_path=model_path,
            conf_thresh=0.10,
            cache_path=f"reports/phase20/cache_{dname}_dets.json"
        )
        print(f"Extracted detections for {len(dets_per_frame)} frames.")

        # 1. Run Baseline Tracker
        print("Running Baseline Tracker (Static box propagation, greedy matching)...")
        baseline_tracker = LegacyBaselineTracker(max_age=30, iou_threshold=0.3, high_threshold=0.4)
        base_preds, base_timing = run_tracker_on_detections(
            baseline_tracker, dets_per_frame, start_frame_idx=d["start_frame"]
        )
        base_metrics = evaluate_tracking_benchmark(gt_by_frame, base_preds, iou_threshold=0.5)
        base_metrics.update(base_timing)

        # 2. Run Improved Tracker
        print("Running Improved Tracker (8-state Kalman Filter, bipartite matching, duplicate suppression)...")
        improved_tracker = ByteTrackTracker(max_age=30, iou_threshold=0.3, high_threshold=0.4, low_threshold=0.1)
        imp_preds, imp_timing = run_tracker_on_detections(
            improved_tracker, dets_per_frame, start_frame_idx=d["start_frame"]
        )
        imp_metrics = evaluate_tracking_benchmark(gt_by_frame, imp_preds, iou_threshold=0.5)
        imp_metrics.update(imp_timing)

        # Print comparison
        print(f"\nResults for {dname}:")
        print(f"{'Metric':<25} | {'Baseline':<15} | {'Improved':<15} | {'Delta':<15}")
        print("-" * 75)
        for k in ["MOTA", "IDF1", "IDP", "IDR", "HOTA", "DetA", "AssA", "id_switches", "fragmentations", "id_transfers", "mostly_tracked", "mean_latency_ms", "p95_latency_ms", "tracker_fps"]:
            b_val = base_metrics[k]
            i_val = imp_metrics[k]
            if isinstance(b_val, float):
                diff = i_val - b_val
                sign = "+" if diff >= 0 else ""
                print(f"{k:<25} | {b_val:<15.4f} | {i_val:<15.4f} | {sign}{diff:<15.4f}")
            else:
                diff = i_val - b_val
                sign = "+" if diff >= 0 else ""
                print(f"{k:<25} | {b_val:<15} | {i_val:<15} | {sign}{diff:<15}")

        results["datasets"][dname] = {
            "ground_truth": {
                "num_frames": len(gt_by_frame),
                "total_boxes": total_gt_boxes
            },
            "baseline": base_metrics,
            "improved": imp_metrics
        }

    with open(reports_dir / "tracking_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark completed successfully. Saved metrics to {reports_dir / 'tracking_metrics.json'}")


if __name__ == "__main__":
    main()
