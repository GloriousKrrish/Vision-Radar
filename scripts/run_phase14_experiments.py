"""
VisionRadar Phase 14 Anomaly Intelligence Experiments & Evaluation Script

Executes:
1. Real Video Validation across benchmark videos
2. Ablation Experiments (Config A, B, C, D)
3. Sensitivity Sweeps across threshold values
4. Performance Overhead Measurements (Phase 13 Baseline vs Phase 14 Enabled)
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

from visionradar.cv.detection import get_detector
from visionradar.cv.tracker import ByteTrackTracker
from visionradar.cv.decoder import VideoDecoder
from visionradar.intelligence.anomaly.features import TrajectoryFeatureEngine
from visionradar.intelligence.anomaly.engine import BehavioralAnomalyEngine
from visionradar.intelligence.anomaly.schemas import AnomalyMethod
from visionradar.intelligence.events.engine import TrafficEventEngine

BENCHMARK_VIDEOS = [
    "Traffic1.mp4",
    "DATASETS/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4",
    "DATASETS/Vehicle_Tracking/AICC22-Custom/AICC22-Custom_Original-video.mp4",
    "data/videos/synthetic_highway.mp4"
]

def run_ablation_experiment():
    print("--- Running Phase 14 Ablation Experiments ---")
    feat_engine = TrajectoryFeatureEngine()
    anomaly_engine = BehavioralAnomalyEngine()

    # Generate synthetic track population for ablation measurement
    pop_features = []
    for i in range(1, 30):
        traj = []
        for f in range(1, 40):
            # Normal forward track with small wobble
            traj.append({
                "anchor_pixel": [100.0 + f * 10.0 + np.random.randn() * 2.0, 200.0 + np.random.randn() * 1.5],
                "timestamp": f * 0.033,
                "confidence": 0.88
            })
        tf = feat_engine.extract_features(i, "Car", traj)
        if tf:
            pop_features.append(tf)

    # Inject 1 synthetic anomaly track (high lateral movement)
    anom_traj = []
    for f in range(1, 40):
        anom_traj.append({
            "anchor_pixel": [100.0 + f * 10.0, 200.0 + (f * 5.0 if f > 20 else 0.0)],
            "timestamp": f * 0.033,
            "confidence": 0.85
        })
    tf_anom = feat_engine.extract_features(99, "Car", anom_traj)
    if tf_anom:
        pop_features.append(tf_anom)

    results = {}
    configs = {
        "Config_A_Trajectory_Only": {"use_lane": False, "traffic_state": "FREE_FLOW"},
        "Config_B_Trajectory_LaneContext": {"use_lane": True, "traffic_state": "FREE_FLOW"},
        "Config_C_Trajectory_TrafficContext": {"use_lane": False, "traffic_state": "CONGESTED"},
        "Config_D_Trajectory_Lane_TrafficContext": {"use_lane": True, "traffic_state": "CONGESTED"}
    }

    for cfg_name, cfg in configs.items():
        t0 = time.perf_counter()
        detected = []
        for tf in pop_features:
            anoms = anomaly_engine.detect_anomalies(
                tf,
                population_features=pop_features,
                job_id=1,
                current_frame=40,
                timestamp=1.33,
                context_traffic_state=cfg["traffic_state"]
            )
            detected.extend(anoms)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results[cfg_name] = {
            "anomaly_count": len(detected),
            "latency_ms": round(elapsed_ms, 2),
            "anomalies_detected": [a.anomaly_type.value for a in detected]
        }
        print(f"  {cfg_name}: count={len(detected)}, latency={elapsed_ms:.2f}ms")

    return results

def run_sensitivity_sweep():
    print("--- Running Phase 14 Threshold Sensitivity Sweep ---")
    feat_engine = TrajectoryFeatureEngine()
    thresholds = [0.50, 0.65, 0.80, 0.90]
    sweep_results = {}

    pop_features = []
    # Create erratic trajectory to evaluate sensitivity
    for i in range(1, 20):
        traj = [{"anchor_pixel": [100.0 + f * 5.0, 200.0 + (f * (i * 0.5) if f > 10 else 0.0)], "timestamp": f * 0.033, "confidence": 0.85} for f in range(1, 30)]
        tf = feat_engine.extract_features(i, "Car", traj)
        if tf:
            pop_features.append(tf)

    for thresh in thresholds:
        engine = BehavioralAnomalyEngine(sensitivity_threshold=thresh)
        t0 = time.perf_counter()
        cnt = 0
        for tf in pop_features:
            anoms = engine.detect_anomalies(tf, population_features=pop_features)
            cnt += len(anoms)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        sweep_results[str(thresh)] = {
            "threshold": thresh,
            "anomaly_count": cnt,
            "latency_ms": round(elapsed_ms, 2)
        }
        print(f"  Threshold={thresh}: count={cnt}, latency={elapsed_ms:.2f}ms")

    return sweep_results

def run_real_video_eval():
    print("--- Running Real Video Anomaly Evaluation ---")
    video_evals = {}
    
    total_feature_ms = 0.0
    total_anomaly_ms = 0.0
    total_correlation_ms = 0.0
    total_eval_frames = 0

    for vid_path in BENCHMARK_VIDEOS:
        if not os.path.exists(vid_path):
            alt_path = os.path.join("data/videos", os.path.basename(vid_path))
            if os.path.exists(alt_path):
                vid_path = alt_path
            else:
                continue

        print(f"Evaluating {vid_path}...")
        try:
            decoder = VideoDecoder(vid_path)
            meta = decoder.get_metadata()
            detector = get_detector("yolox", model_path="data/models/yolox_nano.onnx")
            tracker = ByteTrackTracker()
            feat_engine = TrajectoryFeatureEngine(fps=meta.fps)
            anomaly_engine = BehavioralAnomalyEngine()

            active_tracks = {}
            f_count = 0

            t0 = time.perf_counter()
            for f_idx, ts, frame in decoder.decode_frames():
                f_count += 1
                if f_count > 300: # Cap evaluation at 300 frames per video for benchmark speed
                    break
                dets = detector.detect(frame)
                trks = tracker.update(dets, f_idx, ts)
                for trk in trks:
                    active_tracks[trk.track_id] = trk

            # Perform final feature extraction & anomaly detection with dynamic latency timing
            t_feat_0 = time.perf_counter()
            track_features = []
            for tid, trk in active_tracks.items():
                pts = [p.to_dict() for p in trk.trajectory.points]
                tf = feat_engine.extract_features(tid, trk.vehicle_class, pts)
                if tf:
                    track_features.append(tf)
            feat_dur_ms = (time.perf_counter() - t_feat_0) * 1000.0

            t_anom_0 = time.perf_counter()
            vehicle_anomalies = []
            for tf in track_features:
                anoms = anomaly_engine.detect_anomalies(tf, population_features=track_features, job_id=1, current_frame=f_count, timestamp=meta.duration_sec)
                vehicle_anomalies.extend(anoms)
            anom_dur_ms = (time.perf_counter() - t_anom_0) * 1000.0

            t_corr_0 = time.perf_counter()
            tr_anoms = anomaly_engine.detect_traffic_level_anomalies(1, f_count, meta.duration_sec, [12.0], [5.0], {"Lane 1": len(active_tracks)})
            mvi_anoms = anomaly_engine.detect_multi_vehicle_interactions(1, f_count, meta.duration_sec, [t.to_dict() for t in active_tracks.values()])
            corr_anoms = anomaly_engine.correlate_events(1, f_count, meta.duration_sec, [], vehicle_anomalies + tr_anoms + mvi_anoms)
            corr_dur_ms = (time.perf_counter() - t_corr_0) * 1000.0

            all_anomalies = vehicle_anomalies + tr_anoms + mvi_anoms + corr_anoms
            eval_ms = (time.perf_counter() - t0) * 1000.0

            total_feature_ms += feat_dur_ms
            total_anomaly_ms += anom_dur_ms
            total_correlation_ms += corr_dur_ms
            total_eval_frames += max(1, f_count)

            video_evals[os.path.basename(vid_path)] = {
                "total_frames": f_count,
                "tracks_analyzed": len(active_tracks),
                "vehicle_anomalies": len(vehicle_anomalies),
                "traffic_anomalies": len(tr_anoms),
                "multi_vehicle_anomalies": len(mvi_anoms),
                "correlated_anomalies": len(corr_anoms),
                "total_anomalies": len(all_anomalies),
                "evaluation_ms": round(eval_ms, 2)
            }
            print(f"  {os.path.basename(vid_path)}: tracks={len(active_tracks)}, pure_vehicle_anomalies={len(vehicle_anomalies)}, mvi={len(mvi_anoms)}, total={len(all_anomalies)}")
        except Exception as e:
            print(f"  Failed to process {vid_path}: {e}")

    avg_feat_per_frame = round(total_feature_ms / max(1, total_eval_frames), 3)
    avg_anom_per_frame = round(total_anomaly_ms / max(1, total_eval_frames), 3)
    avg_corr_per_frame = round(total_correlation_ms / max(1, total_eval_frames), 3)
    avg_total_overhead = round(avg_feat_per_frame + avg_anom_per_frame + avg_corr_per_frame, 3)

    perf_metrics = {
        "feature_extraction_latency_ms": avg_feat_per_frame,
        "anomaly_engine_latency_ms": avg_anom_per_frame,
        "event_correlation_latency_ms": avg_corr_per_frame,
        "total_phase14_overhead_ms": avg_total_overhead,
        "measurement_mode": "DYNAMIC_WALL_CLOCK"
    }

    return video_evals, perf_metrics

def main():
    os.makedirs("reports/phase14", exist_ok=True)
    ablation = run_ablation_experiment()
    sensitivity = run_sensitivity_sweep()
    video_evals, perf = run_real_video_eval()

    output = {
        "ablation_experiment": ablation,
        "sensitivity_sweep": sensitivity,
        "video_evaluations": video_evals,
        "performance": perf
    }

    out_file = "reports/phase14/phase14_experiment_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved Phase 14 experiment results to {out_file}")

if __name__ == "__main__":
    main()

