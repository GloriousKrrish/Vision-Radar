"""
VisionRadar Perception Engine — Metrics Collection Module

Tracks rolling-window stage latencies (p50, p95, p99, max), capture vs processed FPS,
frame drop counts, drop reasons, active track count, and first box latency.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from visionradar.perception.schemas import PerceptionMetrics, StageLatency


class PerceptionMetricsCollector:
    """
    Rolling-window metrics collector for perception pipeline stage performance.
    """
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history

        # Timestamps for FPS calculation
        self.capture_timestamps = deque(maxlen=max_history)
        self.processed_timestamps = deque(maxlen=max_history)

        # Stage latency deques (ms)
        self.latencies: Dict[str, deque] = {
            "preprocess": deque(maxlen=max_history),
            "inference": deque(maxlen=max_history),
            "postprocess": deque(maxlen=max_history),
            "tracking": deque(maxlen=max_history),
            "total": deque(maxlen=max_history)
        }

        self.frames_processed = 0
        self.frames_dropped = 0
        self.drop_reasons: Dict[str, int] = {
            "queue_full_capture": 0,
            "queue_full_broadcast": 0,
            "decode_error": 0,
            "inference_timeout": 0
        }
        self.first_box_latency_ms: Optional[float] = None
        self.t_start_pipeline: Optional[float] = None

    def record_capture(self, capture_ts: float):
        if self.t_start_pipeline is None:
            self.t_start_pipeline = capture_ts
        self.capture_timestamps.append(capture_ts)

    def record_drop(self, reason: str):
        self.frames_dropped += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1

    def record_frame_result(
        self,
        capture_ts: float,
        result_ts: float,
        stage_timings_ms: Dict[str, float],
        has_detections: bool = False
    ):
        self.frames_processed += 1
        self.processed_timestamps.append(result_ts)

        total_ms = (result_ts - capture_ts) * 1000.0
        self.latencies["total"].append(total_ms)

        for stage in ("preprocess", "inference", "postprocess", "tracking"):
            if stage in stage_timings_ms:
                self.latencies[stage].append(stage_timings_ms[stage])

        if has_detections and self.first_box_latency_ms is None and self.t_start_pipeline is not None:
            self.first_box_latency_ms = (result_ts - self.t_start_pipeline) * 1000.0

    def get_snapshot(self, window: str = "last_100", active_track_count: int = 0) -> PerceptionMetrics:
        """Alias for get_metrics accepting window string ('last_100' or 'last_1000')."""
        try:
            w_size = int(window.replace("last_", ""))
        except Exception:
            w_size = 100
        return self.get_metrics(window_size=w_size, active_track_count=active_track_count)

    def get_metrics(self, window_size: int = 100, active_track_count: int = 0) -> PerceptionMetrics:
        """Computes PerceptionMetrics snapshot for the given rolling window size."""
        w_label = f"last_{window_size}"

        # Calculate capture FPS
        fps_cap = 0.0
        if len(self.capture_timestamps) >= 2:
            caps = list(self.capture_timestamps)[-window_size:]
            if len(caps) >= 2 and (caps[-1] - caps[0]) > 0:
                fps_cap = (len(caps) - 1) / (caps[-1] - caps[0])

        # Calculate processed FPS
        fps_proc = 0.0
        if len(self.processed_timestamps) >= 2:
            procs = list(self.processed_timestamps)[-window_size:]
            if len(procs) >= 2 and (procs[-1] - procs[0]) > 0:
                fps_proc = (len(procs) - 1) / (procs[-1] - procs[0])

        # Calculate per-stage percentiles
        stage_stats: Dict[str, StageLatency] = {}
        for stage, deq in self.latencies.items():
            vals = list(deq)[-window_size:]
            if vals:
                p50 = float(np.percentile(vals, 50))
                p95 = float(np.percentile(vals, 95))
                p99 = float(np.percentile(vals, 99))
                max_v = float(np.max(vals))
            else:
                p50, p95, p99, max_v = 0.0, 0.0, 0.0, 0.0

            stage_stats[stage] = StageLatency(p50=p50, p95=p95, p99=p99, max=max_v)

        return PerceptionMetrics(
            window=w_label,
            fps_capture=fps_cap,
            fps_processed=fps_proc,
            stage_latency_ms=stage_stats,
            frames_processed=self.frames_processed,
            frames_dropped=self.frames_dropped,
            drop_reasons=dict(self.drop_reasons),
            active_track_count=active_track_count,
            first_box_latency_ms=self.first_box_latency_ms
        )


# Global singleton instance for app-wide metrics collection
global_metrics_collector = PerceptionMetricsCollector()
