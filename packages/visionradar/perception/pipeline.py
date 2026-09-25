"""
VisionRadar Perception Engine — Pipeline Orchestrator

Sequences per-frame processing stages (dequeue -> preprocess -> inference -> postprocess
-> tracker -> lifecycle -> FrameResult assembly -> broadcast push) inside a single dedicated thread.
Times every stage independently with time.perf_counter().
"""

import time
import queue
import asyncio
import threading
import logging
import numpy as np
from typing import Optional, Dict, Any

from visionradar.perception.schemas import FrameResult, PerceptionMetrics
from visionradar.perception.capture import VideoSource, FrameItem
from visionradar.perception.preprocess import letterbox_preprocess
from visionradar.perception.inference_engine import InferenceEngine
from visionradar.perception.postprocess import postprocess_yolox_output
from visionradar.perception.tracker import PerceptionTracker
from visionradar.perception.lifecycle import ObjectLifecycleManager
from visionradar.perception.metrics import PerceptionMetricsCollector

logger = logging.getLogger(__name__)


def _safe_put_broadcast(queue_obj: asyncio.Queue, result: FrameResult, metrics_collector: PerceptionMetricsCollector):
    """Safely pushes FrameResult into asyncio broadcast queue, dropping oldest if full."""
    if queue_obj.full():
        try:
            queue_obj.get_nowait()
            metrics_collector.record_drop("queue_full_broadcast")
        except Exception:
            pass
    try:
        queue_obj.put_nowait(result)
    except Exception:
        metrics_collector.record_drop("queue_full_broadcast")


class PerceptionPipeline:
    """
    Single dedicated perception pipeline thread orchestrating steps 1 through 8.
    """
    def __init__(
        self,
        video_path: str,
        model_path: str = "data/models/yolox_nano.onnx",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        intra_op_threads: int = 8
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.capture_source = VideoSource(video_path, max_queue_size=2)
        self.inference_engine = InferenceEngine(
            model_path=model_path,
            input_size=(640, 640),
            intra_op_threads=intra_op_threads
        )
        self.tracker = PerceptionTracker(max_age=30, iou_threshold=0.3)
        self.lifecycle_manager = ObjectLifecycleManager(confirmation_frames=2, grace_window=15)
        self.metrics_collector = PerceptionMetricsCollector(max_history=1000)

        self.broadcast_queue: Optional[asyncio.Queue] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

        self.running = False
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        """Returns True if the perception pipeline thread is active."""
        return self.running

    def start(self, event_loop: asyncio.AbstractEventLoop, broadcast_queue: asyncio.Queue):
        if self.running:
            return
        self.event_loop = event_loop
        self.broadcast_queue = broadcast_queue
        self.running = True

        self.capture_source.start()
        self._thread = threading.Thread(target=self._pipeline_worker_loop, daemon=True, name="perception-pipeline")
        self._thread.start()

    def stop(self):
        self.running = False
        self.capture_source.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _pipeline_worker_loop(self):
        logger.info(f"[Pipeline] Starting perception worker loop for {self.video_path}")

        while self.running:
            try:
                # 1. Dequeue frame item
                t_deq_start = time.perf_counter()
                item: FrameItem = self.capture_source.out_queue.get(timeout=0.5)
                t_deq_end = time.perf_counter()

                if item.is_sentinel:
                    logger.info(f"[Pipeline] Received sentinel item ({item.error_msg}). Terminating pipeline worker.")
                    break

                self.metrics_collector.record_capture(item.capture_ts)
                t_queue_wait_ms = (t_deq_start - item.capture_ts) * 1000.0

                frame = item.frame
                frame_id = item.frame_id
                h_src, w_src = frame.shape[:2]

                timings: Dict[str, float] = {"queue_wait": t_queue_wait_ms}

                # 2. Preprocess
                t0_prep = time.perf_counter()
                blob, scale, pad = letterbox_preprocess(frame, target_size=(640, 640))
                t1_prep = time.perf_counter()
                timings["preprocess"] = (t1_prep - t0_prep) * 1000.0

                # 3. Inference
                t0_inf = time.perf_counter()
                raw_outputs, inf_ms = self.inference_engine.infer(blob)
                t1_inf = time.perf_counter()
                timings["inference"] = inf_ms

                # 4. Postprocess
                t0_post = time.perf_counter()
                detections = postprocess_yolox_output(
                    raw_outputs=raw_outputs,
                    frame_id=frame_id,
                    source_shape=(w_src, h_src),
                    input_size=(640, 640),
                    confidence_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_threshold,
                    scale=scale,
                    pad=pad
                )
                t1_post = time.perf_counter()
                timings["postprocess"] = (t1_post - t0_post) * 1000.0

                # 5. Tracker Update
                t0_trk = time.perf_counter()
                raw_tracks = self.tracker.update(detections, frame_id, item.capture_ts)
                t1_trk = time.perf_counter()
                timings["tracking"] = (t1_trk - t0_trk) * 1000.0

                # 6. Object Lifecycle Manager
                t0_life = time.perf_counter()
                confirmed_tracks, lifecycle_events = self.lifecycle_manager.update(raw_tracks, frame_id)
                t1_life = time.perf_counter()
                timings["lifecycle"] = (t1_life - t0_life) * 1000.0

                # 7. Assemble FrameResult
                result_ts = time.perf_counter()
                result = FrameResult(
                    frame_id=frame_id,
                    capture_ts=item.capture_ts,
                    result_ts=result_ts,
                    detections=detections,
                    tracks=confirmed_tracks,
                    degraded=False,
                    stage_timings_ms=timings
                )

                # Record metrics
                self.metrics_collector.record_frame_result(
                    capture_ts=item.capture_ts,
                    result_ts=result_ts,
                    stage_timings_ms=timings,
                    has_detections=len(detections) > 0
                )

                # 8. Push to broadcast queue via asyncio thread-safe call
                if self.event_loop and self.broadcast_queue and self.event_loop.is_running():
                    self.event_loop.call_soon_threadsafe(
                        _safe_put_broadcast,
                        self.broadcast_queue,
                        result,
                        self.metrics_collector
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Pipeline] Error during frame processing: {e}", exc_info=True)
                self.metrics_collector.record_drop("decode_error")

        self.running = False
        logger.info(f"[Pipeline] Perception worker loop finished.")

    def get_perception_metrics(self, window_size: int = 100) -> PerceptionMetrics:
        active_cnt = len(self.lifecycle_manager.track_states)
        return self.metrics_collector.get_metrics(window_size=window_size, active_track_count=active_cnt)
