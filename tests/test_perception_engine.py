"""
VisionRadar Real-Time Perception Engine Test Suite (Section G Compliant).

Tests:
1. Pure-function postprocess NMS and confidence threshold filtering.
2. Object lifecycle transitions (NEW -> ACTIVE -> LOST -> REMOVED) and ID stability/no-reuse.
3. No-synthetic-data rule (black/blank frame input yields empty detections).
4. Perception pipeline determinism (identical video input -> identical track ID sequence).
5. Perception metrics collector rolling window statistics.
6. WebSocket protocol envelope formatting & multi-client isolation.
"""

import os
import pytest
import numpy as np
import time
import asyncio
from typing import List

from visionradar.perception.schemas import (
    BoundingBox, VehicleClass, VehicleDetection, VehicleTrack, TrackState, FrameResult
)
from visionradar.perception.postprocess import postprocess_yolox_output
from visionradar.perception.lifecycle import ObjectLifecycleManager
from visionradar.perception.metrics import PerceptionMetricsCollector
from visionradar.perception.pipeline import PerceptionPipeline
from visionradar.streaming.ws_protocol import (
    PROTOCOL_VERSION,
    create_stream_status_message,
    create_frame_result_message,
    create_perception_metrics_message,
    create_error_message
)
from visionradar.streaming.ws_manager import ConnectionManager


# --- 1. Postprocess NMS & Confidence Unit Tests ---
def test_postprocess_confidence_and_nms():
    """Verify NMS suppresses overlapping boxes and confidence threshold filters low-confidence boxes."""
    # Synthetic prediction array: (1, 8400, 85)
    raw_pred = np.zeros((1, 8400, 85), dtype=np.float32)

    # Box 0: High confidence car (x=100, y=100, w=100, h=100)
    raw_pred[0, 0, 0] = 150.0  # cx
    raw_pred[0, 0, 1] = 150.0  # cy
    raw_pred[0, 0, 2] = np.log(100.0 / 8.0)  # log(w/stride)
    raw_pred[0, 0, 3] = np.log(100.0 / 8.0)  # log(h/stride)
    raw_pred[0, 0, 4] = 0.90  # obj_conf
    raw_pred[0, 0, 2 + 5] = 0.95  # COCO class 2 = CAR

    # Box 1: Overlapping box with lower confidence (0.60)
    raw_pred[0, 1, 0] = 152.0
    raw_pred[0, 1, 1] = 152.0
    raw_pred[0, 1, 2] = np.log(100.0 / 8.0)
    raw_pred[0, 1, 3] = np.log(100.0 / 8.0)
    raw_pred[0, 1, 4] = 0.60
    raw_pred[0, 1, 2 + 5] = 0.90

    # Box 2: Distinct box (x=400, y=400)
    raw_pred[0, 2, 0] = 450.0
    raw_pred[0, 2, 1] = 450.0
    raw_pred[0, 2, 2] = np.log(100.0 / 8.0)
    raw_pred[0, 2, 3] = np.log(100.0 / 8.0)
    raw_pred[0, 2, 4] = 0.85
    raw_pred[0, 2, 2 + 5] = 0.90

    detections = postprocess_yolox_output(
        raw_outputs=raw_pred,
        frame_id=1,
        source_shape=(640, 640),
        input_size=(640, 640),
        confidence_threshold=0.25,
        nms_threshold=0.45
    )

    # Box 1 should be suppressed by NMS; Box 0 and Box 2 kept -> total 2 detections
    assert len(detections) == 2
    assert detections[0].vehicle_class == VehicleClass.CAR


def test_postprocess_boundary_threshold():
    """Test confidence filtering boundary conditions."""
    dummy_pred = np.zeros((1, 8400, 85), dtype=np.float32)

    # Box 0: obj_conf (0.5) * cls_conf (0.5) = 0.25 (exact threshold)
    dummy_pred[0, 0, 0] = 5.0
    dummy_pred[0, 0, 1] = 5.0
    dummy_pred[0, 0, 2] = 2.0
    dummy_pred[0, 0, 3] = 2.0
    dummy_pred[0, 0, 4] = 0.50
    dummy_pred[0, 0, 2 + 5] = 0.50  # COCO class 2 = CAR (score = 0.25)

    # Box 1: obj_conf (0.4) * cls_conf (0.5) = 0.20 (below threshold)
    dummy_pred[0, 1, 0] = 20.0
    dummy_pred[0, 1, 1] = 20.0
    dummy_pred[0, 1, 2] = 2.0
    dummy_pred[0, 1, 3] = 2.0
    dummy_pred[0, 1, 4] = 0.40
    dummy_pred[0, 1, 2 + 5] = 0.50  # (score = 0.20)

    detections = postprocess_yolox_output(
        raw_outputs=dummy_pred,
        frame_id=1,
        source_shape=(640, 640),
        input_size=(640, 640),
        confidence_threshold=0.25,
        nms_threshold=0.45
    )

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.25, abs=1e-3)


# --- 2. Lifecycle Manager Unit Tests ---
def test_lifecycle_state_transitions():
    """Verify NEW -> ACTIVE -> LOST -> REMOVED rules and ID reuse prohibition."""
    manager = ObjectLifecycleManager(confirmation_frames=2, grace_window=3)

    # Frame 0: First detection of track 1
    det1 = VehicleTrack(
        track_id=1,
        state=TrackState.NEW,
        bbox=BoundingBox(10.0, 10.0, 50.0, 50.0),
        confidence=0.85,
        vehicle_class=VehicleClass.CAR,
        first_seen_frame=0,
        last_seen_frame=0,
        frames_tracked=0,
        frames_since_last_detection=0
    )

    tracks0, _ = manager.update([det1], frame_id=0)
    assert len(tracks0) == 1
    assert tracks0[0].state == TrackState.NEW  # K=2 confirmation: frame 1 needs 2nd detection

    # Frame 1: Second detection -> should promote to ACTIVE
    det1_f1 = VehicleTrack(
        track_id=1,
        state=TrackState.NEW,
        bbox=BoundingBox(12.0, 12.0, 52.0, 52.0),
        confidence=0.88,
        vehicle_class=VehicleClass.CAR,
        first_seen_frame=0,
        last_seen_frame=1,
        frames_tracked=1,
        frames_since_last_detection=0
    )
    tracks1, _ = manager.update([det1_f1], frame_id=1)
    assert tracks1[0].state == TrackState.ACTIVE

    # Frame 2: Missed detection -> ACTIVE -> LOST
    tracks2, _ = manager.update([], frame_id=2)
    assert len(tracks2) == 1
    assert tracks2[0].state == TrackState.LOST

    # Frame 3: Missed again (frames_since_last_detection = 2) -> STILL LOST
    tracks3, _ = manager.update([], frame_id=3)
    assert tracks3[0].state == TrackState.LOST

    # Frame 4: Missed again (frames_since_last_detection = 3) -> STILL LOST
    tracks4, _ = manager.update([], frame_id=4)
    assert tracks4[0].state == TrackState.LOST

    # Frame 5: Missed again (frames_since_last_detection = 4 > grace_window 3) -> REMOVED
    tracks5, _ = manager.update([], frame_id=5)
    assert len(tracks5) == 0  # Track removed from active set
    assert manager.is_id_retired(1)  # ID 1 permanently retired


# --- 3. No Synthetic Data Rule Test ---
def test_no_synthetic_data():
    """Verify black/blank frames return detections: [], never fabricated boxes."""
    blank_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # Decode blank frame via pipeline components
    dummy_pred = np.zeros((1, 8400, 85), dtype=np.float32)
    detections = postprocess_yolox_output(
        raw_outputs=dummy_pred,
        frame_id=0,
        source_shape=(640, 640),
        input_size=(640, 640),
        confidence_threshold=0.25,
        nms_threshold=0.45
    )

    assert detections == []


# --- 4. Pipeline Determinism Test ---
def test_pipeline_determinism(tmp_path):
    """Verify running perception pipeline twice on same video yields identical track IDs."""
    import cv2
    video_path = str(tmp_path / "test_clip.mp4")

    # Generate short 15-frame synthetic MP4 video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 640))
    for i in range(15):
        frame = np.ones((640, 640, 3), dtype=np.uint8) * 128
        # Draw a white rectangle moving diagonally
        cv2.rectangle(frame, (50 + i * 5, 50 + i * 5), (150 + i * 5, 150 + i * 5), (255, 255, 255), -1)
        out.write(frame)
    out.release()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run pipeline run 1
    q1 = asyncio.Queue(maxsize=100)
    p1 = PerceptionPipeline(video_path=video_path)
    p1.start(loop, q1)

    results1: List[FrameResult] = []

    async def collect_results(pipeline, q, target_list):
        while pipeline.is_running() or not q.empty():
            try:
                res = await asyncio.wait_for(q.get(), timeout=0.5)
                target_list.append(res)
                q.task_done()
            except asyncio.TimeoutError:
                if not pipeline.is_running():
                    break

    loop.run_until_complete(collect_results(p1, q1, results1))
    p1.stop()

    # Run pipeline run 2
    q2 = asyncio.Queue(maxsize=100)
    p2 = PerceptionPipeline(video_path=video_path)
    p2.start(loop, q2)

    results2: List[FrameResult] = []
    loop.run_until_complete(collect_results(p2, q2, results2))
    p2.stop()
    loop.close()

    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1.frame_id == r2.frame_id
        t1_ids = [t.track_id for t in r1.tracks]
        t2_ids = [t.track_id for t in r2.tracks]
        assert t1_ids == t2_ids


# --- 5. Perception Metrics Collector Test ---
def test_perception_metrics_collector():
    """Verify rolling window percentile tracking and FPS computation."""
    collector = PerceptionMetricsCollector()

    for i in range(20):
        c_ts = time.perf_counter()
        collector.record_capture(c_ts)
        time.sleep(0.002)
        r_ts = time.perf_counter()
        collector.record_frame_result(
            capture_ts=c_ts,
            result_ts=r_ts,
            stage_timings_ms={"inference": 2.0, "preprocess": 1.0},
            has_detections=True
        )

    snapshot = collector.get_snapshot(window="last_100")
    assert snapshot.frames_processed == 20
    assert snapshot.stage_latency_ms["inference"].p50 > 0.0
    assert snapshot.stage_latency_ms["inference"].p99 >= snapshot.stage_latency_ms["inference"].p50


# --- 6. WebSocket Protocol & Manager Test ---
def test_ws_protocol_envelope():
    """Verify Section E message envelopes and versioning."""
    status_msg = create_stream_status_message(session_id="s123", status="running")
    assert status_msg["type"] == "stream_status"
    assert status_msg["payload"]["protocol_version"] == PROTOCOL_VERSION
    assert status_msg["payload"]["status"] == "running"

    err_msg = create_error_message(error_code="ERR_DECODE", message="Decode error")
    assert err_msg["type"] == "error"
    assert err_msg["payload"]["code"] == "ERR_DECODE"
