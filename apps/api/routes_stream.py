"""
VisionRadar Real-Time Perception Streaming API Routes.

Provides WebSocket streaming endpoints (/ws/stream/{session_id}) and perception health/metrics HTTP endpoints.
Section C & E & J Compliant.
"""

import asyncio
import logging
import os
from typing import Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import JSONResponse

from visionradar.perception.pipeline import PerceptionPipeline
from visionradar.perception.metrics import global_metrics_collector
from visionradar.streaming.ws_manager import ws_manager
from visionradar.streaming.ws_protocol import (
    create_stream_status_message,
    create_frame_result_message,
    create_perception_metrics_message,
    create_error_message
)

logger = logging.getLogger("visionradar.api.stream")

router = APIRouter(tags=["Real-Time Perception Stream"])

# Global session registry for active PerceptionPipelines
_active_pipelines: Dict[str, PerceptionPipeline] = {}
_pipeline_tasks: Dict[str, asyncio.Task] = {}


def get_or_create_pipeline(
    session_id: str,
    video_path: str,
    model_path: Optional[str] = None
) -> PerceptionPipeline:
    """Returns an existing pipeline for session_id or initializes a new one."""
    if session_id in _active_pipelines:
        return _active_pipelines[session_id]

    if not model_path:
        model_path = os.environ.get("YOLOX_NANO_ONNX_PATH", "packages/visionradar/perception/models/yolox_nano.onnx")

    pipeline = PerceptionPipeline(
        source_path=video_path,
        model_path=model_path,
        confidence_thresh=0.25,
        nms_thresh=0.45,
        fps_cadence=30.0,
        warmup_iters=5,
        metrics_collector=global_metrics_collector
    )
    _active_pipelines[session_id] = pipeline
    return pipeline


async def _pipeline_broadcast_loop(session_id: str, pipeline: PerceptionPipeline):
    """
    Consumes outputs from pipeline's broadcast_queue and broadcasts to WebSocket clients.
    Emits frame_result per frame, and perception_metrics every 30 frames (Section E & J requirement).
    """
    frame_counter = 0
    try:
        pipeline.start()
        # Broadcast stream status: running
        await ws_manager.broadcast(
            session_id,
            create_stream_status_message(session_id, "running", {"fps_cadence": 30.0})
        )

        while pipeline.is_running() or not pipeline.broadcast_queue.empty():
            try:
                # Wait for next frame result with non-blocking timeout
                frame_result = await asyncio.wait_for(pipeline.broadcast_queue.get(), timeout=1.0)
                frame_counter += 1

                # 1. Broadcast frame_result
                msg = create_frame_result_message(frame_result.to_dict(), seq=frame_counter)
                await ws_manager.broadcast(session_id, msg)

                # 2. Fixed cadence: emit perception_metrics every 30 frames (Section J instruction)
                if frame_counter % 30 == 0:
                    metrics_snapshot = global_metrics_collector.get_snapshot(window="last_100").to_dict()
                    metrics_msg = create_perception_metrics_message(metrics_snapshot, seq=frame_counter)
                    await ws_manager.broadcast(session_id, metrics_msg)

                pipeline.broadcast_queue.task_done()

            except asyncio.TimeoutError:
                if not pipeline.is_running():
                    break
            except Exception as e:
                logger.error(f"Error in broadcast loop for session {session_id}: {e}")

        # Stream completed cleanly
        await ws_manager.broadcast(
            session_id,
            create_stream_status_message(session_id, "completed", {"total_frames": frame_counter})
        )

    except Exception as e:
        logger.error(f"Pipeline failure for session {session_id}: {e}")
        await ws_manager.broadcast(
            session_id,
            create_error_message("PIPELINE_ERROR", str(e))
        )
    finally:
        pipeline.stop()
        _active_pipelines.pop(session_id, None)
        _pipeline_tasks.pop(session_id, None)


@router.websocket("/ws/stream/{session_id}")
@router.websocket("/api/v1/stream/{session_id}")
async def websocket_perception_stream(websocket: WebSocket, session_id: str, video_path: Optional[str] = None):
    """
    WebSocket streaming endpoint for real-time Perception Engine (Section E & J).
    Emits frame_result, perception_metrics (every 30 frames), stream_status, error.
    """
    await ws_manager.connect(websocket, session_id)

    # If video_path parameter is supplied and pipeline is not yet running, launch it
    if video_path and session_id not in _active_pipelines:
        if os.path.exists(video_path):
            pipeline = get_or_create_pipeline(session_id, video_path)
            loop = asyncio.get_event_loop()
            task = loop.create_task(_pipeline_broadcast_loop(session_id, pipeline))
            _pipeline_tasks[session_id] = task

    try:
        while True:
            # Keep connection alive until client disconnects (server -> client stream)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.warning(f"WebSocket session {session_id} exception: {e}")
        await ws_manager.disconnect(websocket, session_id)


@router.post("/api/v1/perception/stream/start/{session_id}")
async def start_perception_stream(session_id: str, video_path: str):
    """HTTP endpoint to start pipeline processing for a video source."""
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found at path: {video_path}")

    if session_id in _active_pipelines and _active_pipelines[session_id].is_running():
        return {"session_id": session_id, "status": "already_running"}

    pipeline = get_or_create_pipeline(session_id, video_path)
    loop = asyncio.get_event_loop()
    task = loop.create_task(_pipeline_broadcast_loop(session_id, pipeline))
    _pipeline_tasks[session_id] = task

    return {
        "session_id": session_id,
        "status": "started",
        "video_path": video_path
    }


@router.get("/api/v1/perception/metrics")
def get_perception_metrics(window: str = "last_100"):
    """
    HTTP endpoint returning authoritative rolling PerceptionMetrics (Section F).
    """
    snapshot = global_metrics_collector.get_snapshot(window=window)
    return snapshot.to_dict()


@router.get("/api/v1/perception/health")
def get_perception_health():
    """Returns perception engine operational status."""
    return {
        "status": "online",
        "active_sessions": list(_active_pipelines.keys()),
        "active_tasks": len(_pipeline_tasks),
        "onnx_provider": "CPUExecutionProvider",
        "warmup": "enabled"
    }
