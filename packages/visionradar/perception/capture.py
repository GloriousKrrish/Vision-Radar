"""
VisionRadar Perception Engine — Video Capture Module

Provides VideoSource to read frames from video files or camera streams,
assigning strictly monotonic frame_id and time.perf_counter() capture_ts.
Pushes frames into a bounded queue with drop-oldest backpressure semantics.
"""

import time
import queue
import threading
import logging
import cv2
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FrameItem:
    """Wrapper container for decoded raw BGR frame and metadata."""
    __slots__ = ('frame_id', 'capture_ts', 'frame', 'is_sentinel', 'error_msg')

    def __init__(
        self,
        frame_id: int,
        capture_ts: float,
        frame: Optional[np.ndarray],
        is_sentinel: bool = False,
        error_msg: Optional[str] = None
    ):
        self.frame_id = frame_id
        self.capture_ts = capture_ts
        self.frame = frame
        self.is_sentinel = is_sentinel
        self.error_msg = error_msg


class VideoSource:
    """
    Video reader thread that decodes frames, assigns monotonic frame_id and capture_ts,
    and enqueues items into a bounded queue with drop-oldest semantics.
    """
    def __init__(self, video_path: str, max_queue_size: int = 2):
        self.video_path = video_path
        self.max_queue_size = max_queue_size
        self.out_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.next_frame_id = 0
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.frames_captured = 0
        self.frames_dropped_capture = 0
        self.fps_source = 30.0

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="capture-worker")
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _push_drop_oldest(self, item: FrameItem):
        """Pushes item to out_queue. If full, drops the oldest item first."""
        if self.out_queue.full():
            try:
                _ = self.out_queue.get_nowait()
                self.frames_dropped_capture += 1
            except queue.Empty:
                pass
        try:
            self.out_queue.put_nowait(item)
        except queue.Full:
            self.frames_dropped_capture += 1

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"[Capture] Failed to open video source: {self.video_path}")
            sentinel = FrameItem(-1, time.perf_counter(), None, is_sentinel=True, error_msg="Failed to open video source")
            self._push_drop_oldest(sentinel)
            self.running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.fps_source = fps

        while self.running:
            t_capture = time.perf_counter()
            ret, frame = cap.read()

            if not ret or frame is None:
                # End of stream or decode error
                logger.info(f"[Capture] Video stream ended or decode failed at frame_id={self.next_frame_id}")
                sentinel = FrameItem(self.next_frame_id, t_capture, None, is_sentinel=True, error_msg="EOS or decode failure")
                self._push_drop_oldest(sentinel)
                break

            item = FrameItem(
                frame_id=self.next_frame_id,
                capture_ts=t_capture,
                frame=frame
            )
            self.next_frame_id += 1
            self.frames_captured += 1
            self._push_drop_oldest(item)

        cap.release()
        self.running = False
