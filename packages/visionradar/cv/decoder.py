import cv2
import os
import numpy as np
from typing import Dict, Any, Generator, Tuple, Optional

class VideoMetadata:
    def __init__(self, filename: str, width: int, height: int, fps: float, total_frames: int, duration_sec: float):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
        self.duration_sec = duration_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2)
        }

class VideoDecoder:
    """
    Handles video decoding, extracting frames with accurate presentation timestamps (PTS).
    """
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        self.file_path = file_path

    def get_metadata(self) -> VideoMetadata:
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.file_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(total_frames / fps) if fps > 0 else 0.0
        cap.release()

        return VideoMetadata(
            filename=os.path.basename(self.file_path),
            width=w,
            height=h,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration
        )

    def decode_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (frame_index, timestamp_sec, frame_bgr_image) for each decoded frame.
        """
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.file_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = (msec / 1000.0) if msec > 0 else (frame_idx / fps)

            yield frame_idx, timestamp, frame
            frame_idx += 1

        cap.release()
