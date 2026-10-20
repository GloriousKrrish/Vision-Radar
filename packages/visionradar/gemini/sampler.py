"""
VisionRadar Gemini Multimodal Intelligence Service: Frame Sampler.
Extracts, optimizes, and prepares bounded keyframes for multimodal analysis.
Ensures zero unnecessary upload of continuous video.
"""

import os
import hashlib
from typing import List, Tuple, Optional
import cv2
import numpy as np
from visionradar.gemini.config import gemini_config

class SampledFrame:
    """Encapsulates a selected keyframe ready for Gemini multimodal inspection."""
    def __init__(self, frame_index: int, timestamp_s: float, jpeg_bytes: bytes, width: int, height: int):
        self.frame_index = frame_index
        self.timestamp_s = timestamp_s
        self.jpeg_bytes = jpeg_bytes
        self.width = width
        self.height = height

class GeminiFrameSampler:
    """Selects and optimizes bounded keyframes from physical video files."""

    def __init__(self, max_frames: Optional[int] = None):
        self.max_frames = max_frames or gemini_config.max_selected_frames

    def sample_keyframes(
        self,
        video_path: str,
        target_frame_indices: Optional[List[int]] = None,
        max_dimension: int = 1024
    ) -> List[SampledFrame]:
        """
        Extracts bounded keyframes from the specified video path.
        If target_frame_indices is provided, selects up to max_frames from the target list.
        If target_frame_indices is empty or None, samples evenly across the video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

        # Determine frame indices to extract
        if target_frame_indices and len(target_frame_indices) > 0:
            # Deduplicate and sort
            unique_indices = sorted(list(dict.fromkeys(target_frame_indices)))
            if len(unique_indices) <= self.max_frames:
                selected_indices = unique_indices
            else:
                # Sub-sample evenly across the candidate frame indices
                step = len(unique_indices) / float(self.max_frames)
                selected_indices = [unique_indices[int(i * step)] for i in range(self.max_frames)]
        else:
            # Sample evenly across the video
            count = min(self.max_frames, max(1, total_frames))
            selected_indices = [int(i * (total_frames - 1) / (count - 1)) if count > 1 else 0 for i in range(count)]

        sampled: List[SampledFrame] = []
        for f_idx in selected_indices:
            if f_idx < 0 or f_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]

            # Scale down if excessive to optimize multimodal latency
            if max(h, w) > max_dimension:
                scale = max_dimension / float(max(h, w))
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w

            # JPEG encode with quality 85 for sharp detail
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                ts = round(float(f_idx) / fps, 2)
                sampled.append(SampledFrame(
                    frame_index=f_idx,
                    timestamp_s=ts,
                    jpeg_bytes=buf.tobytes(),
                    width=w,
                    height=h
                ))

        cap.release()
        return sampled

    @staticmethod
    def compute_cache_hash(
        job_id: int,
        module: str,
        model: str,
        frame_indices: List[int],
        extra_key: str = ""
    ) -> str:
        """
        Computes a deterministic SHA-256 cache hash for an analysis request.
        Prevents duplicate API calls when re-visiting or retrying jobs.
        """
        norm_frames = ",".join(str(f) for f in sorted(list(dict.fromkeys(frame_indices))))
        payload = f"{job_id}:{module}:{model}:{norm_frames}:{extra_key}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# Global default sampler
gemini_sampler = GeminiFrameSampler()
