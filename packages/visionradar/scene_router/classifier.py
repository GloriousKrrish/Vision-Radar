import os
import cv2
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from visionradar.scene_router.schemas import (
    SceneType,
    AnalysisModule,
    SceneClassificationResult
)

class SceneClassifier:
    """
    Lightweight, locally-executable automatic video scene classifier (Layer B).
    Analyzes temporal frame samples using multi-cue visual geometry, illumination,
    and semantic structure to classify video into TRAFFIC, CLASSROOM_EXAMINATION,
    OTHER_UNKNOWN, or AMBIGUOUS.
    """
    VERSION = "scene-router-v1.0"

    def __init__(self, n_sample_frames: int = 6, confidence_threshold: float = 0.65):
        self.n_sample_frames = max(3, min(12, n_sample_frames))
        self.confidence_threshold = confidence_threshold
        self._cache: Dict[str, SceneClassificationResult] = {}

    def classify_video(self, video_path: str, use_cache: bool = True) -> SceneClassificationResult:
        """
        Classifies a video file by sampling representative frames across its duration.
        """
        if not os.path.exists(video_path):
            return SceneClassificationResult(
                scene_type=SceneType.OTHER_UNKNOWN,
                confidence=0.0,
                recommended_module=AnalysisModule.NONE,
                requires_user_confirmation=True,
                reason=f"Video file not found at path: {video_path}",
                model_version=self.VERSION,
                sampled_frames_count=0,
                video_path=video_path
            )

        # Compute SHA256 cache key from first 1MB of video file
        file_hash = self._compute_quick_hash(video_path)
        cache_key = f"{file_hash}:{self.VERSION}:{self.n_sample_frames}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Return fresh copy with current video_path
            return SceneClassificationResult(**{**cached.model_dump(), "video_path": video_path})

        # Sample frames
        frames = self._sample_video_frames(video_path, self.n_sample_frames)
        if not frames:
            result = SceneClassificationResult(
                scene_type=SceneType.OTHER_UNKNOWN,
                confidence=0.0,
                recommended_module=AnalysisModule.NONE,
                requires_user_confirmation=True,
                reason="Unable to decode representative frames from video (file may be empty or corrupted)",
                model_version=self.VERSION,
                sampled_frames_count=0,
                video_path=video_path,
                sha256_hash=file_hash
            )
            return result

        result = self.classify_frames(frames, video_path=video_path, sha256_hash=file_hash)
        if use_cache:
            self._cache[cache_key] = result
        return result

    def classify_frames(
        self,
        frames: List[np.ndarray],
        video_path: Optional[str] = None,
        sha256_hash: Optional[str] = None
    ) -> SceneClassificationResult:
        """
        Analyzes a sequence of BGR image frames and returns a calibrated scene classification result.
        """
        if not frames:
            return SceneClassificationResult(
                scene_type=SceneType.OTHER_UNKNOWN,
                confidence=0.0,
                recommended_module=AnalysisModule.NONE,
                requires_user_confirmation=True,
                reason="No frames provided for visual classification",
                model_version=self.VERSION,
                sampled_frames_count=0,
                video_path=video_path,
                sha256_hash=sha256_hash
            )

        roadway_scores = []
        classroom_scores = []
        cue_breakdowns = []

        for frame in frames:
            r_score, c_score, breakdown = self._analyze_single_frame(frame)
            roadway_scores.append(r_score)
            classroom_scores.append(c_score)
            cue_breakdowns.append(breakdown)

        avg_roadway = float(np.mean(roadway_scores))
        avg_classroom = float(np.mean(classroom_scores))

        # Temporal voting consistency
        roadway_votes = sum(1 for r, c in zip(roadway_scores, classroom_scores) if r > c and r >= 0.50)
        classroom_votes = sum(1 for r, c in zip(roadway_scores, classroom_scores) if c > r and c >= 0.50)
        total_frames = len(frames)

        roadway_consistency = roadway_votes / total_frames
        classroom_consistency = classroom_votes / total_frames

        # Aggregate cue metrics
        agg_cues = {
            "roadway_composite": round(avg_roadway, 4),
            "classroom_composite": round(avg_classroom, 4),
            "roadway_consistency": round(roadway_consistency, 4),
            "classroom_consistency": round(classroom_consistency, 4),
            "asphalt_dominance": round(float(np.mean([b["asphalt_ratio"] for b in cue_breakdowns])), 4),
            "linear_perspective_lines": round(float(np.mean([b["road_lines_score"] for b in cue_breakdowns])), 4),
            "indoor_illumination_uniformity": round(float(np.mean([b["indoor_uniformity"] for b in cue_breakdowns])), 4),
            "desk_paper_rectangles": round(float(np.mean([b["desk_paper_score"] for b in cue_breakdowns])), 4)
        }

        # Calibration & Decision Logic
        score_diff = abs(avg_roadway - avg_classroom)

        if avg_roadway < 0.38 and avg_classroom < 0.38:
            # Neither domain exhibits recognizable structure
            scene_type = SceneType.OTHER_UNKNOWN
            confidence = round(1.0 - max(avg_roadway, avg_classroom), 2)
            recommended = AnalysisModule.NONE
            confirmation = True
            reason = "Scene does not display distinctive roadway traffic features or indoor classroom seating patterns"

        elif score_diff < 0.16 or (roadway_consistency < 0.60 and classroom_consistency < 0.60):
            # Cues are close or inconsistent across sampled timestamps
            scene_type = SceneType.AMBIGUOUS
            confidence = round(max(avg_roadway, avg_classroom) * 0.75, 2)
            recommended = AnalysisModule.TRAFFIC if avg_roadway >= avg_classroom else AnalysisModule.EXAMINATION
            confirmation = True
            reason = (
                f"Visual indicators are ambiguous across sampled frames "
                f"(roadway: {avg_roadway:.2f}, classroom: {avg_classroom:.2f}); manual confirmation required"
            )

        elif avg_roadway > avg_classroom and roadway_consistency >= 0.60:
            scene_type = SceneType.TRAFFIC
            confidence = round(min(0.98, max(0.68, avg_roadway * (0.85 + 0.15 * roadway_consistency))), 2)
            recommended = AnalysisModule.TRAFFIC
            confirmation = confidence < self.confidence_threshold
            reason = f"Roadway environment detected with linear perspective lanes and asphalt surface characteristics (confidence: {confidence})"

        elif avg_classroom > avg_roadway and classroom_consistency >= 0.60:
            scene_type = SceneType.CLASSROOM_EXAMINATION
            confidence = round(min(0.98, max(0.68, avg_classroom * (0.85 + 0.15 * classroom_consistency))), 2)
            recommended = AnalysisModule.EXAMINATION
            confirmation = confidence < self.confidence_threshold
            reason = f"Indoor classroom / examination setting detected with structured desk surfaces and uniform interior lighting (confidence: {confidence})"

        else:
            scene_type = SceneType.AMBIGUOUS
            confidence = 0.55
            recommended = AnalysisModule.TRAFFIC if avg_roadway >= avg_classroom else AnalysisModule.EXAMINATION
            confirmation = True
            reason = "Mixed or borderline scene cues detected; user confirmation recommended"

        return SceneClassificationResult(
            scene_type=scene_type,
            confidence=confidence,
            recommended_module=recommended,
            requires_user_confirmation=confirmation,
            reason=reason,
            model_version=self.VERSION,
            sampled_frames_count=total_frames,
            cue_scores=agg_cues,
            video_path=video_path,
            sha256_hash=sha256_hash
        )

    def _analyze_single_frame(self, frame: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """
        Extracts multi-cue normalized scores for a single BGR image.
        Returns: (roadway_score, classroom_score, details_dict)
        """
        h, w = frame.shape[:2]
        # Standardize working resolution for deterministic benchmarking
        target_w, target_h = 480, 270
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # --------------------------------------------------------------------
        # Cue 1: Asphalt / Road Surface Detection (Lower 65% of frame)
        # Asphalt is characterized by low saturation (< 65) and mid-range value (40 - 180)
        # --------------------------------------------------------------------
        lower_region = hsv[int(target_h * 0.35):, :]
        sat = lower_region[:, :, 1]
        val = lower_region[:, :, 2]
        asphalt_mask = (sat < 65) & (val > 35) & (val < 195)
        asphalt_ratio = float(np.mean(asphalt_mask))

        # --------------------------------------------------------------------
        # Cue 2: Perspective Roadway Lines (Hough / Sobel oriented edges)
        # Roadways feature diagonal lines converging towards a vanishing point
        # --------------------------------------------------------------------
        edges = cv2.Canny(gray[int(target_h * 0.30):, :], 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)
        road_lines_score = 0.0
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                coords = line.flatten()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    # Oblique perspective lines typical of roadway lanes (20 to 75 degrees)
                    if 20 <= angle <= 75:
                        angles.append(angle)
            if len(angles) >= 3:
                road_lines_score = min(1.0, len(angles) / 18.0)

        # --------------------------------------------------------------------
        # Cue 3: Outdoor Sky / Horizon Gradient (Upper 35% vs Lower 65%)
        # Road scenes typically feature bright sky or higher luminance in upper third
        # --------------------------------------------------------------------
        top_luminance = float(np.mean(gray[:int(target_h * 0.35), :]))
        bottom_luminance = float(np.mean(gray[int(target_h * 0.35):, :]))
        sky_gradient = max(0.0, min(1.0, (top_luminance - bottom_luminance + 30.0) / 120.0))

        # --------------------------------------------------------------------
        # Cue 4: Indoor Illumination Uniformity (Classroom Indicator)
        # Classrooms typically have uniform artificial overhead lighting with low variance
        # --------------------------------------------------------------------
        grid_luminances = []
        gh, gw = target_h // 3, target_w // 3
        for r in range(3):
            for c in range(3):
                cell = gray[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
                grid_luminances.append(float(np.mean(cell)))
        illumination_std = float(np.std(grid_luminances))
        # Lower std across grid indicates uniform indoor lighting
        indoor_uniformity = max(0.0, min(1.0, 1.0 - (illumination_std / 45.0)))

        # --------------------------------------------------------------------
        # Cue 5: Desk / Table / Paper Geometry (Classroom Indicator)
        # Classrooms exhibit repetitive horizontal rectangular contours (desks, papers)
        # --------------------------------------------------------------------
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        desk_paper_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < (target_w * target_h * 0.15):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    aspect = float(cw) / max(1, ch)
                    # Rectangular sheets of paper or desks (aspect 1.1 to 3.5)
                    if 1.1 <= aspect <= 3.8:
                        desk_paper_count += 1
        desk_paper_score = min(1.0, desk_paper_count / 14.0)
        edge_density = float(np.count_nonzero(edges)) / float(target_w * target_h)

        # --------------------------------------------------------------------
        # Composite Scoring
        # --------------------------------------------------------------------
        roadway_score = float(
            0.45 * asphalt_ratio +
            0.35 * road_lines_score +
            0.20 * sky_gradient
        )

        # Classroom requires physical visual structures (desks, papers, furniture)
        if desk_paper_score < 0.10:
            classroom_score = float(0.40 * desk_paper_score)
        else:
            classroom_score = float(
                0.50 * desk_paper_score +
                0.35 * indoor_uniformity +
                0.15 * max(0.0, 1.0 - asphalt_ratio)
            )

        # Flat / textureless / blank frame guard: penalize if edges are virtually absent
        if edge_density < 0.005:
            roadway_score = min(roadway_score, 0.15)
            classroom_score = min(classroom_score, 0.15)

        breakdown = {
            "asphalt_ratio": asphalt_ratio,
            "road_lines_score": road_lines_score,
            "sky_gradient": sky_gradient,
            "indoor_uniformity": indoor_uniformity,
            "desk_paper_score": desk_paper_score
        }

        return roadway_score, classroom_score, breakdown

    def _sample_video_frames(self, video_path: str, n_samples: int) -> List[np.ndarray]:
        """Fast temporal sampling of video without full sequential decode."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback: sequential read of first few frames
            frames = []
            for _ in range(n_samples):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
            cap.release()
            return frames

        # Select evenly spaced timestamps (from 10% to 90%)
        ratios = np.linspace(0.10, 0.90, n_samples)
        sample_indices = [int(r * total_frames) for r in ratios]

        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        return frames

    def _compute_quick_hash(self, file_path: str) -> str:
        """Fast SHA256 on the first 2MB + file size to uniquely identify video content."""
        hasher = hashlib.sha256()
        try:
            size = os.path.getsize(file_path)
            hasher.update(str(size).encode("utf-8"))
            with open(file_path, "rb") as f:
                chunk = f.read(2 * 1024 * 1024)
                hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return "unknown-hash"
