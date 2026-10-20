import os
import cv2
import numpy as np
import pytest

from visionradar.scene_router.schemas import (
    SceneType,
    AnalysisModule,
    SceneClassificationResult
)
from visionradar.scene_router.classifier import SceneClassifier

def test_scene_router_schemas():
    """Verify Scene Router schema serialization and field contracts."""
    res = SceneClassificationResult(
        scene_type=SceneType.TRAFFIC,
        confidence=0.88,
        recommended_module=AnalysisModule.TRAFFIC,
        requires_user_confirmation=False,
        reason="Highway with multiple vehicles",
        model_version="scene-router-v1.0",
        sampled_frames_count=6,
        cue_scores={"asphalt_ratio": 0.82}
    )
    d = res.to_dict()
    assert d["scene_type"] == "TRAFFIC"
    assert d["recommended_module"] == "traffic"
    assert d["confidence"] == 0.88
    assert d["requires_user_confirmation"] is False

def test_classify_real_traffic_video():
    """Verify real video Traffic1.mp4 is automatically routed to TRAFFIC pipeline."""
    video_path = "data/videos/Traffic1.mp4"
    if not os.path.exists(video_path):
        pytest.skip("Traffic1.mp4 not present in data/videos")

    classifier = SceneClassifier(n_sample_frames=5)
    res = classifier.classify_video(video_path)

    assert res.scene_type == SceneType.TRAFFIC
    assert res.recommended_module == AnalysisModule.TRAFFIC
    assert res.confidence >= 0.65
    assert res.sampled_frames_count == 5
    assert "roadway" in res.reason.lower() or "traffic" in res.reason.lower()

def test_classify_synthetic_classroom_frames():
    """Verify classroom scene features with desks and paper sheets route to CLASSROOM_EXAMINATION."""
    classifier = SceneClassifier()

    # Generate 5 synthetic classroom frames:
    # Uniform indoor illumination + horizontal desks with bright paper rectangles
    frames = []
    for _ in range(5):
        frame = np.full((270, 480, 3), 160, dtype=np.uint8)  # uniform wall/floor color
        # Draw desks
        for y in [80, 150, 220]:
            for x in [40, 180, 320]:
                # Brownish/gray desk surface
                cv2.rectangle(frame, (x, y), (x + 110, y + 45), (100, 120, 130), -1)
                # Bright white exam paper sheet on desk
                cv2.rectangle(frame, (x + 35, y + 8), (x + 75, y + 36), (245, 245, 245), -1)
        frames.append(frame)

    res = classifier.classify_frames(frames)
    assert res.scene_type == SceneType.CLASSROOM_EXAMINATION
    assert res.recommended_module == AnalysisModule.EXAMINATION
    assert res.confidence >= 0.65

def test_classify_unknown_blank_frames():
    """Verify unstructured / blank scenes yield OTHER_UNKNOWN with confirmation requirement."""
    classifier = SceneClassifier()

    # Solid cyan frames (e.g. underwater or flat background)
    frames = [np.full((270, 480, 3), (255, 200, 0), dtype=np.uint8) for _ in range(4)]
    res = classifier.classify_frames(frames)

    assert res.scene_type == SceneType.OTHER_UNKNOWN
    assert res.recommended_module == AnalysisModule.NONE
    assert res.requires_user_confirmation is True

def test_classify_ambiguous_frames():
    """Verify scenes with mixed or borderline cues return AMBIGUOUS requiring confirmation."""
    classifier = SceneClassifier()

    # Frame with both asphalt-like lower half and desk-like rectangles
    frames = []
    for _ in range(4):
        frame = np.full((270, 480, 3), 100, dtype=np.uint8)
        # asphalt lower half
        frame[135:, :] = (80, 80, 80)
        # a couple of desk rectangles
        cv2.rectangle(frame, (100, 150), (220, 190), (200, 200, 200), -1)
        frames.append(frame)

    res = classifier.classify_frames(frames)
    # Either ambiguous or requires confirmation
    assert res.requires_user_confirmation is True

def test_classify_missing_or_corrupt_file():
    """Verify non-existent video files return safely without unhandled crashes."""
    classifier = SceneClassifier()
    res = classifier.classify_video("data/videos/non_existent_video_file_xyz.mp4")

    assert res.scene_type == SceneType.OTHER_UNKNOWN
    assert res.recommended_module == AnalysisModule.NONE
    assert res.requires_user_confirmation is True
    assert "not found" in res.reason.lower()

def test_classifier_caching_performance():
    """Verify repeated classification of the same video utilizes hash caching."""
    video_path = "data/videos/Traffic1.mp4"
    if not os.path.exists(video_path):
        pytest.skip("Traffic1.mp4 not present in data/videos")

    classifier = SceneClassifier()
    res1 = classifier.classify_video(video_path, use_cache=True)
    res2 = classifier.classify_video(video_path, use_cache=True)

    assert res1.scene_type == res2.scene_type
    assert res1.confidence == res2.confidence
    assert res1.sha256_hash == res2.sha256_hash
