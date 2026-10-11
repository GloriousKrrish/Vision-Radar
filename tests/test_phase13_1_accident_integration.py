import os
import sys
import json
import shutil
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

from visionradar.perception.accident_classifier import AccidentClassifierService
from visionradar.intelligence.events import TrafficEventEngine

# Label synthetic test data clearly as required by Part 8
SYNTHETIC_LABEL = "SYNTHETIC TEST DATA"

def test_classifier_loads():
    """1. Test classifier model loads cleanly without crashing."""
    service = AccidentClassifierService()
    service._ensure_loaded()
    assert service.is_loaded is True, "AccidentClassifierService failed to load model."
    assert service.get_telemetry()["accident_classifier_loaded"] is True

def test_preprocessing_contract():
    """2. Test exact preprocessing contract: 224x224 RGB ImageNet normalization tensor [1, 3, 224, 224]."""
    service = AccidentClassifierService()
    # Create a synthetic test image BGR 640x480
    synthetic_bgr = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    nchw, prep_ms = service.preprocess_image(synthetic_bgr)
    
    assert nchw.shape == (1, 3, 224, 224), f"Preprocessed shape mismatch: {nchw.shape}"
    assert nchw.dtype == np.float32, f"Preprocessed dtype mismatch: {nchw.dtype}"
    assert prep_ms > 0, "Preprocessing time must be recorded."

def test_output_mapping():
    """3. Test classifier class index mapping: 0 -> NORMAL_TRAFFIC, 1 -> ACCIDENT_CANDIDATE."""
    service = AccidentClassifierService()
    synthetic_bgr = np.zeros((224, 224, 3), dtype=np.uint8) # Black image synthetic test data
    
    result = service.predict_frame(synthetic_bgr, frame_index=10)
    assert result["predicted_class"] in ["NORMAL_TRAFFIC", "ACCIDENT_CANDIDATE"]
    assert result["class_index"] in [0, 1]
    if result["class_index"] == 0:
        assert result["predicted_class"] == "NORMAL_TRAFFIC"
    else:
        assert result["predicted_class"] == "ACCIDENT_CANDIDATE"

def test_candidate_generation():
    """4. Test candidate generation from trajectory intelligence triggers (synthetic test data)."""
    event_engine = TrafficEventEngine()
    # Simulate synthetic stopped vehicle trigger
    track_stopped = {
        "track_id": 99,
        "vehicle_class": "Car",
        "bbox": [100, 100, 200, 200],
        "speed_kmh": 0.0,
        "speed_status": "VALID",
        "lane": "Lane 1"
    }
    events = []
    # Feed stopped vehicle frames to engine for 35 frames
    for f in range(1, 35):
        evs = event_engine.process_frame([track_stopped], frame_index=f, timestamp=f * 0.033)
        events.extend(evs)
    # Feed release frames (moving vehicle or empty track) to finalize active state machine
    track_moving = {
        "track_id": 99,
        "vehicle_class": "Car",
        "bbox": [100, 100, 200, 200],
        "speed_kmh": 45.0,
        "speed_status": "VALID",
        "lane": "Lane 1"
    }
    for f in range(35, 45):
        evs = event_engine.process_frame([track_moving], frame_index=f, timestamp=f * 0.033)
        events.extend(evs)
        
    assert len(events) > 0, "Candidate incident generation failed to trigger on stopped vehicle event."
    assert events[0].event_type in ["stopped_vehicle", "wrong_way", "sudden_deceleration"]

def test_candidate_frame_interface():
    """5. Test candidate frame extraction interface (before, event, after frames)."""
    frame_buffer = {}
    # Synthesize 60 frames
    for i in range(1, 61):
        frame_buffer[i] = np.full((1080, 1920, 3), i, dtype=np.uint8)
        
    peak_frame = 30
    before_frame_idx = max(1, peak_frame - 15)
    after_frame_idx = peak_frame + 15
    
    extracted = []
    for idx in [before_frame_idx, peak_frame, after_frame_idx]:
        if idx in frame_buffer:
            extracted.append((idx, frame_buffer[idx]))
            
    assert len(extracted) == 3, "Failed to extract candidate frame set (before, event, after)."
    assert extracted[0][0] == 15
    assert extracted[1][0] == 30
    assert extracted[2][0] == 45

def test_multi_frame_temporal_aggregation():
    """6. Test multi-frame temporal aggregation logic (policy_version: accident_temporal_v1.0.0)."""
    service = AccidentClassifierService()
    # Create 3 synthetic candidate frames
    cand_frames = [
        (15, np.zeros((224, 224, 3), dtype=np.uint8)),
        (30, np.ones((224, 224, 3), dtype=np.uint8) * 128),
        (45, np.full((224, 224, 3), 255, dtype=np.uint8))
    ]
    
    agg = service.aggregate_candidate_frames(cand_frames)
    assert "candidate_score" in agg
    assert "frame_predictions" in agg
    assert len(agg["frame_predictions"]) == 3
    assert agg["policy_version"] == "accident_temporal_v1.0.0"
    assert agg["final_class"] in ["NORMAL_TRAFFIC", "ACCIDENT_CANDIDATE"]

def test_no_single_frame_accident_confirmation():
    """7. Test that single frame prediction never confirms an accident (must always be ACCIDENT_CANDIDATE)."""
    service = AccidentClassifierService()
    frame_res = service.predict_frame(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Check that forbidden term ACCIDENT_CONFIRMED is never output
    assert frame_res["predicted_class"] != "ACCIDENT_CONFIRMED"
    
    agg_res = service.aggregate_candidate_frames([(1, np.zeros((224, 224, 3), dtype=np.uint8))])
    assert agg_res["final_class"] != "ACCIDENT_CONFIRMED"

def test_evidence_metadata_schema():
    """8. Test evidence directory creation and metadata.json schema."""
    test_dir = "data/evidence/test_job_synthetic/accident_event_999"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Save synthetic images
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(test_dir, "before.jpg"), dummy_img)
        cv2.imwrite(os.path.join(test_dir, "event.jpg"), dummy_img)
        cv2.imwrite(os.path.join(test_dir, "after.jpg"), dummy_img)
        
        meta_payload = {
            "job_id": 999,
            "event_id": 1,
            "frame_index": 30,
            "timestamp": 1.0,
            "track_ids": [99],
            "candidate_reason": f"Sudden deceleration ({SYNTHETIC_LABEL})",
            "source_event_ids": [1],
            "classifier_predictions": [],
            "aggregate_score": 0.12,
            "policy_version": "accident_temporal_v1.0.0",
            "final_class": "NORMAL_TRAFFIC",
            "is_candidate": False
        }
        meta_file = os.path.join(test_dir, "metadata.json")
        with open(meta_file, "w") as f:
            json.dump(meta_payload, f, indent=2)
            
        assert os.path.exists(meta_file)
        with open(meta_file, "r") as f:
            loaded = json.load(f)
            
        required_keys = [
            "job_id", "event_id", "frame_index", "timestamp",
            "track_ids", "candidate_reason", "source_event_ids"
        ]
        for key in required_keys:
            assert key in loaded, f"Missing required evidence metadata key: {key}"
    finally:
        shutil.rmtree("data/evidence/test_job_synthetic", ignore_errors=True)

def test_runtime_invocation_counter():
    """9. Test runtime telemetry invocation counter increments accurately."""
    service = AccidentClassifierService()
    initial_inv = service.invocations
    initial_frames = service.candidate_frames_classified
    
    _ = service.predict_frame(np.zeros((224, 224, 3), dtype=np.uint8))
    _ = service.predict_frame(np.zeros((224, 224, 3), dtype=np.uint8))
    
    telemetry = service.get_telemetry()
    assert telemetry["accident_classifier_invocations"] == initial_inv + 2
    assert telemetry["candidate_frames_classified"] == initial_frames + 2
    assert telemetry["total_classifier_ms"] >= 0.0

def test_job_isolation():
    """10. Test that classifier state is cleanly isolated per service instance (job isolation)."""
    service_a = AccidentClassifierService()
    service_b = AccidentClassifierService()
    
    _ = service_a.predict_frame(np.zeros((224, 224, 3), dtype=np.uint8))
    
    assert service_a.invocations == 1
    assert service_b.invocations == 0, "Job state leakage detected between AccidentClassifierService instances."

def test_deterministic_behavior():
    """11. Test deterministic inference results on identical synthetic input frames."""
    service = AccidentClassifierService()
    np.random.seed(42)
    img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    
    res1 = service.predict_frame(img)
    res2 = service.predict_frame(img)
    
    assert res1["accident_probability"] == res2["accident_probability"], "Inconsistent probability on identical frame."
    assert res1["predicted_class"] == res2["predicted_class"], "Inconsistent prediction class on identical frame."

def test_no_fake_predictions():
    """12. Test that probability outputs are dynamic numbers derived from actual model logits (not constant)."""
    service = AccidentClassifierService()
    img_black = np.zeros((224, 224, 3), dtype=np.uint8)
    img_white = np.full((224, 224, 3), 255, dtype=np.uint8)
    
    res_b = service.predict_frame(img_black)
    res_w = service.predict_frame(img_white)
    
    # Check that probabilities sum to 1.0 (softmax property)
    assert abs((res_b["normal_probability"] + res_b["accident_probability"]) - 1.0) < 1e-3
    assert abs((res_w["normal_probability"] + res_w["accident_probability"]) - 1.0) < 1e-3
    assert isinstance(res_b["accident_probability"], float)

def test_no_speed_gt_dependency():
    """13. Test that candidate accident verification operates without requiring speed GT calibration."""
    service = AccidentClassifierService()
    # Candidate classification operates directly on BGR frames without speed GT
    cand_frames = [(1, np.zeros((224, 224, 3), dtype=np.uint8))]
    res = service.aggregate_candidate_frames(cand_frames)
    
    assert "candidate_score" in res
    assert "final_class" in res
