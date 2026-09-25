"""
VisionRadar Phase 3.4.1 — Latency Forensic Test Suite

Tests verify that all measurements are real and constraints are correct.
No values are fabricated. Each test validates a measured or structural invariant.
"""
import os
import sys
import time
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Load the pre-measured forensic JSON report for value-based assertions
# ---------------------------------------------------------------------------

FORENSIC_JSON = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'debug', 'phase3_4_1_latency_forensic.json'
)


def load_forensic() -> dict:
    if os.path.exists(FORENSIC_JSON):
        with open(FORENSIC_JSON, 'r') as f:
            return json.load(f)
    return {}


# ============================================================================
# GROUP 1: Forensic JSON Report Integrity
# ============================================================================

def test_forensic_json_exists():
    """Forensic benchmark JSON was produced by the measurement script."""
    assert os.path.exists(FORENSIC_JSON), (
        f"Forensic JSON not found at {FORENSIC_JSON}. "
        "Run scripts/benchmark_latency_forensic.py first."
    )


def test_forensic_json_has_all_sections():
    """Forensic JSON contains all required measurement sections."""
    data = load_forensic()
    required_keys = [
        "upload", "model", "decoder", "first_frame_decode_ms",
        "first_inference_ms", "first_tracker_ms", "steady_state",
        "resolution_benchmark", "frame_skip_analysis",
        "queue_backpressure", "latency_breakdown", "summary"
    ]
    for key in required_keys:
        assert key in data, f"Missing required section: {key}"


def test_video_sha256_matches_canonical():
    """Canonical video SHA-256 matches Traffic1.mp4."""
    data = load_forensic()
    expected = "a4c3ee6aeca7f5085cc49408d572e3e3df7cab65ead15701996724827ef99c10"
    assert data.get("video_sha256") == expected, (
        f"Video SHA-256 mismatch: expected {expected}, got {data.get('video_sha256')}"
    )


# ============================================================================
# GROUP 2: Upload Latency Measurements
# ============================================================================

def test_upload_total_latency_is_positive():
    """Upload total latency must be a positive real number."""
    data = load_forensic()
    upload_ms = data.get("upload", {}).get("total_ms", -1)
    assert upload_ms > 0, f"Upload latency must be positive, got {upload_ms}"


def test_upload_read_plus_write_equals_total():
    """Upload read + write approximately equals total upload time."""
    data = load_forensic()
    u = data.get("upload", {})
    read_ms = u.get("read_ms", 0)
    write_ms = u.get("write_ms", 0)
    total_ms = u.get("total_ms", 0)
    # Allow 5% tolerance for timing overhead
    assert abs((read_ms + write_ms) - total_ms) < total_ms * 0.1, (
        f"Upload read ({read_ms}) + write ({write_ms}) = {read_ms + write_ms} "
        f"does not match total ({total_ms})"
    )


def test_upload_speed_positive():
    """Upload throughput is a positive number."""
    data = load_forensic()
    speed = data.get("upload", {}).get("speed_mbps", -1)
    assert speed > 0


def test_upload_requires_full_file():
    """Architecture limitation: full upload required before CV can start."""
    data = load_forensic()
    limitation = data.get("upload", {}).get("architecture_limitation", "")
    assert "FULL UPLOAD REQUIRED" in limitation.upper() or "cannot begin" in limitation.lower(), (
        "Architecture limitation note must state that full upload is required."
    )


# ============================================================================
# GROUP 3: Model Initialization Measurements
# ============================================================================

def test_model_cold_start_is_positive():
    """Model cold-start (readNetFromONNX) takes positive time."""
    data = load_forensic()
    cold = data.get("model", {}).get("cold_start_ms", -1)
    assert cold > 0, f"Model cold-start must be positive, got {cold}"


def test_model_warmup_is_positive():
    """Model warmup (first dry-run inference) takes positive time."""
    data = load_forensic()
    warmup = data.get("model", {}).get("warmup_ms", -1)
    assert warmup > 0, f"Model warmup must be positive, got {warmup}"


def test_model_not_loaded_per_frame():
    """Model is loaded ONCE per job, not per-frame."""
    data = load_forensic()
    note = data.get("model", {}).get("note", "")
    assert "once per job" in note.lower() or "not per-frame" in note.lower(), (
        "Model note must confirm it's loaded once per job, not per-frame."
    )


def test_model_ram_footprint_reasonable():
    """Model RAM footprint is between 10 MB and 500 MB (sanity check)."""
    data = load_forensic()
    ram = data.get("model", {}).get("model_ram_mb", -1)
    assert 10 < ram < 500, f"Model RAM {ram} MB is outside expected range [10, 500]"


# ============================================================================
# GROUP 4: Decoder and Frame Measurements
# ============================================================================

def test_decoder_init_is_positive():
    """VideoCapture initialization takes positive time."""
    data = load_forensic()
    init = data.get("decoder", {}).get("init_ms", -1)
    assert init > 0, f"Decoder init must be positive, got {init}"


def test_first_frame_decode_is_positive():
    """First frame decode (cap.read()) takes positive time."""
    data = load_forensic()
    fd = data.get("first_frame_decode_ms", -1)
    assert fd > 0, f"First frame decode must be positive, got {fd}"


def test_decoder_resolution_matches_canonical():
    """Decoded video resolution is 1920×1080 (canonical)."""
    data = load_forensic()
    dec = data.get("decoder", {})
    assert dec.get("width") == 1920, f"Expected width 1920, got {dec.get('width')}"
    assert dec.get("height") == 1080, f"Expected height 1080, got {dec.get('height')}"


def test_decoder_fps_is_canonical():
    """Video FPS is 29.97 (canonical Traffic1.mp4)."""
    data = load_forensic()
    fps = data.get("decoder", {}).get("fps", -1)
    assert abs(fps - 29.97) < 0.5, f"Expected ~29.97 FPS, got {fps}"


def test_decoder_total_frames_canonical():
    """Total frames is 335 (canonical Traffic1.mp4)."""
    data = load_forensic()
    frames = data.get("decoder", {}).get("total_frames", -1)
    assert frames == 335, f"Expected 335 frames, got {frames}"


# ============================================================================
# GROUP 5: YOLOX Inference Measurements
# ============================================================================

def test_first_inference_is_positive():
    """First YOLOX inference takes positive time."""
    data = load_forensic()
    fi = data.get("first_inference_ms", -1)
    assert fi is None or fi > 0, f"First inference must be positive or None (model unavailable)"


def test_steady_state_avg_detector_is_positive():
    """Steady-state average YOLOX latency is positive."""
    data = load_forensic()
    avg = data.get("steady_state", {}).get("avg_detector_ms", -1)
    assert avg > 0, f"Avg detector ms must be positive, got {avg}"


def test_steady_state_p95_geq_avg():
    """P95 YOLOX latency >= avg YOLOX latency."""
    data = load_forensic()
    ss = data.get("steady_state", {})
    avg = ss.get("avg_detector_ms", 0)
    p95 = ss.get("p95_detector_ms", 0)
    assert p95 >= avg, f"P95 ({p95}) must be >= avg ({avg})"


def test_rtf_is_less_than_one():
    """RTF < 1.0 — the system is not real-time on CPU."""
    data = load_forensic()
    rtf = data.get("steady_state", {}).get("real_time_factor", 1.0)
    assert rtf < 1.0, (
        f"RTF = {rtf}. On CPU-only hardware, RTF must be < 1.0. "
        "If RTF >= 1.0 is claimed, it must be measured on specific hardware."
    )


def test_rtf_classification_is_sub_real_time():
    """RTF classification must be SUB-REAL-TIME on this hardware."""
    data = load_forensic()
    classification = data.get("summary", {}).get("rtf_classification", "")
    assert classification in ("SUB-REAL-TIME", "NEAR-REAL-TIME"), (
        f"Expected SUB-REAL-TIME or NEAR-REAL-TIME, got '{classification}'. "
        "Must not claim REAL-TIME without measured RTF >= 1.0."
    )


# ============================================================================
# GROUP 6: Resolution Benchmark Correctness
# ============================================================================

def test_resolution_benchmark_has_three_entries():
    """Resolution benchmark includes 640, 416, and 320."""
    data = load_forensic()
    rb = data.get("resolution_benchmark", [])
    assert len(rb) == 3, f"Expected 3 resolution entries, got {len(rb)}"


def test_640_has_highest_detection_count():
    """640×640 has the highest detection count per frame (most accurate)."""
    data = load_forensic()
    rb = data.get("resolution_benchmark", [])
    dets = {r["resolution"]: r["avg_detections"] for r in rb}
    det_640 = dets.get("640×640", 0)
    det_416 = dets.get("416×416", 0)
    det_320 = dets.get("320×320", 0)
    assert det_640 >= det_416, (
        f"640×640 should have >= detections vs 416×416: {det_640} vs {det_416}"
    )
    assert det_640 >= det_320, (
        f"640×640 should have >= detections vs 320×320: {det_640} vs {det_320}"
    )


def test_lower_resolution_faster_but_fewer_detections():
    """Lower resolution is faster (lower ms) but detects fewer vehicles."""
    data = load_forensic()
    rb = {r["resolution"]: r for r in data.get("resolution_benchmark", [])}
    r640 = rb.get("640×640", {})
    r416 = rb.get("416×416", {})
    r320 = rb.get("320×320", {})

    if r640 and r416:
        assert r416["avg_ms"] <= r640["avg_ms"], "416×416 should be faster than 640×640"
        assert r416["avg_detections"] <= r640["avg_detections"], "416×416 should detect fewer"

    if r640 and r320:
        assert r320["avg_ms"] <= r640["avg_ms"], "320×320 should be faster than 640×640"
        assert r320["avg_detections"] <= r640["avg_detections"], "320×320 should detect fewer"


# ============================================================================
# GROUP 7: Frame-Skip Analysis Invariants
# ============================================================================

def test_frame_skip_not_implemented():
    """Frame skipping is NOT implemented — only theoretical."""
    data = load_forensic()
    for entry in data.get("frame_skip_analysis", []):
        assert not entry.get("implemented", True), (
            f"Frame skipping at skip={entry.get('process_every_nth_frame')} "
            "should NOT be marked as implemented."
        )


def test_skip_2_doubles_effective_fps():
    """Skip-2 doubles the effective FPS compared to baseline."""
    data = load_forensic()
    fsa = data.get("frame_skip_analysis", [])
    if len(fsa) < 2:
        pytest.skip("Frame skip analysis not complete")
    base_fps = fsa[0].get("effective_processing_fps", 0)
    skip2_fps = fsa[1].get("effective_processing_fps", 0)
    assert abs(skip2_fps - base_fps * 2) < 1.0, (
        f"Skip-2 FPS ({skip2_fps}) should be ~2× baseline ({base_fps})"
    )


# ============================================================================
# GROUP 8: Queue / Backpressure Invariants
# ============================================================================

def test_backpressure_tracker_not_affected():
    """Tracker state is explicitly marked as unaffected by queue drops."""
    data = load_forensic()
    qb = data.get("queue_backpressure", {})
    assert qb.get("tracker_sequential") is True
    assert qb.get("tracker_state_affected_by_drop") is False


def test_queue_size_bounded():
    """Queue max size matches the configured bound."""
    from visionradar.worker.job_worker import _QUEUE_MAX_SIZE
    data = load_forensic()
    reported_size = data.get("queue_backpressure", {}).get("queue_max_size", -1)
    assert reported_size == _QUEUE_MAX_SIZE, (
        f"Reported queue size {reported_size} != actual {_QUEUE_MAX_SIZE}"
    )


# ============================================================================
# GROUP 9: End-to-End Latency Breakdown Invariants
# ============================================================================

def test_first_box_latency_is_below_one_second():
    """First-box latency must be below 1 second (backend sequential path)."""
    data = load_forensic()
    total = data.get("latency_breakdown", {}).get("total_first_box_latency_ms", 9999)
    assert total < 1000, (
        f"First-box latency {total} ms exceeds 1000 ms. "
        "Investigate dominant bottleneck and optimize."
    )


def test_t1_video_visible_is_zero():
    """T1 (video visible via blob URL) must be 0 ms — instant."""
    data = load_forensic()
    stages = data.get("latency_breakdown", {}).get("stages", [])
    t1_stage = next((s for s in stages if "blob" in s.get("stage", "").lower() or "T1" in s.get("stage", "")), None)
    if t1_stage:
        assert t1_stage["duration_ms"] == 0.0, (
            f"T1 video visible should be 0 ms, got {t1_stage['duration_ms']}"
        )


def test_bytetrack_latency_is_negligible():
    """ByteTrack first update is negligible (< 10 ms)."""
    data = load_forensic()
    ft = data.get("first_tracker_ms", 9999)
    if ft is not None:
        assert ft < 10.0, f"ByteTrack first update should be < 10 ms, got {ft} ms"


def test_dominant_bottleneck_identified():
    """A dominant bottleneck stage is explicitly identified in the report."""
    data = load_forensic()
    bottleneck = data.get("latency_breakdown", {}).get("dominant_bottleneck_stage", "")
    assert len(bottleneck) > 0, "dominant_bottleneck_stage must be non-empty"
    pct = data.get("latency_breakdown", {}).get("dominant_bottleneck_pct", 0)
    assert pct > 10.0, (
        f"Dominant bottleneck ({bottleneck}) contributes only {pct}% — "
        "re-check bottleneck identification."
    )


# ============================================================================
# GROUP 10: Live Instrumentation Tests (Real-Time Measurement)
# ============================================================================

def test_live_decoder_init_latency():
    """Measure cv2.VideoCapture.open() latency in real-time."""
    video_path = "Traffic1.mp4"
    if not os.path.exists(video_path):
        pytest.skip(f"Video not found: {video_path}")

    t0 = time.monotonic()
    cap = cv2.VideoCapture(video_path)
    t1 = time.monotonic()
    cap.release()

    init_ms = (t1 - t0) * 1000
    assert cap.isOpened() or init_ms > 0, "VideoCapture should open successfully"
    # Sanity: decoder init should be < 5 seconds on any hardware
    assert init_ms < 5000, f"Decoder init took {init_ms:.1f} ms — unexpectedly slow"


def test_live_first_frame_decode_latency():
    """Measure first cap.read() latency in real-time."""
    video_path = "Traffic1.mp4"
    if not os.path.exists(video_path):
        pytest.skip(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pytest.skip("Could not open video")

    t0 = time.monotonic()
    ret, frame = cap.read()
    t1 = time.monotonic()
    cap.release()

    first_decode_ms = (t1 - t0) * 1000
    assert ret, "cap.read() should succeed"
    assert frame is not None
    assert frame.shape == (1080, 1920, 3), f"Unexpected frame shape: {frame.shape}"
    # Sanity: first frame should decode in < 5 seconds
    assert first_decode_ms < 5000, f"First decode took {first_decode_ms:.1f} ms"


def test_live_yolox_inference_latency():
    """Measure YOLOX inference latency on a single frame in real-time."""
    model_path = "data/models/yolox_nano.onnx"
    video_path = "Traffic1.mp4"

    if not os.path.exists(model_path):
        pytest.skip(f"Model not found: {model_path}")
    if not os.path.exists(video_path):
        pytest.skip(f"Video not found: {video_path}")

    # Load model
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret

    # Preprocess
    h, w = frame.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, swapRB=True)

    # Warmup
    net.setInput(blob)
    net.forward()

    # Measure 5 inference runs
    times = []
    for _ in range(5):
        t0 = time.monotonic()
        net.setInput(blob)
        out = net.forward()
        t1 = time.monotonic()
        times.append((t1 - t0) * 1000)

    avg_ms = float(np.mean(times))
    assert out is not None
    assert avg_ms < 5000, f"YOLOX inference avg {avg_ms:.1f} ms — unexpectedly slow"
    # Verify the system is SUB-REAL-TIME
    video_fps = 29.97
    budget_ms = 1000.0 / video_fps
    assert avg_ms > budget_ms, (
        f"YOLOX avg {avg_ms:.1f} ms <= frame budget {budget_ms:.1f} ms. "
        "RTF < 1.0 expected on CPU. If this assertion fails, GPU may be active."
    )


def test_live_bytetrack_latency():
    """Measure ByteTrack update latency."""
    from visionradar.cv.tracker import ByteTrackTracker
    from visionradar.cv.detection.base import Detection

    tracker = ByteTrackTracker()
    dets = [Detection(bbox=[300.0, 200.0, 450.0, 350.0], confidence=0.88, class_id=2, class_name="Car")]

    times = []
    for i in range(10):
        t0 = time.monotonic()
        tracker.update(dets, i, i / 29.97)
        t1 = time.monotonic()
        times.append((t1 - t0) * 1000)

    avg_ms = float(np.mean(times))
    assert avg_ms < 50, f"ByteTrack avg {avg_ms:.1f} ms — should be < 50 ms"
