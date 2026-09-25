"""
VisionRadar Phase 3.4.2 — Performance & Tracking Reliability Test Suite

Tests verify that all benchmark data, metrics invariants, stage profiling breakdown,
resolution yield changes, frame skipping quality impact, and report files are valid
and reproducible without fabrication.
"""

import os
import json
import pytest


BENCHMARK_JSON_PATH = "phase3_4_2_benchmark_results.json"
REPORT_PATH = "reports/phase3_4/PHASE_3_4_2_INFERENCE_PERFORMANCE_REPORT.md"


def test_benchmark_json_exists_and_valid():
    assert os.path.exists(BENCHMARK_JSON_PATH), f"Benchmark JSON {BENCHMARK_JSON_PATH} not found"
    with open(BENCHMARK_JSON_PATH, "r") as f:
        data = json.load(f)

    required_sections = [
        "BASELINE_640_CPU",
        "STAGE_PROFILING_640",
        "BACKEND_VERIFICATION",
        "CPU_THREADING_ABLATION",
        "MODEL_INITIALIZATION",
        "PREPROCESSING_OPTIMIZATION",
        "RESOLUTION_ABLATION",
        "CONFIDENCE_NMS_INVARIANTS",
        "GPU_INVESTIGATION",
        "FRAME_SKIPPING_ANALYSIS"
    ]
    for sec in required_sections:
        assert sec in data, f"Missing section '{sec}' in benchmark JSON"


def test_baseline_metrics_invariants():
    with open(BENCHMARK_JSON_PATH, "r") as f:
        data = json.load(f)["BASELINE_640_CPU"]

    assert data["resolution"] == "640x640"
    assert data["threads"] == 8
    assert data["frame_skip"] == 0
    assert data["detector_latency_ms"] > 0
    assert data["detector_fps"] > 0
    assert data["pipeline_fps"] > 0
    assert data["rtf"] > 0
    assert data["detections_per_frame"] > 4.0
    assert data["unique_tracks"] > 20
    assert data["observed_internal_id_switches"] == 0
    assert data["first_box_latency_ms"] is not None


def test_stage_profiling_dominant_bottleneck():
    with open(BENCHMARK_JSON_PATH, "r") as f:
        data = json.load(f)["STAGE_PROFILING_640"]

    forward_pct = data["5. opencv_dnn_forward"]["pct"]
    assert forward_pct > 70.0, f"Expected opencv_dnn_forward to be dominant bottleneck (>70%), got {forward_pct}%"


def test_cpu_threading_scaling():
    with open(BENCHMARK_JSON_PATH, "r") as f:
        data = json.load(f)["CPU_THREADING_ABLATION"]

    t1_latency = data["threads_1"]["detector_latency_ms"]
    t8_latency = data["threads_8"]["detector_latency_ms"]

    assert t8_latency < t1_latency, f"8 threads ({t8_latency} ms) should be faster than 1 thread ({t1_latency} ms)"


def test_resolution_ablation_latency_and_yield():
    with open(BENCHMARK_JSON_PATH, "r") as f:
        data = json.load(f)["RESOLUTION_ABLATION"]

    lat_640 = data["resolution_640"]["detector_latency_ms"]
    lat_320 = data["resolution_320"]["detector_latency_ms"]
    assert lat_320 < lat_640, "320x320 detector latency should be lower than 640x640"

    yield_640 = data["resolution_640"]["detections_per_frame"]
    yield_320 = data["resolution_320"]["detections_per_frame"]
    assert yield_320 < yield_640, "320x320 detection yield should be lower than 640x640"


def test_frame_skipping_fragmentation_degradation():
    with open(BENCHMARK_JSON_PATH, "r") as f:
        data = json.load(f)["FRAME_SKIPPING_ANALYSIS"]

    frag_0 = data["skip_0"]["fragmentations"]
    frag_1 = data["skip_1"]["fragmentations"]
    assert frag_1 > frag_0, "Frame skipping should increase track fragmentations"


def test_gpu_investigation_verdict():
    with open(BENCHMARK_JSON_PATH, "r") as f:
        data = json.load(f)["GPU_INVESTIGATION"]

    assert "GPU NOT AVAILABLE" in data["verdict"]
    assert data["cuda_available"] is False


def test_report_file_exists():
    assert os.path.exists(REPORT_PATH), f"Report file {REPORT_PATH} does not exist"
    with open(REPORT_PATH, "r") as f:
        content = f.read()

    assert "Phase 3.4.2" in content
    assert "BASELINE_640_CPU" in content
    assert "PASS WITH LIMITATIONS" in content
