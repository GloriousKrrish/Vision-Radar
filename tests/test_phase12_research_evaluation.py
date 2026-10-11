"""
VisionRadar Phase 12 Research-Grade Evaluation Test Suite.

Verifies:
1. Ground-Truth Taxonomy & Inventory schema and usability status.
2. Machine-Readable Phase 12 results schema and reproducibility metadata.
3. Speed GT remains explicitly BLOCKED.
4. Object Detection benchmark metrics (Precision, Recall, F1, mAP50).
5. ByteTrack Multi-Object Tracking benchmark metrics (Track Recall, IDF1, MOTA).
6. Calibration perturbation sensitivity metrics.
7. Detector threshold sensitivity metrics.
8. Resolution performance scaling benchmark metrics.
9. Frame-skip temporal ablation metrics.
10. Failure case mining schema and categorization.
11. Supported vs Unsupported research claims discipline.
12. Model hash consistency.
"""

import os
import sys
import json
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(BASE_DIR, "reports/phase12")

GT_INV_PATH = os.path.join(REPORTS_DIR, "ground_truth_inventory.json")
RESULTS_PATH = os.path.join(REPORTS_DIR, "phase12_results.json")
FAILURES_PATH = os.path.join(REPORTS_DIR, "failure_cases.json")


def test_phase12_artifacts_exist():
    """Verify all required Phase 12 artifacts exist on disk."""
    assert os.path.exists(GT_INV_PATH), f"Missing {GT_INV_PATH}"
    assert os.path.exists(RESULTS_PATH), f"Missing {RESULTS_PATH}"
    assert os.path.exists(FAILURES_PATH), f"Missing {FAILURES_PATH}"
    assert os.path.exists(os.path.join(REPORTS_DIR, "PHASE_12_PRE_AUDIT.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "PHASE_12_RESEARCH_EVALUATION_REPORT.md"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "figures/fig1_precision_recall_curve.png"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "figures/fig2_fps_vs_resolution.png"))
    assert os.path.exists(os.path.join(REPORTS_DIR, "figures/fig3_calibration_perturbation_sensitivity.png"))


def test_ground_truth_inventory_schema():
    """Verify ground_truth_inventory.json schema and taxonomies."""
    with open(GT_INV_PATH, "r") as f:
        data = json.load(f)

    assert "detection_gt" in data
    assert "tracking_gt" in data
    assert "event_gt" in data
    assert "speed_gt" in data
    assert "missing_required_assets" in data

    assert len(data["detection_gt"]) >= 2
    assert len(data["tracking_gt"]) >= 1
    assert data["speed_gt"]["status"] == "BLOCKED"
    assert "unavailable" in data["speed_gt"]["reason"].lower()


def test_speed_gt_remains_blocked():
    """Verify Speed Ground-Truth remains explicitly BLOCKED in results."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    speed_info = data.get("speed_gt", {})
    assert speed_info.get("status") == "BLOCKED"
    assert "unavailable" in speed_info.get("reason", "").lower()

    unsupported = data.get("claims_discipline", {}).get("unsupported", [])
    assert any("speed" in c.lower() for c in unsupported)


def test_detection_benchmark_metrics():
    """Verify Object Detection benchmark values and per-class metrics."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    det = data["detection_benchmark"]
    assert det["samples"] >= 80
    assert 0.0 <= det["precision"] <= 1.0
    assert 0.0 <= det["recall"] <= 1.0
    assert 0.0 <= det["f1"] <= 1.0
    assert 0.0 <= det["map50"] <= 1.0
    assert det["tp"] > 0
    assert "per_class" in det
    assert "car" in det["per_class"]
    assert "bus" in det["per_class"]
    assert "truck" in det["per_class"]
    assert "motorcycle" in det["per_class"]


def test_tracking_benchmark_metrics():
    """Verify ByteTrack tracking benchmark values."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    trk = data["tracking_benchmark"]
    assert trk["total_frames_evaluated"] >= 100
    assert trk["unique_tracks_count"] > 0
    assert 0.0 <= trk["track_recall"] <= 1.0
    assert 0.0 <= trk["id_precision"] <= 1.0
    assert 0.0 <= trk["idf1"] <= 1.0
    assert 0.0 <= trk["mota"] <= 1.0


def test_calibration_perturbation_sensitivity():
    """Verify calibration perturbation sensitivity experiment metrics."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    calib = data["calibration_perturbation_sensitivity"]
    assert len(calib) >= 5
    baseline = calib[0]
    assert baseline["perturbation_px"] == 0
    assert baseline["projected_pos_shift_m"] == 0.0

    # Monotonic shift increase with perturbation
    shifts = [c["projected_pos_shift_m"] for c in calib]
    assert shifts == sorted(shifts)


def test_threshold_sensitivity_experiment():
    """Verify confidence threshold sensitivity experiment."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    thresh = data["threshold_sensitivity"]
    assert len(thresh) >= 5
    t_vals = [t["confidence_threshold"] for t in thresh]
    assert 0.25 in t_vals
    assert 0.50 in t_vals


def test_resolution_performance_benchmark():
    """Verify resolution latency and FPS scaling benchmark."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    res_bench = data["resolution_benchmark"]
    assert len(res_bench) >= 4
    res_names = [r["input_resolution"] for r in res_bench]
    assert "320x320" in res_names
    assert "640x640" in res_names

    # 320x320 FPS should be faster than 640x640 FPS
    fps_320 = [r["ai_processing_fps"] for r in res_bench if r["input_resolution"] == "320x320"][0]
    fps_640 = [r["ai_processing_fps"] for r in res_bench if r["input_resolution"] == "640x640"][0]
    assert fps_320 > fps_640


def test_failure_cases_schema():
    """Verify failure case mining schema and entries."""
    with open(FAILURES_PATH, "r") as f:
        cases = json.load(f)

    assert len(cases) >= 5
    for case in cases:
        assert "case_id" in case
        assert "category" in case
        assert "video" in case
        assert "description" in case
        assert "impact" in case
        assert "mitigation_strategy" in case


def test_model_hash_reproducibility():
    """Verify model hash reproducibility metadata."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["model_sha256"] == "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
    assert meta["status"] == "PASS"
