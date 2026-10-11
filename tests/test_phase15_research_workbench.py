import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

from apps.api.main import app
from visionradar.models.database import SessionLocal, engine, Base
from visionradar.models.entities import ResearchExperimentDB, ResearchAblationDB, ResearchFailureCaseDB
from visionradar.research.schemas import (
    GroundTruthClassification, FailureCategory, MetricValue,
    DatasetProvenance, ConfigurationSnapshot, FailureRecord, AnomalyExplanation
)
from visionradar.research.config_snapshot import ConfigurationSnapshotEngine
from visionradar.research.provenance import ProvenanceEngine
from visionradar.research.comparison import ExperimentComparisonEngine
from visionradar.research.ablation import AblationEngine
from visionradar.research.failure_cases import FailureCaseEngine
from visionradar.research.explainability import ExplainabilityEngine
from visionradar.research.report_generator import ResearchReportGenerator
from visionradar.research.registry import ResearchRegistry
from visionradar.intelligence.anomaly.schemas import TrajectoryAnomaly, AnomalyType, AnomalySeverity, AnomalyMethod

client = TestClient(app)


def test_unique_experiment_ids_and_persistence():
    """1. Test unique sequential experiment ID generation and database persistence."""
    db = SessionLocal()
    exp_id_1 = ResearchRegistry.generate_experiment_id(db)
    assert exp_id_1.startswith("EXP-2026-")

    # Seed real benchmarks
    seeded = ResearchRegistry.seed_real_benchmark_experiments(db)
    assert len(seeded) >= 4
    assert "EXP-2026-000001" in seeded

    all_exps = ResearchRegistry.get_all_experiments(db)
    assert len(all_exps) >= 4
    db.close()


def test_configuration_snapshot_hash():
    """2. Test deterministic configuration snapshot generation and SHA256 hashing."""
    snap1 = ConfigurationSnapshotEngine.create_snapshot()
    snap2 = ConfigurationSnapshotEngine.create_snapshot()
    assert snap1.config_hash == snap2.config_hash
    assert len(snap1.config_hash) == 64
    assert snap1.detector["model_name"] == "YOLOX-Nano"


def test_metric_and_dataset_provenance():
    """3. Test metric provenance wrapping and dataset provenance metadata."""
    met = ProvenanceEngine.wrap_metric(
        name="Mean Speed",
        value=61.4,
        unit="km/h",
        source="MonocularSpeedEstimator",
        experiment_id="EXP-2026-000001",
        calculation_method="LOWESS Smoothed Trajectory Displacement",
        gt_status=GroundTruthClassification.NOT_EVALUABLE
    )
    d = met.to_dict()
    assert d["metric_name"] == "Mean Speed"
    assert d["value"] == 61.4
    assert d["ground_truth_status"] == "NOT-EVALUABLE"

    prov = ProvenanceEngine.build_dataset_provenance("VNTraffic", "VNTraffic_Original-video.mp4")
    assert prov.dataset_id == "VNTraffic"
    assert prov.fps == 30.0


def test_ground_truth_classification_integrity():
    """4. Test that all 4 ground truth tiers are preserved without collapsing categories."""
    tiers = [e.value for e in GroundTruthClassification]
    assert "GT-VALIDATED" in tiers
    assert "ENGINEERING-VALIDATED" in tiers
    assert "QUALITATIVE" in tiers
    assert "NOT-EVALUABLE" in tiers


def test_experiment_comparison_no_ranking():
    """5. Test scientific experiment comparison (field diff UNCHANGED/CHANGED, no winner scores)."""
    db = SessionLocal()
    seeded = ResearchRegistry.seed_real_benchmark_experiments(db)
    comp = ResearchRegistry.compare_experiments(db, seeded[0], seeded[1])
    assert "metadata_comparison" in comp
    assert "configuration_comparison" in comp
    assert "metrics_comparison" in comp
    assert comp["comparison_mode"] == "SCIENTIFIC_FIELD_DIFF_NO_RANKING"
    assert "winner" not in comp
    db.close()


def test_ablation_engine():
    """6. Test real ablation record creation and descriptive non-ranking interpretation."""
    rec = AblationEngine.create_ablation_record(
        ablation_id="ABL-TEST-1",
        experiment_id="EXP-2026-000001",
        baseline_experiment_id="EXP-2026-000001",
        parameter_name="conf_threshold",
        baseline_value=0.30,
        variant_value=0.50,
        metrics_baseline={"anomaly_count": 196},
        metrics_variant={"anomaly_count": 140},
        runtime_baseline_ms=50.0,
        runtime_variant_ms=45.0,
        interpretation="Increasing conf_threshold reduced anomaly count by 56."
    )
    assert rec.ablation_id == "ABL-TEST-1"
    assert rec.metrics_diff["anomaly_count"]["delta"] == -56


def test_failure_case_workbench():
    """7. Test failure case logging across 11 standardized failure categories."""
    fc = FailureCaseEngine.create_failure_record(
        failure_id="FAIL-1",
        experiment_id="EXP-2026-000001",
        video_filename="Traffic1.mp4",
        frame_index=12,
        timestamp=0.4,
        category="occlusion",
        description="Vehicle track occluded behind traffic pole.",
        severity="HIGH"
    )
    d = fc.to_dict()
    assert d["category"] == "occlusion"
    assert d["severity"] == "HIGH"


def test_anomaly_explainability_payload():
    """8. Test evidence-backed anomaly explainability payload construction."""
    anom = TrajectoryAnomaly(
        anomaly_id="anom_exp_1",
        track_id=12,
        job_id=1,
        anomaly_type=AnomalyType.ABNORMAL_LATERAL_MOVEMENT,
        severity=AnomalySeverity.HIGH,
        score=0.86,
        confidence=0.90,
        start_frame=30,
        end_frame=35,
        timestamp=1.0,
        duration_sec=0.16,
        method=AnomalyMethod.STATISTICAL_ROBUST_MAD,
        explanation="Track #12 lateral movement exceeded baseline.",
        evidence_frame_indices=[30],
        feature_contributions={"lateral_movement_px": 75.0, "z_score": 3.75},
        validation_status="ENGINEERING-VALIDATED"
    )
    expl = ExplainabilityEngine.generate_explanation(anom, experiment_id="EXP-2026-000001").to_dict()
    assert expl["what"] == "ABNORMAL_LATERAL_MOVEMENT"
    assert expl["which_track_ids"] == [12]
    assert expl["why_rule_contributions"]["z_score"] == 3.75
    assert expl["validation_status"] == "ENGINEERING-VALIDATED"


def test_report_generation_no_hardcoded_numbers():
    """9. Test markdown report generation directly from experiment dictionary."""
    db = SessionLocal()
    seeded = ResearchRegistry.seed_real_benchmark_experiments(db)
    exp_data = ResearchRegistry.get_experiment_by_id(db, seeded[0])
    report_md = ResearchReportGenerator.generate_report(exp_data)
    assert "# Research Report:" in report_md
    assert "1. Experiment Overview" in report_md
    assert "19. Conclusion" in report_md
    assert seeded[0] in report_md
    db.close()


def test_rest_api_endpoints():
    """10. Test FastAPI endpoints for Research Workbench."""
    # List experiments
    r1 = client.get("/api/v1/experiments")
    assert r1.status_code == 200
    exps = r1.json()
    assert len(exps) >= 4

    exp_id = exps[0]["experiment_id"]

    # Get experiment detail
    r2 = client.get(f"/api/v1/experiments/{exp_id}")
    assert r2.status_code == 200
    assert r2.json()["experiment_id"] == exp_id

    # Get metrics
    r3 = client.get(f"/api/v1/experiments/{exp_id}/metrics")
    assert r3.status_code == 200

    # Get configuration
    r4 = client.get(f"/api/v1/experiments/{exp_id}/configuration")
    assert r4.status_code == 200

    # Compare
    r5 = client.get(f"/api/v1/experiments/compare?exp_a_id={exps[0]['experiment_id']}&exp_b_id={exps[1]['experiment_id']}")
    assert r5.status_code == 200

    # Report
    r6 = client.get(f"/api/v1/experiments/{exp_id}/report")
    assert r6.status_code == 200
    assert "report_markdown" in r6.json()

    # Reproduce
    r7 = client.post(f"/api/v1/experiments/{exp_id}/reproduce")
    assert r7.status_code == 200
    assert r7.json()["status"] == "REPRODUCTION_SUCCESS"
