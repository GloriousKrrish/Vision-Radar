"""
VisionRadar Phase 15 — Experiment Comparison Engine

Performs rigorous side-by-side scientific comparison between Experiment A and Experiment B.
Categorizes configuration and metric fields as UNCHANGED, CHANGED, or NOT AVAILABLE.
NO winner scores, NO rankings, NO subjective judgment.
"""

from typing import Dict, Any, Optional


class ExperimentComparisonEngine:
    """
    Compares two experiment manifests field by field for scientific analysis.
    """

    @staticmethod
    def compare(exp_a: Dict[str, Any], exp_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates field-by-field differences between Experiment A and Experiment B.
        """
        def diff_field(val_a: Any, val_b: Any) -> Dict[str, Any]:
            if val_a is None and val_b is None:
                return {"status": "NOT AVAILABLE", "val_a": None, "val_b": None}
            elif val_a == val_b:
                return {"status": "UNCHANGED", "val_a": val_a, "val_b": val_b}
            else:
                return {"status": "CHANGED", "val_a": val_a, "val_b": val_b}

        # 1. Metadata Comparison
        metadata_diff = {
            "experiment_id": diff_field(exp_a.get("experiment_id"), exp_b.get("experiment_id")),
            "name": diff_field(exp_a.get("name"), exp_b.get("name")),
            "dataset_name": diff_field(exp_a.get("dataset_name"), exp_b.get("dataset_name")),
            "video_filename": diff_field(exp_a.get("video_filename"), exp_b.get("video_filename")),
            "model_id": diff_field(exp_a.get("model_id"), exp_b.get("model_id")),
            "software_version": diff_field(exp_a.get("software_version"), exp_b.get("software_version")),
            "ground_truth_status": diff_field(exp_a.get("ground_truth_status"), exp_b.get("ground_truth_status"))
        }

        # 2. Configuration Comparison
        cfg_a = exp_a.get("configuration", {}) or exp_a.get("configuration_json", {}) or {}
        cfg_b = exp_b.get("configuration", {}) or exp_b.get("configuration_json", {}) or {}

        config_diff = {
            "detector": diff_field(cfg_a.get("detector"), cfg_b.get("detector")),
            "tracker": diff_field(cfg_a.get("tracker"), cfg_b.get("tracker")),
            "speed": diff_field(cfg_a.get("speed"), cfg_b.get("speed")),
            "traffic_intelligence": diff_field(cfg_a.get("traffic_intelligence"), cfg_b.get("traffic_intelligence")),
            "anomaly_intelligence": diff_field(cfg_a.get("anomaly_intelligence"), cfg_b.get("anomaly_intelligence")),
            "accident_intelligence": diff_field(cfg_a.get("accident_intelligence"), cfg_b.get("accident_intelligence"))
        }

        # 3. Metrics Comparison
        met_a = exp_a.get("metrics", {}) or exp_a.get("metrics_json", {}) or {}
        met_b = exp_b.get("metrics", {}) or exp_b.get("metrics_json", {}) or {}

        all_metric_keys = set(met_a.keys()).union(set(met_b.keys()))
        metrics_diff = {}
        for k in sorted(all_metric_keys):
            va = met_a.get(k)
            vb = met_b.get(k)
            diff_item = diff_field(va, vb)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diff_item["numeric_delta"] = round(vb - va, 4)
            metrics_diff[k] = diff_item

        return {
            "experiment_a_id": exp_a.get("experiment_id", "EXP_A"),
            "experiment_b_id": exp_b.get("experiment_id", "EXP_B"),
            "metadata_comparison": metadata_diff,
            "configuration_comparison": config_diff,
            "metrics_comparison": metrics_diff,
            "comparison_mode": "SCIENTIFIC_FIELD_DIFF_NO_RANKING"
        }
