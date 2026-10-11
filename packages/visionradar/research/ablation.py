"""
VisionRadar Phase 15 — Ablation Workbench Engine

Executes and records real ablation experiments (sensitivity sweeps, parameter variants,
context filtering toggles) with immutable record logging and descriptive interpretations.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from visionradar.research.schemas import AblationRecord


class AblationEngine:
    """
    Manages ablation experiments across detector, tracker, speed, and anomaly configurations.
    """

    @staticmethod
    def create_ablation_record(
        ablation_id: str,
        experiment_id: str,
        baseline_experiment_id: str,
        parameter_name: str,
        baseline_value: Any,
        variant_value: Any,
        metrics_baseline: Dict[str, Any],
        metrics_variant: Dict[str, Any],
        runtime_baseline_ms: float,
        runtime_variant_ms: float,
        interpretation: str
    ) -> AblationRecord:

        metrics_diff = {}
        all_keys = set(metrics_baseline.keys()).union(set(metrics_variant.keys()))
        for k in sorted(all_keys):
            vb = metrics_baseline.get(k)
            vv = metrics_variant.get(k)
            if isinstance(vb, (int, float)) and isinstance(vv, (int, float)):
                metrics_diff[k] = {
                    "baseline": vb,
                    "variant": vv,
                    "delta": round(vv - vb, 4)
                }
            else:
                metrics_diff[k] = {
                    "baseline": vb,
                    "variant": vv,
                    "delta": "NON_NUMERIC"
                }

        runtime_diff_ms = round(runtime_variant_ms - runtime_baseline_ms, 2)

        return AblationRecord(
            ablation_id=ablation_id,
            experiment_id=experiment_id,
            baseline_experiment_id=baseline_experiment_id,
            parameter_name=parameter_name,
            baseline_value=baseline_value,
            variant_value=variant_value,
            metrics_diff=metrics_diff,
            runtime_diff_ms=runtime_diff_ms,
            interpretation=interpretation
        )

    @staticmethod
    def generate_descriptive_interpretation(
        param_name: str,
        base_val: Any,
        var_val: Any,
        metrics_diff: Dict[str, Any],
        runtime_diff_ms: float
    ) -> str:
        """
        Generates objective, non-ranking descriptive interpretation of ablation results.
        """
        desc_parts = [f"Modifying '{param_name}' from {base_val} to {var_val}:"]
        
        if "anomaly_count" in metrics_diff:
            diff_item = metrics_diff["anomaly_count"]
            delta = diff_item.get("delta", 0)
            if delta > 0:
                desc_parts.append(f"Increased detected anomaly count by {delta}.")
            elif delta < 0:
                desc_parts.append(f"Decreased detected anomaly count by {abs(delta)}.")
            else:
                desc_parts.append("Anomaly detection count remained constant.")

        if abs(runtime_diff_ms) > 0.01:
            speed_str = "increased" if runtime_diff_ms > 0 else "decreased"
            desc_parts.append(f"Execution runtime {speed_str} by {abs(runtime_diff_ms):.2f} ms.")

        return " ".join(desc_parts)
