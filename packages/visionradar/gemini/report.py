"""
VisionRadar Gemini Multimodal Intelligence Service: Report Synthesizer.
Synthesizes comprehensive proctoring audit reports that strictly distinguish:
1. Local deterministic computer-vision detector observations
2. Gemini multimodal contextual interpretations
3. Human proctor disciplinary decisions
"""

from typing import Dict, Any, List, Optional
import datetime

class GeminiReportService:
    """Produces multi-perspective proctoring audit reports."""

    @staticmethod
    def build_multimodal_proctoring_report(
        job_metadata: Dict[str, Any],
        local_report: Dict[str, Any],
        gemini_analysis: Optional[Dict[str, Any]] = None,
        proctor_review_status: str = "PENDING",
        proctor_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Combines local detector findings with Gemini interpretations and human decisions.
        Preserves provenance for every finding.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Local Deterministic CV Findings
        local_primary = local_report.get("primary_candidate_finding") or {}
        local_all_events = local_report.get("all_candidate_events") or []

        # 2. Gemini Multimodal Interpretations
        gemini_meta = {}
        gemini_findings = []
        gemini_scene_summary = "Gemini multimodal contextual analysis not performed or unavailable."
        contradictory_evidence = []
        proctor_recommendation = "HUMAN_REVIEW_RECOMMENDED"

        if gemini_analysis and "analysis" in gemini_analysis:
            g_data = gemini_analysis["analysis"]
            gemini_scene_summary = g_data.get("scene_summary", gemini_scene_summary)
            gemini_findings = g_data.get("candidate_findings", [])
            contradictory_evidence = g_data.get("contradictory_evidence_notes", [])
            proctor_recommendation = g_data.get("proctor_action_recommendation", proctor_recommendation)
            gemini_meta = {
                "model_identifier": gemini_analysis.get("model", "unknown"),
                "latency_sec": gemini_analysis.get("latency_sec", 0.0),
                "cached": gemini_analysis.get("cached", False),
                "sampled_frame_count": gemini_analysis.get("sampled_frame_count", 0),
                "usage_metadata": gemini_analysis.get("usage_metadata", {})
            }

        # 3. Provenance Mapping: Correlate local events with Gemini assessments
        evidence_records: List[Dict[str, Any]] = []
        gemini_finding_by_event_id = {f["event_id"]: f for f in gemini_findings if "event_id" in f}

        for loc_evt in local_all_events:
            e_id = loc_evt.get("event_id")
            g_eval = gemini_finding_by_event_id.get(e_id, {})

            combined_record = {
                "event_id": e_id,
                "event_type": loc_evt.get("event_type"),
                "person_track_id": loc_evt.get("person_track_id"),
                "start_time": loc_evt.get("start_time"),
                "end_time": loc_evt.get("end_time"),
                "supporting_frames": loc_evt.get("supporting_frames", []),
                # Provenance Layer 1: Local CV Detector
                "local_cv_evidence": {
                    "detection_confidence": loc_evt.get("detection_confidence"),
                    "association_confidence": loc_evt.get("association_confidence"),
                    "heuristic_evidence_score": loc_evt.get("heuristic_evidence_score"),
                    "evidence_strength": loc_evt.get("evidence_strength"),
                    "local_observations": loc_evt.get("supporting_observations"),
                    "local_uncertainties": loc_evt.get("uncertainty_reasons", [])
                },
                # Provenance Layer 2: Gemini Multimodal Interpretation
                "gemini_multimodal_evidence": {
                    "evaluated_by_gemini": bool(g_eval),
                    "observed_behaviour": g_eval.get("observed_behaviour", "No secondary Gemini assessment requested for this event"),
                    "contextual_assessment": g_eval.get("contextual_assessment"),
                    "gemini_uncertainties": g_eval.get("uncertainty_notes", []),
                    "gemini_visibility_warnings": g_eval.get("visibility_warnings", []),
                    "recommended_review_priority": g_eval.get("recommended_review_priority", loc_evt.get("priority", "AMBER")),
                    "heuristic_relevance_score": g_eval.get("heuristic_relevance_score")
                },
                # Provenance Layer 3: Human Proctor Decision
                "human_proctor_review": {
                    "review_status": loc_evt.get("review_status", proctor_review_status),
                    "reviewer_notes": loc_evt.get("reviewer_notes", proctor_notes),
                    "decision_locked": loc_evt.get("review_status", "PENDING") != "PENDING"
                }
            }
            evidence_records.append(combined_record)

        return {
            "report_version": "gemini-proctoring-audit-v2.0",
            "generated_at": now_iso,
            "disclaimer": (
                "REGULATORY NOTICE: This report combines automated local computer vision with Google Gemini "
                "multimodal contextual analysis. All model outputs are candidate visual observations, NOT legal "
                "determinations of academic misconduct. Final disciplinary authority resides exclusively with human proctors."
            ),
            "video_metadata": job_metadata,
            "local_cv_pipeline_version": local_report.get("configuration", {}),
            "gemini_multimodal_service": {
                "active": bool(gemini_analysis),
                "metadata": gemini_meta,
                "scene_summary": gemini_scene_summary,
                "contradictory_evidence_notes": contradictory_evidence,
                "proctor_action_recommendation": proctor_recommendation
            },
            "candidate_events_provenance": evidence_records,
            "human_proctor_audit_record": {
                "overall_review_status": proctor_review_status,
                "proctor_notes": proctor_notes,
                "audit_timestamp": now_iso
            }
        }

# Global report service
gemini_reporter = GeminiReportService()
