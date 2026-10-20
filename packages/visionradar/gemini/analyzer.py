"""
VisionRadar Gemini Multimodal Intelligence Service: Analysis Service.
Coordinates bounded frame sampling, prompt formulation, multimodal inference,
caching, and strict schema validation for examination and traffic domains.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from visionradar.gemini.config import gemini_config
from visionradar.gemini.client import gemini_client
from visionradar.gemini.sampler import gemini_sampler
from visionradar.gemini.validator import gemini_validator
from visionradar.gemini.schemas import (
    GeminiExamAnalysisOutput,
    GeminiTrafficSummaryOutput
)

logger = logging.getLogger("visionradar.gemini.analyzer")

# In-memory LRU / cache for fast lookups
_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}

EXAMINATION_SYSTEM_INSTRUCTION = """You are an objective, evidence-based computer vision analyst for academic examination proctoring footage.
Your role is to provide neutral, factual observations about physical interactions visible in the provided video frames.

CRITICAL ETHICAL & FORENSIC CONSTRAINTS:
1. NEVER declare that a student cheated or committed misconduct. All findings are candidate visual observations for human proctor review.
2. NEVER infer intent, guilt, dishonesty, or academic status from student appearance, clothing, ethnicity, gaze direction, or normal body movement (e.g. looking down to write is normal examination activity).
3. Ground all observations exclusively in physical objects, posture, and person-object proximity visible in the provided timestamped frames.
4. If a mobile phone, tablet, or secondary reference material is visible, describe its physical location, whether it is held or resting on a surface, and its proximity to the person track.
5. If visual evidence is ambiguous, occluded by hands, or blurry, explicitly record this in uncertainty_notes and visibility_warnings.
6. Return your evaluation strictly in the requested JSON structure.
"""

TRAFFIC_SYSTEM_INSTRUCTION = """You are an objective traffic scene intelligence analyst.
Your role is to summarize road conditions, traffic density, congestion patterns, and observable events based on the provided video frames and local telemetry.

CRITICAL PRESERVATION CONSTRAINTS:
1. NEVER calculate, estimate, or guess vehicle speeds. Vehicle speed measurements are generated exclusively by VisionRadar's local homography calibration pipeline.
2. Rely only on visible traffic cues (vehicle types, queue formation, lane alignment, brake lights, weather).
3. Return your evaluation strictly in the requested JSON structure.
"""

class GeminiAnalysisService:
    """Provides high-level multimodal contextual analysis for VisionRadar jobs."""

    def __init__(self, client=None, sampler=None, validator=None, config=None):
        self.client = client or gemini_client
        self.sampler = sampler or gemini_sampler
        self.validator = validator or gemini_validator
        self.config = config or gemini_config

    def analyze_examination_context(
        self,
        job_id: int,
        video_path: str,
        local_candidate_events: List[Dict[str, Any]],
        local_tracks: List[Dict[str, Any]],
        force_refresh: bool = False,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs multimodal contextual analysis on an examination job.
        Selects relevant keyframes from candidate event timestamps, queries Gemini,
        validates the output, and caches the result.
        """
        # Determine candidate frames to inspect from local events
        supporting_frames: List[int] = []
        for evt in local_candidate_events:
            frames = evt.get("supporting_frames") or []
            supporting_frames.extend(frames)

        # Compute deterministic cache hash
        active_model = model_override or self.config.model
        cache_key = self.sampler.compute_cache_hash(
            job_id=job_id,
            module="examination",
            model=active_model,
            frame_indices=supporting_frames
        )

        if not force_refresh and cache_key in _ANALYSIS_CACHE:
            logger.info("Returning cached Gemini examination analysis for job #%d (hash: %s)", job_id, cache_key[:12])
            cached = _ANALYSIS_CACHE[cache_key].copy()
            cached["cached"] = True
            return cached

        # Sample bounded keyframes
        sampled_frames = self.sampler.sample_keyframes(
            video_path=video_path,
            target_frame_indices=supporting_frames,
            max_dimension=1024
        )

        if not sampled_frames:
            raise ValueError(f"No valid video frames could be extracted from {video_path}")

        # Formulate multimodal contents
        from google.genai import types

        prompt_text = (
            f"Analyze these {len(sampled_frames)} selected keyframes from examination job #{job_id}.\n"
            f"Local Computer Vision Findings:\n"
        )
        for evt in local_candidate_events:
            prompt_text += (
                f"- Candidate Event {evt.get('event_id')}: Type: {evt.get('event_type')}, "
                f"Track ID: #{evt.get('person_track_id')}, "
                f"Times: {evt.get('start_time')}s - {evt.get('end_time')}s, "
                f"Local Detection Conf: {evt.get('detection_confidence')}, "
                f"Heuristic Strength: {evt.get('evidence_strength')}\n"
            )

        prompt_text += (
            "\nProvide a structured JSON assessment conforming to the schema:\n"
            "{\n"
            '  "scene_summary": "factual overview of the visible examination area",\n'
            '  "environment_type": "CLASSROOM_EXAMINATION",\n'
            '  "candidate_findings": [\n'
            "    {\n"
            '      "event_id": "matching event_id from local findings",\n'
            '      "event_type": "event type name",\n'
            '      "observed_behaviour": "neutral factual description of observable action",\n'
            '      "contextual_assessment": "assessment of desk reach and device interaction",\n'
            '      "supporting_frame_references": [frame_indices],\n'
            '      "uncertainty_notes": ["reasons for ambiguity or limitations"],\n'
            '      "visibility_warnings": ["hand occlusion or blur warnings"],\n'
            '      "recommended_review_priority": "RED | AMBER | INFO",\n'
            '      "heuristic_relevance_score": 0.0 to 1.0\n'
            "    }\n"
            "  ],\n"
            '  "contradictory_evidence_notes": ["observations supporting normal exam activity"],\n'
            '  "primary_finding_event_id": "id of the event warranting most urgent proctor review",\n'
            '  "proctor_action_recommendation": "HUMAN_REVIEW_RECOMMENDED | NO_ACTION_REQUIRED",\n'
            f'  "model_identifier": "{active_model}"\n'
            "}\n"
        )

        contents: List[Any] = [prompt_text]
        for sf in sampled_frames:
            part = types.Part.from_bytes(data=sf.jpeg_bytes, mime_type="image/jpeg")
            contents.append(f"Frame #{sf.frame_index} (timestamp {sf.timestamp_s}s):")
            contents.append(part)

        # Execute generation
        raw_result = self.client.generate_structured_multimodal(
            contents=contents,
            system_instruction=EXAMINATION_SYSTEM_INSTRUCTION,
            model_override=active_model
        )

        # Validate with strict Pydantic model
        validated_output: GeminiExamAnalysisOutput = self.validator.parse_and_validate(
            raw_text=raw_result["text"],
            target_schema=GeminiExamAnalysisOutput
        )

        output_dict = validated_output.model_dump()
        result_payload = {
            "job_id": job_id,
            "module": "examination",
            "cache_key": cache_key,
            "model": raw_result.get("model", active_model),
            "latency_sec": raw_result.get("latency_sec", 0.0),
            "usage_metadata": raw_result.get("usage_metadata", {}),
            "sampled_frame_count": len(sampled_frames),
            "sampled_frames_meta": [
                {"frame": sf.frame_index, "timestamp_s": sf.timestamp_s}
                for sf in sampled_frames
            ],
            "analysis": output_dict,
            "cached": False
        }

        # Cache valid analysis
        _ANALYSIS_CACHE[cache_key] = result_payload
        return result_payload

    def summarize_traffic_context(
        self,
        job_id: int,
        video_path: str,
        traffic_overview: Dict[str, Any],
        force_refresh: bool = False,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a natural-language traffic scene summary using Gemini multimodal reasoning.
        Strictly preserves metric speed estimation from local calibration.
        """
        active_model = model_override or self.config.model
        cache_key = self.sampler.compute_cache_hash(
            job_id=job_id,
            module="traffic",
            model=active_model,
            frame_indices=[0]
        )

        if not force_refresh and cache_key in _ANALYSIS_CACHE:
            cached = _ANALYSIS_CACHE[cache_key].copy()
            cached["cached"] = True
            return cached

        # Sample 3 representative frames across the video
        sampled_frames = self.sampler.sample_keyframes(
            video_path=video_path,
            target_frame_indices=None,
            max_dimension=1024
        )

        from google.genai import types

        prompt_text = (
            f"Analyze these {len(sampled_frames)} selected keyframes from traffic job #{job_id}.\n"
            f"Local Telemetry Metrics (VERIFIED LOCAL DATA):\n"
            f"- Total Tracked Vehicles: {traffic_overview.get('total_vehicles', 0)}\n"
            f"- Congestion Level: {traffic_overview.get('congestion_level', 'NORMAL')}\n"
            f"- Flow Rate: {traffic_overview.get('flow_rate_vpm', 0.0)} vehicles/min\n"
            f"- Calibrated Camera: {traffic_overview.get('is_calibrated', False)}\n"
            "\nProvide a structured JSON traffic summary adhering to the schema:\n"
            "{\n"
            '  "scene_summary": "factual overview of the roadway, lanes, and weather",\n'
            '  "traffic_flow_condition": "FREE_FLOW | MODERATE | CONGESTED | STOP_AND_GO",\n'
            '  "incident_or_congestion_observations": ["observations regarding queues or flow"],\n'
            '  "speed_disclaimer": "Vehicle speed metrics are produced exclusively by VisionRadar\'s local homography calibration and ByteTrack pipeline. Gemini does not calculate or estimate speed.",\n'
            '  "uncertainty_notes": ["camera angle, shadows, or lighting limitations"],\n'
            f'  "model_identifier": "{active_model}"\n'
            "}\n"
        )

        contents: List[Any] = [prompt_text]
        for sf in sampled_frames:
            part = types.Part.from_bytes(data=sf.jpeg_bytes, mime_type="image/jpeg")
            contents.append(f"Frame #{sf.frame_index} (timestamp {sf.timestamp_s}s):")
            contents.append(part)

        raw_result = self.client.generate_structured_multimodal(
            contents=contents,
            system_instruction=TRAFFIC_SYSTEM_INSTRUCTION,
            model_override=active_model
        )

        validated_output: GeminiTrafficSummaryOutput = self.validator.parse_and_validate(
            raw_text=raw_result["text"],
            target_schema=GeminiTrafficSummaryOutput
        )

        result_payload = {
            "job_id": job_id,
            "module": "traffic",
            "cache_key": cache_key,
            "model": raw_result.get("model", active_model),
            "latency_sec": raw_result.get("latency_sec", 0.0),
            "usage_metadata": raw_result.get("usage_metadata", {}),
            "analysis": validated_output.model_dump(),
            "cached": False
        }

        _ANALYSIS_CACHE[cache_key] = result_payload
        return result_payload

# Global analysis service
gemini_analyzer = GeminiAnalysisService()
