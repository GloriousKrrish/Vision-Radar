"""
VisionRadar Gemini Multimodal Intelligence Package.
Provides server-side, evidence-linked contextual reasoning across examination and traffic domains.
"""

from visionradar.gemini.config import gemini_config, GeminiConfig
from visionradar.gemini.client import gemini_client, GeminiClientService
from visionradar.gemini.sampler import gemini_sampler, GeminiFrameSampler, SampledFrame
from visionradar.gemini.validator import gemini_validator, GeminiResponseValidator
from visionradar.gemini.analyzer import gemini_analyzer, GeminiAnalysisService
from visionradar.gemini.report import gemini_reporter, GeminiReportService
from visionradar.gemini.schemas import (
    GeminiStatusResponse,
    GeminiExamAnalysisOutput,
    GeminiTrafficSummaryOutput,
    GeminiCandidateEventInterpretation,
    GeminiAnalysisRequestPayload
)

__all__ = [
    "gemini_config",
    "GeminiConfig",
    "gemini_client",
    "GeminiClientService",
    "gemini_sampler",
    "GeminiFrameSampler",
    "SampledFrame",
    "gemini_validator",
    "GeminiResponseValidator",
    "gemini_analyzer",
    "GeminiAnalysisService",
    "gemini_reporter",
    "GeminiReportService",
    "GeminiStatusResponse",
    "GeminiExamAnalysisOutput",
    "GeminiTrafficSummaryOutput",
    "GeminiCandidateEventInterpretation",
    "GeminiAnalysisRequestPayload"
]
