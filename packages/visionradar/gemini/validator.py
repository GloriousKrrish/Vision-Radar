"""
VisionRadar Gemini Multimodal Intelligence Service: Validator.
Enforces strict schema validation, prompt-injection defense, and string sanitization.
"""

import json
import re
import html
import logging
from typing import Dict, Any, Type, TypeVar, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("visionradar.gemini.validator")

T = TypeVar("T", bound=BaseModel)

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?rules", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"system\s*:\s*override", re.IGNORECASE),
    re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
]

class GeminiResponseValidator:
    """Validates and sanitizes all inputs and outputs for Gemini interactions."""

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitizes text by escaping HTML entities and removing potential injection markers."""
        if not text:
            return ""
        # Escape HTML entities
        clean = html.escape(text.strip())
        # Filter dangerous control characters
        clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", clean)
        return clean

    @staticmethod
    def check_prompt_injection(text: str) -> bool:
        """Returns True if any prompt-injection signatures are detected."""
        if not text:
            return False
        for pat in INJECTION_PATTERNS:
            if pat.search(text):
                return True
        return False

    @staticmethod
    def parse_and_validate(raw_text: str, target_schema: Type[T]) -> T:
        """
        Parses raw text into JSON and validates against the specified Pydantic schema.
        Handles markdown backticks (```json ... ```) gracefully.
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Empty response received from Gemini.")

        # Clean markdown wrappers if returned
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error from Gemini response: %s", e)
            raise ValueError(f"Malformed JSON response from Gemini: {str(e)[:100]}") from e

        try:
            validated = target_schema.model_validate(data)
            return validated
        except ValidationError as e:
            logger.error("Schema validation failed on Gemini response: %s", e)
            raise ValueError(f"Schema validation error on Gemini response: {str(e)[:150]}") from e

# Global validator
gemini_validator = GeminiResponseValidator()
