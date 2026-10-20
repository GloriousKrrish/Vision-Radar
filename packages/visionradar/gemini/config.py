"""
VisionRadar Gemini Multimodal Intelligence Service: Configuration.
Loads server-side configuration safely without exposing keys or crashing on missing values.
"""

import os
from typing import Dict, Any, List, Optional
import dotenv

# Load environment variables from .env if present
dotenv.load_dotenv()

DEFAULT_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]

class GeminiConfig:
    """Manages Gemini configuration strictly on the server side."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enabled: Optional[bool] = None,
        max_selected_frames: Optional[int] = None,
        request_timeout_sec: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        if api_key is not None:
            self.api_key: Optional[str] = api_key.strip() or None
        else:
            self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY", "").strip() or None

        if model is not None:
            self.model: str = model.strip()
        else:
            self.model: str = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview").strip()

        self.fallback_models: List[str] = [m for m in DEFAULT_MODELS if m != self.model]

        if enabled is not None:
            self.enabled: bool = bool(enabled)
        else:
            self.enabled: bool = os.environ.get("GEMINI_ENABLED", "true").lower() in ("1", "true", "yes")
        
        if max_selected_frames is not None:
            self.max_selected_frames = max_selected_frames
        else:
            try:
                self.max_selected_frames: int = max(1, min(12, int(os.environ.get("GEMINI_MAX_SELECTED_FRAMES", "6"))))
            except (ValueError, TypeError):
                self.max_selected_frames = 6

        if request_timeout_sec is not None:
            self.request_timeout_sec = request_timeout_sec
        else:
            try:
                self.request_timeout_sec: float = max(5.0, min(120.0, float(os.environ.get("GEMINI_REQUEST_TIMEOUT_SEC", "25.0"))))
            except (ValueError, TypeError):
                self.request_timeout_sec = 25.0

        if max_retries is not None:
            self.max_retries = max_retries
        else:
            try:
                self.max_retries: int = max(0, min(5, int(os.environ.get("GEMINI_MAX_RETRIES", "2"))))
            except (ValueError, TypeError):
                self.max_retries = 2

    @property
    def is_configured(self) -> bool:
        """True if an API key is present in the server environment."""
        return bool(self.api_key and len(self.api_key) > 8)

    @property
    def is_ready(self) -> bool:
        """True if Gemini is enabled and configured."""
        return self.enabled and self.is_configured

    @property
    def is_available(self) -> bool:
        """Alias for is_ready."""
        return self.is_ready

    def get_safe_status(self) -> Dict[str, Any]:
        """
        Returns runtime status without ever revealing API keys or sensitive strings.
        Safe for public UI/API consumption.
        """
        if not self.enabled:
            status = "DISABLED"
            detail = "Gemini multimodal service is disabled by server configuration."
        elif not self.is_configured:
            status = "KEY_MISSING"
            detail = "Server environment variable GEMINI_API_KEY is not set."
        else:
            status = "READY"
            detail = f"Gemini multimodal service active using model '{self.model}'."

        return {
            "enabled": self.enabled,
            "configured": self.is_configured,
            "status": status,
            "model": self.model,
            "max_selected_frames": self.max_selected_frames,
            "timeout_sec": self.request_timeout_sec,
            "detail": detail
        }

# Global singleton configuration
gemini_config = GeminiConfig()

def get_gemini_config() -> GeminiConfig:
    """Returns the active global GeminiConfig singleton."""
    return gemini_config
