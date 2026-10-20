"""
VisionRadar Gemini Multimodal Intelligence Service: Client Wrapper.
Wraps Google GenAI SDK with bounded retries, model fallback, and comprehensive error handling.
"""

import time
import logging
from typing import Optional, List, Any, Dict
from visionradar.gemini.config import gemini_config

logger = logging.getLogger("visionradar.gemini.client")

class GeminiClientService:
    """Manages secure communication with the Google Gemini API."""

    def __init__(self, config=None):
        self.config = config or gemini_config
        self._client = None
        self._initialized = False

    def _get_client(self):
        """Initializes and returns the official Google GenAI Client."""
        if not self.config.is_configured:
            raise ValueError("GEMINI_API_KEY is not configured in the server environment.")

        if self._client is None:
            try:
                from google import genai
                from google.genai import types

                # Configure client with bounded HTTP timeout
                http_options = types.HttpOptions(
                    timeout=int(self.config.request_timeout_sec * 1000)
                )
                self._client = genai.Client(
                    api_key=self.config.api_key,
                    http_options=http_options
                )
                self._initialized = True
            except Exception as e:
                logger.error("Failed to initialize Google GenAI Client: %s", type(e).__name__)
                raise RuntimeError(f"Gemini client initialization failed: {type(e).__name__}") from e

        return self._client

    def test_connection(self) -> Dict[str, Any]:
        """
        Tests the connection to the Gemini API without consuming substantial quota.
        Returns a dict with success boolean, active model, and error message if any.
        """
        if not self.config.is_configured:
            return {"success": False, "error": "API key missing", "model": self.config.model}

        try:
            client = self._get_client()
            # Fast ping with minimal token cost
            response = client.models.generate_content(
                model=self.config.model,
                contents="ping"
            )
            return {
                "success": True,
                "model": self.config.model,
                "latency_sec": 0.0,
                "status": "ONLINE"
            }
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:150]}"
            logger.warning("Gemini connection test failed: %s", err_msg)
            return {
                "success": False,
                "error": err_msg,
                "model": self.config.model,
                "status": "UNAVAILABLE"
            }

    def generate_structured_multimodal(
        self,
        contents: List[Any],
        system_instruction: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a multimodal structured JSON generation request with bounded retries
        and automatic fallback across available models.
        """
        if not self.config.is_ready:
            raise RuntimeError("Gemini service is disabled or not configured.")

        client = self._get_client()
        from google.genai import types

        # Build candidate models list: preferred model first, then fallbacks
        primary_model = model_override or self.config.model
        candidate_models = [primary_model] + [m for m in self.config.fallback_models if m != primary_model]

        last_error = None
        for model_name in candidate_models:
            for attempt in range(1 + self.config.max_retries):
                try:
                    logger.info("Calling Gemini model %s (attempt %d/%d)", model_name, attempt + 1, 1 + self.config.max_retries)
                    start_t = time.time()

                    gen_config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        system_instruction=system_instruction
                    )

                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=gen_config
                    )

                    elapsed = round(time.time() - start_t, 3)
                    logger.info("Gemini call to %s succeeded in %.2fs", model_name, elapsed)

                    usage_meta = {}
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage_meta = {
                            "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", None),
                            "candidates_token_count": getattr(response.usage_metadata, "candidates_token_count", None),
                            "total_token_count": getattr(response.usage_metadata, "total_token_count", None)
                        }

                    return {
                        "text": response.text,
                        "model": model_name,
                        "latency_sec": elapsed,
                        "usage_metadata": usage_meta
                    }

                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    err_type = type(e).__name__
                    logger.warning("Gemini attempt %d on model %s failed: %s (%s)", attempt + 1, model_name, err_type, err_str[:120])

                    # If model not found (404), break retry loop and switch immediately to next candidate model
                    if "404" in err_str or "NOT_FOUND" in err_str:
                        logger.info("Model %s returned 404, attempting fallback...", model_name)
                        break

                    # Backoff before retry
                    if attempt < self.config.max_retries:
                        time.sleep(1.0 * (2 ** attempt))

        raise RuntimeError(f"All Gemini generation attempts failed across models {candidate_models}. Last error: {type(last_error).__name__} - {str(last_error)[:150]}")

# Global service instance
gemini_client = GeminiClientService()
