"""
VisionRadar WebSocket Protocol Specification.

Defines standard message envelopes, message types, and versioning for real-time perception streaming.
Section E Compliant:
Envelope format:
{
  "type": "frame_result" | "perception_metrics" | "stream_status" | "error",
  "seq": 1042,
  "payload": { ... }
}
"""

from typing import Any, Dict, Literal, Optional

PROTOCOL_VERSION = 1

MessageType = Literal["frame_result", "perception_metrics", "stream_status", "error"]


def create_envelope(
    msg_type: MessageType,
    payload: Dict[str, Any],
    seq: int = 0
) -> Dict[str, Any]:
    """Wraps a payload dictionary into the standard Section E WebSocket envelope."""
    return {
        "type": msg_type,
        "seq": seq,
        "payload": payload
    }


def create_stream_status_message(
    session_id: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    seq: int = 0
) -> Dict[str, Any]:
    """
    Creates a stream_status envelope.
    Includes protocol_version = 1 per section E.
    """
    payload = {
        "session_id": session_id,
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "details": details or {}
    }
    return create_envelope("stream_status", payload, seq=seq)


def create_frame_result_message(
    frame_result_dict: Dict[str, Any],
    seq: int = 0
) -> Dict[str, Any]:
    """Creates a frame_result envelope containing a FrameResult to_dict() payload."""
    return create_envelope("frame_result", frame_result_dict, seq=seq)


def create_perception_metrics_message(
    metrics_dict: Dict[str, Any],
    seq: int = 0
) -> Dict[str, Any]:
    """Creates a perception_metrics envelope containing a PerceptionMetrics to_dict() payload."""
    return create_envelope("perception_metrics", metrics_dict, seq=seq)


def create_error_message(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    seq: int = 0
) -> Dict[str, Any]:
    """Creates a structured error envelope."""
    payload = {
        "code": error_code,
        "message": message,
        "details": details or {}
    }
    return create_envelope("error", payload, seq=seq)
