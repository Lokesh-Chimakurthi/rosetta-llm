"""Endpoint-format-aware error envelope formatting."""

from __future__ import annotations

from typing import Any

import orjson


def format_error(
    inbound_format: str, status_code: int, error_type: str, message: str
) -> dict[str, Any]:
    """Produce an error response body matching the inbound endpoint format."""
    if inbound_format == "anthropic":
        return {
            "type": "error",
            "error": {"type": error_type, "message": message},
        }
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": None,
            "param": None,
        }
    }


def format_stream_error(inbound_format: str, error_type: str, message: str) -> bytes:
    """Produce an inline stream error event matching the inbound format."""
    payload = format_error(inbound_format, 0, error_type, message)
    body = orjson.dumps(payload)
    if inbound_format == "anthropic":
        return b"event: error\ndata: " + body + b"\n\n"
    return b"data: " + body + b"\n\n"
