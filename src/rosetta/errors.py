"""Endpoint-format-aware error envelope formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import orjson

from rosetta.config import Format


def inbound_format_for(path: str, headers: Mapping[str, str]) -> Format:
    """The API format whose error envelope a request gets.

    Args:
        headers: Request headers with lower-case names.
    """
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path == "/v1/chat/completions":
        return "openai_chat"
    if path.startswith("/v1/responses"):
        return "openai_responses"
    if path.startswith("/v1/models") and "anthropic-version" in headers:
        return "anthropic"
    return "openai_chat"


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
