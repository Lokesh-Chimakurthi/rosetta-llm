"""Local tiktoken-based token counting for /v1/messages/count_tokens.

This is an approximation. Anthropic uses its own tokenizer; tiktoken's
o200k_base is an OpenAI-family encoding. Real counts can differ by
~5–15%. Use this only when an upstream count_tokens isn't available.
"""

from __future__ import annotations

from typing import Any

import orjson
import tiktoken

_ENCODING = tiktoken.get_encoding("o200k_base")
_TOOL_OVERHEAD = 16  # rough tokens for the tool envelope per call


def _encode_len(text: str) -> int:
    if not text:
        return 0
    return len(_ENCODING.encode(text, disallowed_special=()))


def _walk_content(content: Any) -> int:
    if isinstance(content, str):
        return _encode_len(content)
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            total += _encode_len(block.get("text", ""))
        elif btype == "tool_use":
            total += _encode_len(block.get("name", ""))
            inp = block.get("input")
            if isinstance(inp, str):
                total += _encode_len(inp)
            elif inp is not None:
                total += _encode_len(orjson.dumps(inp).decode())
            total += _TOOL_OVERHEAD
        elif btype == "tool_result":
            total += _walk_content(block.get("content", ""))
        elif btype == "image":
            total += 85  # rough Anthropic image baseline
        elif btype == "document":
            total += 256  # rough PDF baseline
        elif btype == "thinking":
            total += _encode_len(block.get("thinking", ""))
    return total


def count_tokens_anthropic(body: dict[str, Any]) -> int:
    """Approximate token count for an Anthropic Messages request body."""
    total = 0

    system = body.get("system")
    if isinstance(system, str):
        total += _encode_len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                total += _encode_len(block.get("text", ""))

    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            total += _walk_content(msg.get("content", ""))

    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        total += _encode_len(tool.get("name", ""))
        total += _encode_len(tool.get("description", ""))
        schema = tool.get("input_schema")
        if isinstance(schema, dict):
            total += _encode_len(orjson.dumps(schema).decode())

    return total
