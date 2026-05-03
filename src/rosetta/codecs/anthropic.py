"""Anthropic Messages API codec — parse/render to/from canonical IR.

Lossless reasoning round-trip: Responses' `(encrypted_content, id)` pair is
encoded into Anthropic's `signature` field as `<encrypted>@<id>`. Decoded
on parse by splitting on the *last* `@`.
"""

from __future__ import annotations

from typing import Any

import orjson

from rosetta.ir.helpers import with_raw
from rosetta.ir.request import (
    CanonicalRequest,
    ContentPart,
    DocumentPart,
    ImagePart,
    Message,
    ReasoningConfig,
    ReasoningPart,
    RefusalPart,
    SamplingConfig,
    TextPart,
    Tool,
    ToolCallPart,
    ToolResultPart,
)
from rosetta.ir.response import CanonicalResponse, StopInfo, Usage
from rosetta.stop_reasons import ANTHROPIC_STOP as _STOP_REASON_MAP

_REQUEST_TOP_LEVEL_KEYS = frozenset(
    {
        "model",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "thinking",
        "output_config",
        "stream",
        "metadata",
    }
)


def _decode_signature(signature: str) -> tuple[str, str]:
    """Split signature on the LAST '@' into (encrypted_content, reasoning_id)."""
    if not signature:
        return "", ""
    idx = signature.rfind("@")
    if idx <= 0 or idx == len(signature) - 1:
        return signature, ""
    return signature[:idx], signature[idx + 1 :]


def _encode_signature(encrypted_content: str, reasoning_id: str) -> str:
    if not reasoning_id:
        return encrypted_content
    return f"{encrypted_content}@{reasoning_id}"


def _arguments_to_text(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if input_value is None:
        return "{}"
    return orjson.dumps(input_value).decode()


def _arguments_from_text(text: str) -> Any:
    if not text:
        return {}
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        return {"raw_arguments": text}


def _parse_content_block(block: dict[str, Any]) -> ContentPart:
    block_type = block.get("type", "")
    if block_type == "text":
        return with_raw(TextPart(text=block.get("text", "")), block)
    if block_type == "image":
        source = block.get("source", {})
        return with_raw(
            ImagePart(
                source_type="url" if source.get("type") == "url" else "base64",
                media_type=source.get("media_type", "image/png"),
                data=source.get("url") or source.get("data", ""),
            ),
            block,
        )
    if block_type == "document":
        source = block.get("source", {})
        return with_raw(
            DocumentPart(
                media_type=source.get("media_type", "application/pdf"),
                data=source.get("data", ""),
            ),
            block,
        )
    if block_type == "tool_use":
        return with_raw(
            ToolCallPart(
                call_id=block.get("id", ""),
                name=block.get("name", ""),
                arguments_json_text=_arguments_to_text(block.get("input")),
            ),
            block,
        )
    if block_type == "tool_result":
        content = block.get("content", "")
        parts: list[ContentPart] = []
        if isinstance(content, str):
            parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(_parse_content_block(item))
        return with_raw(
            ToolResultPart(
                call_id=block.get("tool_use_id", ""),
                content_parts=parts,
                is_error=bool(block.get("is_error", False)),
            ),
            block,
        )
    if block_type == "thinking":
        signature = block.get("signature", "")
        encrypted, rid = _decode_signature(signature)
        return with_raw(
            ReasoningPart(
                visibility="visible",
                text=block.get("thinking", ""),
                signature=signature,
                encrypted_content=encrypted,
                reasoning_id=rid,
            ),
            block,
        )
    if block_type == "redacted_thinking":
        return with_raw(
            ReasoningPart(
                visibility="redacted",
                text="",
                signature=block.get("data", ""),
            ),
            block,
        )
    if block_type == "refusal":
        return with_raw(RefusalPart(text=block.get("refusal", "") or block.get("text", "")), block)
    return with_raw(TextPart(text=str(block)), block)


def _render_content_part(part: ContentPart) -> dict[str, Any]:
    cc = part._raw.get("cache_control") if hasattr(part, "_raw") else None
    if isinstance(part, TextPart):
        block: dict[str, Any] = {"type": "text", "text": part.text}
        if cc:
            block["cache_control"] = cc
        return block
    if isinstance(part, ImagePart):
        if part.source_type == "url":
            block = {"type": "image", "source": {"type": "url", "url": part.data}}
        else:
            block = {
                "type": "image",
                "source": {"type": "base64", "media_type": part.media_type, "data": part.data},
            }
        if cc:
            block["cache_control"] = cc
        return block
    if isinstance(part, DocumentPart):
        block = {
            "type": "document",
            "source": {"type": "base64", "media_type": part.media_type, "data": part.data},
        }
        if cc:
            block["cache_control"] = cc
        return block
    if isinstance(part, ToolCallPart):
        block = {
            "type": "tool_use",
            "id": part.call_id,
            "name": part.name,
            "input": _arguments_from_text(part.arguments_json_text),
        }
        if cc:
            block["cache_control"] = cc
        return block
    if isinstance(part, ToolResultPart):
        if not part.content_parts:
            content: str | list[dict[str, Any]] = ""
        elif len(part.content_parts) == 1 and isinstance(part.content_parts[0], TextPart):
            content = part.content_parts[0].text
        else:
            content = [
                _render_content_part(cp)
                for cp in part.content_parts
                if isinstance(cp, TextPart | ImagePart | DocumentPart)
            ]
        block = {"type": "tool_result", "tool_use_id": part.call_id, "content": content}
        if cc:
            block["cache_control"] = cc
        if part.is_error:
            block["is_error"] = True
        return block
    if isinstance(part, ReasoningPart):
        if part.visibility == "redacted":
            return {"type": "redacted_thinking", "data": part.signature}
        signature = part.signature or _encode_signature(part.encrypted_content, part.reasoning_id)
        block = {"type": "thinking", "thinking": part.text, "signature": signature}
        if cc:
            block["cache_control"] = cc
        return block
    if isinstance(part, RefusalPart):
        return {"type": "refusal", "refusal": part.text}
    return {"type": "text", "text": str(part)}


def parse_request(payload: dict[str, Any]) -> CanonicalRequest:
    messages: list[Message] = []
    for msg in payload.get("messages", []):
        role = msg.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        parts: list[ContentPart] = []
        if isinstance(content, str):
            parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    parts.append(_parse_content_block(block))
        messages.append(with_raw(Message(role=role, parts=parts), msg))

    system_raw = payload.get("system")
    system: str | list[TextPart] | None
    if isinstance(system_raw, str):
        system = system_raw
    elif isinstance(system_raw, list):
        system = [
            TextPart(text=b.get("text", "")) if isinstance(b, dict) else TextPart(text=str(b))
            for b in system_raw
        ]
    else:
        system = None

    tools = [
        with_raw(
            Tool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("input_schema", {}),
            ),
            t,
        )
        for t in payload.get("tools", [])
        if isinstance(t, dict)
    ]

    reasoning = ReasoningConfig()
    thinking = payload.get("thinking")
    if isinstance(thinking, dict):
        ttype = thinking.get("type")
        if ttype in ("enabled", "adaptive", "disabled"):
            reasoning.thinking_type = ttype
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int):
            reasoning.budget_tokens = budget
    output_config = payload.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if effort in ("low", "medium", "high", "xhigh", "max"):
            reasoning.effort = effort

    raw_extras = {k: v for k, v in payload.items() if k not in _REQUEST_TOP_LEVEL_KEYS}

    return CanonicalRequest(
        model=payload.get("model", ""),
        messages=messages,
        system=system,
        tools=tools,
        tool_choice=payload.get("tool_choice", "auto"),
        max_output_tokens=payload.get("max_tokens"),
        sampling=SamplingConfig(
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            top_k=payload.get("top_k"),
        ),
        stop_sequences=list(payload.get("stop_sequences") or []),
        reasoning=reasoning,
        stream=bool(payload.get("stream", False)),
        metadata=payload.get("metadata") or {},
        raw_extras=raw_extras,
    )


def _enforce_tool_result_ordering(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic requires tool_result blocks to precede text in user messages."""
    tool_results = [p for p in parts if p.get("type") == "tool_result"]
    others = [p for p in parts if p.get("type") != "tool_result"]
    return tool_results + others


def render_request(ir: CanonicalRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": ir.model,
        "messages": [],
        "max_tokens": ir.max_output_tokens or 4096,
    }

    if ir.system is not None:
        if isinstance(ir.system, list):
            body["system"] = [{"type": "text", "text": p.text} for p in ir.system]
        else:
            body["system"] = ir.system

    pending_tool_results: list[dict[str, Any]] = []

    for msg in ir.messages:
        if msg.role == "system":
            continue

        if msg.role == "tool":
            for part in msg.parts:
                if isinstance(part, ToolResultPart):
                    pending_tool_results.append(_render_content_part(part))
            continue

        rendered = [_render_content_part(p) for p in msg.parts]

        if msg.role == "user":
            content = pending_tool_results + rendered
            pending_tool_results = []
            content = _enforce_tool_result_ordering(content)
            body["messages"].append({"role": "user", "content": content})
        elif msg.role == "assistant":
            if pending_tool_results:
                body["messages"].append(
                    {
                        "role": "user",
                        "content": _enforce_tool_result_ordering(pending_tool_results),
                    }
                )
                pending_tool_results = []
            body["messages"].append({"role": "assistant", "content": rendered})

    if pending_tool_results:
        body["messages"].append(
            {"role": "user", "content": _enforce_tool_result_ordering(pending_tool_results)}
        )

    if ir.tools:
        body["tools"] = []
        for t in ir.tools:
            tool_body: dict[str, Any] = {}
            if hasattr(t, "_raw") and isinstance(t._raw, dict):
                tool_body.update(t._raw)
            tool_body["name"] = t.name
            tool_body["description"] = t.description
            tool_body["input_schema"] = t.input_schema or {"type": "object", "properties": {}}
            body["tools"].append(tool_body)
        body["tool_choice"] = _render_tool_choice(ir.tool_choice)

    if ir.sampling.temperature is not None:
        body["temperature"] = ir.sampling.temperature
    if ir.sampling.top_p is not None:
        body["top_p"] = ir.sampling.top_p
    if ir.sampling.top_k is not None:
        body["top_k"] = ir.sampling.top_k

    if ir.stop_sequences:
        body["stop_sequences"] = ir.stop_sequences

    # Reasoning / thinking
    if ir.reasoning.thinking_type == "disabled":
        body["thinking"] = {"type": "disabled"}
    elif ir.reasoning.budget_tokens or ir.reasoning.thinking_type:
        thinking_obj: dict[str, Any] = {"type": ir.reasoning.thinking_type or "enabled"}
        if ir.reasoning.budget_tokens:
            thinking_obj["budget_tokens"] = ir.reasoning.budget_tokens
        body["thinking"] = thinking_obj
    if ir.reasoning.effort:
        body.setdefault("output_config", {})["effort"] = ir.reasoning.effort

    if ir.metadata:
        body["metadata"] = ir.metadata
    if ir.stream:
        body["stream"] = True

    for k, v in ir.raw_extras.items():
        body.setdefault(k, v)

    return body


def _render_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict):
        raw = dict(tool_choice)
        if raw.get("type") == "function":
            name = (raw.get("function") or {}).get("name")
            raw = {"type": "tool", "name": name} if name else {"type": "auto"}
        if "disable_parallel_tool_use" in raw:
            raw.setdefault("type", "auto")
        return raw
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice in ("any", "required"):
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    return tool_choice


def parse_response(payload: dict[str, Any]) -> CanonicalResponse:
    parts: list[ContentPart] = [
        _parse_content_block(block)
        for block in payload.get("content", [])
        if isinstance(block, dict)
    ]
    output_messages = [Message(role="assistant", parts=parts)] if parts else []

    stop_reason = payload.get("stop_reason") or ""
    usage_data = payload.get("usage", {}) or {}

    return with_raw(
        CanonicalResponse(
            id=payload.get("id", ""),
            model=payload.get("model", ""),
            output_messages=output_messages,
            stop=StopInfo(
                normalized=_STOP_REASON_MAP.get(stop_reason, stop_reason),
                provider_raw=stop_reason,
                stop_sequence=payload.get("stop_sequence"),
            ),
            usage=Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                cache_read_input_tokens=int(usage_data.get("cache_read_input_tokens", 0) or 0),
                cache_creation_input_tokens=int(
                    usage_data.get("cache_creation_input_tokens", 0) or 0
                ),
            ),
        ),
        payload,
    )


def render_response(ir: CanonicalResponse) -> dict[str, Any]:
    content = [_render_content_part(part) for msg in ir.output_messages for part in msg.parts]

    return {
        "id": ir.id,
        "type": "message",
        "role": "assistant",
        "model": ir.model,
        "content": content,
        "stop_reason": _STOP_REASON_MAP.get(ir.stop.normalized, ir.stop.provider_raw or None),
        "stop_sequence": ir.stop.stop_sequence,
        "usage": {
            "input_tokens": ir.usage.input_tokens,
            "output_tokens": ir.usage.output_tokens,
            "cache_read_input_tokens": ir.usage.cache_read_input_tokens,
            "cache_creation_input_tokens": ir.usage.cache_creation_input_tokens,
        },
    }
