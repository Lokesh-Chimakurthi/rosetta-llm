"""OpenAI Responses API codec — parse/render to/from canonical IR.

Reasoning items: `(encrypted_content, id)` round-trip via Anthropic's
`signature` (encoded as `<encrypted>@<id>`). Compaction items use a
`cm1#<encrypted>@<id>` carrier prefix.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

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
    ServerToolUsePart,
    TextPart,
    Tool,
    ToolCallPart,
    ToolResultPart,
    ToolSearchResultPart,
)
from rosetta.ir.response import CanonicalResponse, StopInfo, Usage

_REQUEST_TOP_LEVEL_KEYS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "reasoning",
        "stream",
        "include",
        "metadata",
        "user",
        "store",
        "previous_response_id",
        "parallel_tool_calls",
        "service_tier",
        "prompt_cache_key",
    }
)


def _parse_reasoning_item(item: dict[str, Any]) -> Message:
    encrypted = item.get("encrypted_content", "") or ""
    rid = item.get("id", "") or ""
    summary_text = "".join(
        s.get("text", "")
        for s in item.get("summary", []) or []
        if isinstance(s, dict) and s.get("type") == "summary_text"
    )
    return with_raw(
        Message(
            role="assistant",
            parts=[
                with_raw(
                    ReasoningPart(
                        visibility="redacted" if encrypted and not summary_text else "visible",
                        text=summary_text,
                        encrypted_content=encrypted,
                        reasoning_id=rid,
                    ),
                    item,
                ),
            ],
        ),
        item,
    )


def _parse_tool_definition(t: dict[str, Any]) -> Tool:
    ttype = t.get("type", "function")
    kind: Literal["function", "hosted", "search"]
    if ttype == "function":
        kind = "function"
    elif ttype == "tool_search":
        kind = "search"
    else:
        kind = "hosted"
    return with_raw(
        Tool(
            name=t.get("name", "") or ttype,
            description=t.get("description", ""),
            input_schema=t.get("parameters", {}) or {},
            kind=kind,
            strict=bool(t.get("strict", False)),
            deferred=bool(t.get("defer_loading", False)),
        ),
        t,
    )


def _render_tool_definition(t: Tool) -> dict[str, Any]:
    if t.kind == "search":
        execution = "server"
        raw = t._raw if isinstance(t._raw, dict) else {}
        if raw.get("execution") in ("server", "client"):
            execution = raw["execution"]
        return {"type": "tool_search", "execution": execution}
    raw = t._raw if isinstance(t._raw, dict) else {}
    entry_type = (raw.get("type") or "function") if t.kind == "hosted" else "function"
    entry: dict[str, Any] = {
        "type": entry_type,
        "name": t.name,
        "description": t.description,
        "parameters": t.input_schema or {"type": "object", "properties": {}},
        "strict": t.strict,
    }
    if t.kind == "function" and t.deferred:
        entry["defer_loading"] = True
    return entry


def _parse_tool_search_item(
    item: dict[str, Any], pending_call_id: str | None
) -> tuple[Message, str]:
    """Parse a tool_search_call / tool_search_output item into a message part.

    Server-executed search items carry `call_id: null`; the proxy synthesizes
    the anthropic-side `srvtoolu_` id and pairs the output item to the call
    item through it. Client-executed search has no anthropic equivalent and
    is refused here, at the boundary, before any rendering.
    """
    execution = item.get("execution")
    if execution == "client":
        raise ValueError(
            "client-executed tool search has no anthropic equivalent; refusing to translate"
        )
    if execution is None:
        raise ValueError("tool search item without execution cannot be translated")
    if item.get("type") == "tool_search_call":
        call_id = item.get("call_id") or f"srvtoolu_{uuid.uuid4().hex[:16]}"
        # Anthropic carries the search variant in the block name; Responses
        # has no variant field, so the proxy re-emits `search_variant`.
        name = (
            "tool_search_tool_bm25"
            if item.get("search_variant") == "bm25"
            else "tool_search_tool_regex"
        )
        return (
            with_raw(
                Message(
                    role="assistant",
                    parts=[
                        with_raw(
                            ServerToolUsePart(
                                call_id=call_id,
                                name=name,
                                input=item.get("arguments") or {},
                            ),
                            item,
                        )
                    ],
                ),
                item,
            ),
            call_id,
        )
    result_call_id: str | None = item.get("call_id") or pending_call_id
    if not result_call_id:
        raise ValueError(
            "tool_search_output without a preceding tool_search_call cannot be translated"
        )
    tools = [_parse_tool_definition(t) for t in item.get("tools") or [] if isinstance(t, dict)]
    return (
        with_raw(
            Message(
                role="assistant",
                parts=[with_raw(ToolSearchResultPart(call_id=result_call_id, tools=tools), item)],
            ),
            item,
        ),
        result_call_id,
    )


def _pair_search_result(
    messages: list[Message], last_call: Message | None, msg: Message, call_id: str
) -> None:
    """Append a tool_search_output part to its preceding tool_search_call message."""
    if last_call is not None:
        call_part = last_call.parts[0]
        if isinstance(call_part, ServerToolUsePart) and call_part.call_id == call_id:
            last_call.parts.append(msg.parts[0])
            return
    messages.append(msg)


def _parse_input_item(item: dict[str, Any]) -> Message | None:
    item_type = item.get("type", "")
    role = item.get("role", "user")

    if item_type == "message" or item_type == "":
        content = item.get("content", "")
        parts: list[ContentPart] = []
        if isinstance(content, str):
            parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type", "")
                if ctype in ("input_text", "output_text"):
                    parts.append(with_raw(TextPart(text=c.get("text", "")), c))
                elif ctype == "input_image":
                    url = c.get("image_url", "") or ""
                    if isinstance(url, str) and url.startswith("data:"):
                        header, _, b64 = url[5:].partition(",")
                        media_type = header.split(";")[0] or "image/png"
                        parts.append(
                            with_raw(
                                ImagePart(source_type="base64", media_type=media_type, data=b64), c
                            )
                        )
                    else:
                        parts.append(
                            with_raw(
                                ImagePart(source_type="url", media_type="image/png", data=url), c
                            )
                        )
                elif ctype == "input_file":
                    parts.append(
                        with_raw(
                            DocumentPart(
                                media_type="application/pdf",
                                data=c.get("file_data", "") or c.get("file_id", ""),
                            ),
                            c,
                        )
                    )
                elif ctype == "input_video":
                    parts.append(
                        with_raw(TextPart(text="[Video input not supported by target]"), c)
                    )
                elif ctype == "refusal":
                    parts.append(with_raw(RefusalPart(text=c.get("refusal", "")), c))
        if role not in ("user", "assistant", "system"):
            role = "user"
        return with_raw(Message(role=role, parts=parts), item)

    if item_type == "function_call":
        return with_raw(
            Message(
                role="assistant",
                parts=[
                    with_raw(
                        ToolCallPart(
                            call_id=item.get("call_id", ""),
                            name=item.get("name", ""),
                            arguments_json_text=item.get("arguments", "") or "{}",
                        ),
                        item,
                    ),
                ],
            ),
            item,
        )

    if item_type == "function_call_output":
        output = item.get("output", "")
        if isinstance(output, list):
            content_parts: list[ContentPart] = []
            for c in output:
                if isinstance(c, dict) and c.get("type") in ("input_text", "output_text"):
                    content_parts.append(TextPart(text=c.get("text", "")))
        else:
            content_parts = [TextPart(text=output if isinstance(output, str) else str(output))]
        return with_raw(
            Message(
                role="tool",
                parts=[
                    with_raw(
                        ToolResultPart(
                            call_id=item.get("call_id", ""), content_parts=content_parts
                        ),
                        item,
                    ),
                ],
            ),
            item,
        )

    if item_type == "reasoning":
        return _parse_reasoning_item(item)

    if item_type == "compaction":
        encrypted = item.get("encrypted_content", "") or ""
        rid = item.get("id", "") or ""
        return with_raw(
            Message(
                role="assistant",
                parts=[
                    with_raw(
                        ReasoningPart(
                            visibility="redacted",
                            text="",
                            signature=f"cm1#{encrypted}@{rid}",
                            encrypted_content=encrypted,
                            reasoning_id=rid,
                        ),
                        item,
                    ),
                ],
            ),
            item,
        )

    if item_type == "item_reference":
        # Reference to another item by id — proxy can't resolve; preserve raw.
        return with_raw(Message(role="user", parts=[TextPart(text="")]), item)

    return None


def parse_request(payload: dict[str, Any]) -> CanonicalRequest:
    messages = []
    pending_search_id = ""
    last_search_msg: Message | None = None
    for item in payload.get("input", []) or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        if item_type == "tool_search_call":
            msg, call_id = _parse_tool_search_item(item, None)
            messages.append(msg)
            pending_search_id = call_id
            last_search_msg = msg
            continue
        if item_type == "tool_search_output":
            msg, call_id = _parse_tool_search_item(item, pending_search_id or None)
            pending_search_id = ""
            _pair_search_result(messages, last_search_msg, msg, call_id)
            last_search_msg = None
            continue
        parsed = _parse_input_item(item)
        if parsed is not None:
            messages.append(parsed)
            last_search_msg = None

    tools = [_parse_tool_definition(t) for t in payload.get("tools") or [] if isinstance(t, dict)]

    reasoning = ReasoningConfig()
    reasoning_data = payload.get("reasoning")
    if isinstance(reasoning_data, dict):
        effort = reasoning_data.get("effort")
        if effort in ("low", "medium", "high", "xhigh"):
            reasoning.effort = effort
        summary = reasoning_data.get("summary")
        if summary in ("auto", "detailed", "concise"):
            reasoning.summary = summary

    include = payload.get("include")
    if isinstance(include, list):
        reasoning.include_encrypted = "reasoning.encrypted_content" in include

    instructions = payload.get("instructions")

    raw_extras = {k: v for k, v in payload.items() if k not in _REQUEST_TOP_LEVEL_KEYS}

    return CanonicalRequest(
        model=payload.get("model", ""),
        messages=messages,
        system=instructions if isinstance(instructions, str) and instructions else None,
        tools=tools,
        tool_choice=payload.get("tool_choice", "auto"),
        max_output_tokens=payload.get("max_output_tokens"),
        sampling=SamplingConfig(
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            seed=payload.get("seed"),
        ),
        reasoning=reasoning,
        stream=bool(payload.get("stream", False)),
        metadata=payload.get("metadata") or {},
        raw_extras=raw_extras,
    )


def render_request(ir: CanonicalRequest) -> dict[str, Any]:
    body: dict[str, Any] = {"model": ir.model, "input": []}

    if ir.system is not None:
        instructions = (
            ir.system if isinstance(ir.system, str) else "\n\n".join(p.text for p in ir.system)
        )
        if instructions:
            body["instructions"] = instructions

    for msg in ir.messages:
        if msg.role == "system":
            continue

        message_content: list[dict[str, Any]] = []

        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                if message_content:
                    body["input"].append(_message_item(msg.role, message_content))
                    message_content = []
                body["input"].append(
                    {
                        "type": "function_call",
                        "call_id": part.call_id,
                        "name": part.name,
                        "arguments": part.arguments_json_text or "{}",
                    }
                )
            elif isinstance(part, ToolResultPart):
                if message_content:
                    body["input"].append(_message_item(msg.role, message_content))
                    message_content = []
                output_text = "".join(
                    cp.text for cp in part.content_parts if isinstance(cp, TextPart)
                )
                body["input"].append(
                    {
                        "type": "function_call_output",
                        "call_id": part.call_id,
                        "output": output_text,
                    }
                )
            elif isinstance(part, ServerToolUsePart):
                if message_content:
                    body["input"].append(_message_item(msg.role, message_content))
                    message_content = []
                body["input"].append(_search_call_item(part))
            elif isinstance(part, ToolSearchResultPart):
                if message_content:
                    body["input"].append(_message_item(msg.role, message_content))
                    message_content = []
                body["input"].append(
                    {
                        "type": "tool_search_output",
                        "execution": "server",
                        "call_id": None,
                        "status": "completed",
                        "tools": [_render_tool_definition(t) for t in part.tools],
                    }
                )
            elif isinstance(part, ReasoningPart):
                if message_content:
                    body["input"].append(_message_item(msg.role, message_content))
                    message_content = []
                if part.signature.startswith("cm1#"):
                    body["input"].append(
                        {
                            "type": "compaction",
                            "id": part.reasoning_id,
                            "encrypted_content": part.encrypted_content,
                        }
                    )
                else:
                    item: dict[str, Any] = {"type": "reasoning"}
                    if part.reasoning_id:
                        item["id"] = part.reasoning_id
                    if part.encrypted_content:
                        item["encrypted_content"] = part.encrypted_content
                    item["summary"] = (
                        [{"type": "summary_text", "text": part.text}] if part.text else []
                    )
                    body["input"].append(item)
            elif isinstance(part, TextPart):
                ctype = "input_text" if msg.role == "user" else "output_text"
                message_content.append({"type": ctype, "text": part.text})
            elif isinstance(part, ImagePart):
                url = (
                    part.data
                    if part.source_type == "url"
                    else f"data:{part.media_type};base64,{part.data}"
                )
                message_content.append({"type": "input_image", "image_url": url, "detail": "auto"})
            elif isinstance(part, DocumentPart):
                message_content.append(
                    {
                        "type": "input_file",
                        "file_data": f"data:{part.media_type};base64,{part.data}",
                        "filename": "document.pdf",
                    }
                )
            elif isinstance(part, RefusalPart):
                message_content.append({"type": "refusal", "refusal": part.text})

        if message_content:
            body["input"].append(_message_item(msg.role, message_content))

    if ir.tools:
        body["tools"] = [_render_tool_definition(t) for t in ir.tools]
        if any(t.deferred for t in ir.tools) and not any(t.kind == "search" for t in ir.tools):
            body["tools"].append({"type": "tool_search", "execution": "server"})
        body["tool_choice"] = _render_tool_choice(ir.tool_choice)

    if ir.max_output_tokens:
        body["max_output_tokens"] = ir.max_output_tokens
    if ir.sampling.temperature is not None:
        body["temperature"] = ir.sampling.temperature
    if ir.sampling.top_p is not None:
        body["top_p"] = ir.sampling.top_p

    if ir.reasoning.effort or ir.reasoning.summary:
        reasoning_obj: dict[str, Any] = {}
        if ir.reasoning.effort:
            reasoning_obj["effort"] = ir.reasoning.effort
        if ir.reasoning.summary:
            reasoning_obj["summary"] = ir.reasoning.summary
        body["reasoning"] = reasoning_obj
    if ir.reasoning.include_encrypted:
        body["include"] = ["reasoning.encrypted_content"]
    if ir.stream:
        body["stream"] = True

    for k, v in ir.raw_extras.items():
        body.setdefault(k, v)

    return body


def _message_item(role: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    if role not in ("user", "assistant", "system"):
        role = "user"
    return {"type": "message", "role": role, "content": content}


def _search_call_item(part: ServerToolUsePart) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "tool_search_call",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "arguments": part.input,
    }
    # Responses has no variant channel, so bm25 rides this extra field; a
    # strict upstream may 400 on it, which is loud where dropping the field
    # would silently collapse bm25 to regex.
    if part.name == "tool_search_tool_bm25":
        item["search_variant"] = "bm25"
    return item


def _render_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "function" in tool_choice:
            return tool_choice
        if "name" in tool_choice:
            return {"type": "function", "name": tool_choice["name"]}
        if tool_choice.get("type") in ("auto", "required", "none"):
            return tool_choice["type"]
    if tool_choice == "auto":
        return "auto"
    if tool_choice in ("any", "required"):
        return "required"
    if tool_choice == "none":
        return "none"
    return tool_choice


def parse_response(payload: dict[str, Any]) -> CanonicalResponse:
    output_messages: list[Message] = []
    pending_search_id = ""
    last_search_msg: Message | None = None
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")

        if item_type == "tool_search_call":
            msg, call_id = _parse_tool_search_item(item, None)
            output_messages.append(msg)
            pending_search_id = call_id
            last_search_msg = msg
            continue
        if item_type == "tool_search_output":
            msg, call_id = _parse_tool_search_item(item, pending_search_id or None)
            pending_search_id = ""
            _pair_search_result(output_messages, last_search_msg, msg, call_id)
            last_search_msg = None
            continue

        last_search_msg = None

        if item_type == "message":
            parts: list[ContentPart] = []
            for c in item.get("content", []) or []:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type", "")
                if ctype == "output_text":
                    parts.append(with_raw(TextPart(text=c.get("text", "")), c))
                elif ctype == "refusal":
                    parts.append(with_raw(RefusalPart(text=c.get("refusal", "")), c))
            if parts:
                output_messages.append(with_raw(Message(role="assistant", parts=parts), item))

        elif item_type == "function_call":
            output_messages.append(
                with_raw(
                    Message(
                        role="assistant",
                        parts=[
                            with_raw(
                                ToolCallPart(
                                    call_id=item.get("call_id", ""),
                                    name=item.get("name", ""),
                                    arguments_json_text=item.get("arguments", "") or "{}",
                                ),
                                item,
                            ),
                        ],
                    ),
                    item,
                )
            )

        elif item_type == "reasoning":
            output_messages.append(_parse_reasoning_item(item))

        elif item_type == "compaction":
            encrypted = item.get("encrypted_content", "") or ""
            rid = item.get("id", "") or ""
            output_messages.append(
                with_raw(
                    Message(
                        role="assistant",
                        parts=[
                            with_raw(
                                ReasoningPart(
                                    visibility="redacted",
                                    text="",
                                    signature=f"cm1#{encrypted}@{rid}",
                                    encrypted_content=encrypted,
                                    reasoning_id=rid,
                                ),
                                item,
                            ),
                        ],
                    ),
                    item,
                )
            )

    status = payload.get("status", "completed") or "completed"
    incomplete_reason = (payload.get("incomplete_details") or {}).get("reason", "")

    if status == "completed":
        normalized = (
            "tool_use"
            if any(isinstance(p, ToolCallPart) for m in output_messages for p in m.parts)
            else "end_turn"
        )
    elif status == "incomplete":
        normalized = "max_tokens" if incomplete_reason == "max_output_tokens" else "end_turn"
    elif status == "failed":
        normalized = "refusal"
    else:
        normalized = status

    usage_data = payload.get("usage", {}) or {}
    in_details = usage_data.get("input_tokens_details", {}) or {}
    out_details = usage_data.get("output_tokens_details", {}) or {}

    return with_raw(
        CanonicalResponse(
            id=payload.get("id", ""),
            model=payload.get("model", ""),
            output_messages=output_messages,
            stop=StopInfo(normalized=normalized, provider_raw=status),
            usage=Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                cache_read_input_tokens=int(in_details.get("cached_tokens", 0) or 0),
                cache_creation_input_tokens=int(in_details.get("cache_write_tokens", 0) or 0),
                reasoning_tokens=int(out_details.get("reasoning_tokens", 0) or 0),
            ),
        ),
        payload,
    )


def render_response(ir: CanonicalResponse) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    for msg in ir.output_messages:
        text_parts: list[dict[str, Any]] = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                text_parts.append({"type": "output_text", "text": part.text, "annotations": []})
            elif isinstance(part, ToolCallPart):
                output.append(
                    {
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex[:12]}",
                        "call_id": part.call_id,
                        "name": part.name,
                        "arguments": part.arguments_json_text or "{}",
                        "status": "completed",
                    }
                )
            elif isinstance(part, ServerToolUsePart):
                output.append(_search_call_item(part))
            elif isinstance(part, ToolSearchResultPart):
                output.append(
                    {
                        "type": "tool_search_output",
                        "execution": "server",
                        "call_id": None,
                        "status": "completed",
                        "tools": [_render_tool_definition(t) for t in part.tools],
                    }
                )
            elif isinstance(part, ReasoningPart):
                item: dict[str, Any] = {"type": "reasoning"}
                if part.reasoning_id:
                    item["id"] = part.reasoning_id
                if part.encrypted_content:
                    item["encrypted_content"] = part.encrypted_content
                item["summary"] = (
                    [{"type": "summary_text", "text": part.text}] if part.text else []
                )
                output.append(item)
            elif isinstance(part, RefusalPart):
                text_parts.append({"type": "refusal", "refusal": part.text})

        if text_parts:
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:12]}",
                    "status": "completed",
                    "role": "assistant",
                    "content": text_parts,
                }
            )

    status_map = {
        "end_turn": "completed",
        "max_tokens": "incomplete",
        "refusal": "failed",
        "tool_use": "completed",
    }
    status = status_map.get(ir.stop.normalized, ir.stop.provider_raw or "completed")

    return {
        "id": ir.id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "model": ir.model,
        "status": status,
        "output": output,
        "usage": {
            "input_tokens": ir.usage.input_tokens,
            "output_tokens": ir.usage.output_tokens,
            "total_tokens": ir.usage.input_tokens + ir.usage.output_tokens,
            **(
                {
                    "input_tokens_details": {
                        **(
                            {"cached_tokens": ir.usage.cache_read_input_tokens}
                            if ir.usage.cache_read_input_tokens
                            else {}
                        ),
                        **(
                            {"cache_write_tokens": ir.usage.cache_creation_input_tokens}
                            if ir.usage.cache_creation_input_tokens
                            else {}
                        ),
                    }
                }
                if ir.usage.cache_read_input_tokens or ir.usage.cache_creation_input_tokens
                else {}
            ),
            **(
                {"output_tokens_details": {"reasoning_tokens": ir.usage.reasoning_tokens}}
                if ir.usage.reasoning_tokens
                else {}
            ),
        },
    }
