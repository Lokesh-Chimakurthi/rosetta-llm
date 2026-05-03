"""OpenAI Chat Completions API codec — parse/render to/from canonical IR."""

from __future__ import annotations

import time
import uuid
from typing import Any

from rosetta.ir.helpers import with_raw
from rosetta.ir.request import (
    CanonicalRequest,
    ContentPart,
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
from rosetta.stop_reasons import OPENAI_CHAT_STOP_IN as _STOP_IN
from rosetta.stop_reasons import OPENAI_CHAT_STOP_OUT as _STOP_OUT

_REQUEST_TOP_LEVEL_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop",
        "reasoning_effort",
        "stream",
        "user",
    }
)


def _parse_content_part(part: dict[str, Any]) -> ContentPart:
    part_type = part.get("type", "text")
    if part_type == "text":
        return with_raw(TextPart(text=part.get("text", "")), part)
    if part_type == "image_url":
        url_data = part.get("image_url", {}) or {}
        url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
        if url.startswith("data:"):
            header, _, b64 = url[5:].partition(",")
            media_type = header.split(";")[0] or "image/png"
            return with_raw(ImagePart(source_type="base64", media_type=media_type, data=b64), part)
        return with_raw(ImagePart(source_type="url", media_type="image/png", data=url), part)
    if part_type == "input_audio":
        return with_raw(TextPart(text="[Audio input not supported by target]"), part)
    if part_type == "file":
        filename = part.get("filename", "file")
        return with_raw(TextPart(text=f"[File attached: {filename}]"), part)
    return with_raw(TextPart(text=part.get("text") or str(part)), part)


def parse_request(payload: dict[str, Any]) -> CanonicalRequest:
    messages: list[Message] = []
    system_texts: list[str] = []

    for msg in payload.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system" or role == "developer":
            if isinstance(content, str):
                system_texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        system_texts.append(item.get("text", ""))
            continue

        parts: list[ContentPart] = []

        if role in ("tool", "function"):
            text = content if isinstance(content, str) else str(content)
            call_id = str(msg.get("tool_call_id") or msg.get("function_call_id") or "")
            parts.append(
                with_raw(
                    ToolResultPart(
                        call_id=call_id,
                        content_parts=[TextPart(text=text)],
                    ),
                    msg,
                )
            )
            messages.append(with_raw(Message(role="tool", parts=parts), msg))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                parts.append(TextPart(text=content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        parts.append(_parse_content_part(item))

            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {}) or {}
                parts.append(
                    with_raw(
                        ToolCallPart(
                            call_id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments_json_text=func.get("arguments", "") or "{}",
                        ),
                        tc,
                    )
                )

            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning:
                parts.append(ReasoningPart(visibility="visible", text=str(reasoning)))

            refusal = msg.get("refusal")
            if refusal:
                parts.append(RefusalPart(text=str(refusal)))

            messages.append(with_raw(Message(role="assistant", parts=parts), msg))
            continue

        # user (default)
        if isinstance(content, str):
            parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(_parse_content_part(item))
        if parts:
            messages.append(with_raw(Message(role="user", parts=parts), msg))

    tools: list[Tool] = []
    for t in payload.get("tools") or []:
        if not isinstance(t, dict):
            continue
        func = t.get("function", {}) or {}
        params = func.get("parameters", {}) or {}
        if (
            isinstance(params, dict)
            and params.get("type") == "object"
            and "properties" not in params
        ):
            params = {**params, "properties": {}}
        tools.append(
            with_raw(
                Tool(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    input_schema=params,
                    strict=bool(t.get("strict", func.get("strict", False))),
                ),
                t,
            )
        )

    reasoning = ReasoningConfig()
    effort = payload.get("reasoning_effort")
    if effort in ("low", "medium", "high", "xhigh"):
        reasoning.effort = effort

    stop_raw = payload.get("stop")
    if isinstance(stop_raw, str):
        stop_sequences = [stop_raw]
    elif isinstance(stop_raw, list):
        stop_sequences = [s for s in stop_raw if isinstance(s, str)]
    else:
        stop_sequences = []

    raw_extras = {k: v for k, v in payload.items() if k not in _REQUEST_TOP_LEVEL_KEYS}

    return CanonicalRequest(
        model=payload.get("model", ""),
        messages=messages,
        system="\n\n".join(system_texts) if system_texts else None,
        tools=tools,
        tool_choice=payload.get("tool_choice", "auto"),
        max_output_tokens=payload.get("max_tokens") or payload.get("max_completion_tokens"),
        sampling=SamplingConfig(
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            seed=payload.get("seed"),
        ),
        stop_sequences=stop_sequences,
        reasoning=reasoning,
        stream=bool(payload.get("stream", False)),
        metadata={"user": payload["user"]} if isinstance(payload.get("user"), str) else {},
        raw_extras=raw_extras,
    )


def render_request(ir: CanonicalRequest) -> dict[str, Any]:
    body: dict[str, Any] = {"model": ir.model, "messages": []}

    if ir.system is not None:
        system_text = (
            ir.system if isinstance(ir.system, str) else "\n\n".join(p.text for p in ir.system)
        )
        if system_text:
            body["messages"].append({"role": "system", "content": system_text})

    for msg in ir.messages:
        if msg.role == "system":
            continue

        # Handle tool results: emit each as its own role=tool message first.
        for part in msg.parts:
            if isinstance(part, ToolResultPart):
                body["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": part.call_id,
                        "content": _tool_result_to_text(part),
                    }
                )

        if msg.role == "tool":
            continue

        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_text = ""
        refusal_text = ""

        for part in msg.parts:
            if isinstance(part, TextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                url = (
                    part.data
                    if part.source_type == "url"
                    else f"data:{part.media_type};base64,{part.data}"
                )
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            elif isinstance(part, ToolCallPart):
                tool_calls.append(
                    {
                        "id": part.call_id,
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": part.arguments_json_text or "{}",
                        },
                    }
                )
            elif isinstance(part, ReasoningPart):
                if part.text:
                    reasoning_text += part.text
            elif isinstance(part, RefusalPart):
                refusal_text = part.text
            # ToolResultPart already handled above

        if msg.role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if content_parts:
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    assistant_msg["content"] = content_parts[0]["text"]
                else:
                    assistant_msg["content"] = content_parts
            elif not tool_calls:
                assistant_msg["content"] = ""
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            if reasoning_text:
                assistant_msg["reasoning_content"] = reasoning_text
            if refusal_text:
                assistant_msg["refusal"] = refusal_text
            body["messages"].append(assistant_msg)
        elif msg.role == "user":
            if not content_parts:
                continue
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                body["messages"].append({"role": "user", "content": content_parts[0]["text"]})
            else:
                body["messages"].append({"role": "user", "content": content_parts})

    if ir.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema or {"type": "object", "properties": {}},
                    **({"strict": True} if t.strict else {}),
                },
            }
            for t in ir.tools
        ]
        body["tool_choice"] = _render_tool_choice(ir.tool_choice)

    if ir.max_output_tokens:
        body["max_tokens"] = ir.max_output_tokens
    if ir.sampling.temperature is not None:
        body["temperature"] = ir.sampling.temperature
    if ir.sampling.top_p is not None:
        body["top_p"] = ir.sampling.top_p
    if ir.sampling.seed is not None:
        body["seed"] = ir.sampling.seed
    if ir.stop_sequences:
        body["stop"] = ir.stop_sequences if len(ir.stop_sequences) > 1 else ir.stop_sequences[0]
    if ir.reasoning.effort:
        body["reasoning_effort"] = ir.reasoning.effort
    if ir.stream:
        body["stream"] = True
    if isinstance(ir.metadata.get("user"), str):
        body["user"] = ir.metadata["user"]

    for k, v in ir.raw_extras.items():
        body.setdefault(k, v)

    return body


def _tool_result_to_text(part: ToolResultPart) -> str:
    chunks: list[str] = []
    for cp in part.content_parts:
        if isinstance(cp, TextPart):
            chunks.append(cp.text)
    return "\n\n".join(chunks)


def _render_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "function" in tool_choice:
            return tool_choice
        if "name" in tool_choice:
            return {"type": "function", "function": {"name": tool_choice["name"]}}
        if tool_choice.get("type") in ("auto", "any", "none", "required"):
            return _render_tool_choice(tool_choice["type"])
        return tool_choice
    if tool_choice == "auto":
        return "auto"
    if tool_choice in ("any", "required"):
        return "required"
    if tool_choice == "none":
        return "none"
    return tool_choice


def parse_response(payload: dict[str, Any]) -> CanonicalResponse:
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message", {}) or {}
    parts: list[ContentPart] = []

    content = msg.get("content")
    if isinstance(content, str) and content:
        parts.append(TextPart(text=content))
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                parts.append(_parse_content_part(item))

    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        parts.append(ReasoningPart(visibility="visible", text=str(reasoning)))

    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {}) or {}
        parts.append(
            with_raw(
                ToolCallPart(
                    call_id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments_json_text=func.get("arguments", "") or "{}",
                ),
                tc,
            )
        )

    refusal = msg.get("refusal")
    if refusal:
        parts.append(RefusalPart(text=str(refusal)))

    output_messages = [Message(role="assistant", parts=parts)] if parts else []

    finish = choice.get("finish_reason", "") or ""
    usage_data = payload.get("usage", {}) or {}
    prompt_details = usage_data.get("prompt_tokens_details", {}) or {}
    completion_details = usage_data.get("completion_tokens_details", {}) or {}

    return with_raw(
        CanonicalResponse(
            id=payload.get("id", ""),
            model=payload.get("model", ""),
            output_messages=output_messages,
            stop=StopInfo(normalized=_STOP_IN.get(finish, finish), provider_raw=finish),
            usage=Usage(
                input_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage_data.get("completion_tokens", 0) or 0),
                cache_read_input_tokens=int(prompt_details.get("cached_tokens", 0) or 0),
                cache_creation_input_tokens=int(prompt_details.get("cache_write_tokens", 0) or 0),
                reasoning_tokens=int(completion_details.get("reasoning_tokens", 0) or 0),
            ),
        ),
        payload,
    )


def render_response(ir: CanonicalResponse) -> dict[str, Any]:
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_text = ""
    refusal_text = ""

    for msg in ir.output_messages:
        for part in msg.parts:
            if isinstance(part, TextPart):
                text_chunks.append(part.text)
            elif isinstance(part, ToolCallPart):
                tool_calls.append(
                    {
                        "id": part.call_id,
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": part.arguments_json_text or "{}",
                        },
                    }
                )
            elif isinstance(part, ReasoningPart):
                reasoning_text += part.text
            elif isinstance(part, RefusalPart):
                refusal_text = part.text

    message: dict[str, Any] = {"role": "assistant"}
    message["content"] = "".join(text_chunks) if text_chunks else None
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    if refusal_text:
        message["refusal"] = refusal_text

    result: dict[str, Any] = {
        "id": ir.id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ir.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _STOP_OUT.get(ir.stop.normalized, ir.stop.provider_raw or "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": ir.usage.input_tokens,
            "completion_tokens": ir.usage.output_tokens,
            "total_tokens": ir.usage.input_tokens + ir.usage.output_tokens,
            **(
                {
                    "prompt_tokens_details": {
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
                {"completion_tokens_details": {"reasoning_tokens": ir.usage.reasoning_tokens}}
                if ir.usage.reasoning_tokens
                else {}
            ),
        },
    }
    # Carry through provider-specific response fields when present in _raw.
    raw = ir._raw if hasattr(ir, "_raw") and isinstance(ir._raw, dict) else {}
    for field in ("system_fingerprint", "service_tier", "store"):
        if field in raw:
            result[field] = raw[field]
    return result
