"""OpenAI Responses streaming codec — semantic event parse/render via IR events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import orjson

from rosetta.ir.events import (
    CanonicalStreamEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PartDeltaEvent,
    PartStartEvent,
    PartStopEvent,
)
from rosetta.ir.helpers import with_raw
from rosetta.ir.response import StopInfo, Usage

MAX_WHITESPACE_RUN = 20


async def parse(chunks: AsyncIterator[bytes]) -> AsyncIterator[CanonicalStreamEvent]:
    """Parse OpenAI Responses semantic event stream into canonical IR events."""
    buffer = b""
    current_item_index = -1
    ws_counters: dict[int, int] = {}

    async for chunk in chunks:
        buffer += chunk
        while b"\n\n" in buffer:
            line_block, buffer = buffer.split(b"\n\n", 1)
            text = line_block.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            for line in text.split("\n"):
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].lstrip()
                if data_str == "[DONE]":
                    yield MessageStopEvent()
                    continue

                try:
                    data = orjson.loads(data_str)
                except orjson.JSONDecodeError:
                    continue

                evt_type = data.get("type", "")

                if evt_type == "response.created":
                    resp = data.get("response", {}) or {}
                    yield with_raw(MessageStartEvent(model=resp.get("model", "")), data)

                elif evt_type == "response.output_item.added":
                    item = data.get("item", {}) or {}
                    current_item_index = data.get("output_index", 0)
                    item_type = item.get("type", "")
                    if item_type == "message":
                        yield with_raw(
                            PartStartEvent(index=current_item_index, part_type="text"), data
                        )
                    elif item_type == "function_call":
                        ws_counters[current_item_index] = 0
                        yield with_raw(
                            PartStartEvent(
                                index=current_item_index,
                                part_type="tool_call",
                                call_id=item.get("call_id", ""),
                                name=item.get("name", ""),
                            ),
                            data,
                        )
                    elif item_type == "reasoning":
                        yield with_raw(
                            PartStartEvent(index=current_item_index, part_type="reasoning"), data
                        )

                elif evt_type == "response.output_text.delta":
                    delta_text = data.get("delta", "")
                    if delta_text:
                        yield with_raw(
                            PartDeltaEvent(
                                index=data.get("output_index", current_item_index),
                                delta_type="text",
                                text=delta_text,
                            ),
                            data,
                        )

                elif evt_type == "response.output_text.done":
                    yield with_raw(
                        PartStopEvent(index=data.get("output_index", current_item_index)), data
                    )

                elif evt_type == "response.reasoning_summary_text.delta":
                    delta_text = data.get("delta", "")
                    if delta_text:
                        yield with_raw(
                            PartDeltaEvent(
                                index=data.get("output_index", current_item_index),
                                delta_type="reasoning",
                                text=delta_text,
                            ),
                            data,
                        )

                elif evt_type == "response.reasoning_summary_text.done":
                    yield with_raw(
                        PartStopEvent(index=data.get("output_index", current_item_index)), data
                    )

                elif evt_type == "response.function_call_arguments.delta":
                    idx = data.get("output_index", current_item_index)
                    delta_text = data.get("delta", "")
                    if not delta_text:
                        continue
                    if delta_text.strip() == "":
                        ws_counters[idx] = ws_counters.get(idx, 0) + len(delta_text)
                        if ws_counters[idx] > MAX_WHITESPACE_RUN:
                            yield with_raw(
                                ErrorEvent(
                                    error_type="invalid_request_error",
                                    message="Runaway whitespace in tool call arguments",
                                ),
                                data,
                            )
                            continue
                    else:
                        ws_counters[idx] = 0
                    yield with_raw(
                        PartDeltaEvent(index=idx, delta_type="json", text=delta_text), data
                    )

                elif evt_type in (
                    "response.function_call_arguments.done",
                    "response.output_item.done",
                ):
                    yield with_raw(
                        PartStopEvent(index=data.get("output_index", current_item_index)), data
                    )

                elif evt_type in ("response.completed", "response.incomplete"):
                    resp = data.get("response", {}) or {}
                    usage_data = resp.get("usage", {}) or {}
                    in_details = usage_data.get("input_tokens_details", {}) or {}
                    out_details = usage_data.get("output_tokens_details", {}) or {}
                    incomplete_reason = (resp.get("incomplete_details") or {}).get("reason") or ""
                    if evt_type == "response.completed":
                        normalized = "end_turn"
                    elif incomplete_reason == "max_output_tokens":
                        normalized = "max_tokens"
                    else:
                        normalized = "end_turn"
                    yield with_raw(
                        MessageDeltaEvent(
                            stop=StopInfo(
                                normalized=normalized,
                                provider_raw=resp.get("status", evt_type.split(".")[-1]),
                            ),
                            usage=Usage(
                                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                                cache_read_input_tokens=int(
                                    in_details.get("cached_tokens", 0) or 0
                                ),
                                reasoning_tokens=int(out_details.get("reasoning_tokens", 0) or 0),
                            ),
                        ),
                        data,
                    )

                elif evt_type == "response.failed":
                    err = (data.get("response", {}) or {}).get("error", {}) or {}
                    yield with_raw(
                        ErrorEvent(
                            error_type=err.get("type", "response_failed"),
                            message=err.get("message", ""),
                        ),
                        data,
                    )

                elif evt_type == "error":
                    err = data.get("error", {}) or {}
                    yield with_raw(
                        ErrorEvent(
                            error_type=err.get("type", "error"),
                            message=err.get("message", ""),
                        ),
                        data,
                    )


def _sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + orjson.dumps(payload) + b"\n\n"


async def render(events: AsyncIterator[CanonicalStreamEvent]) -> AsyncIterator[bytes]:
    """Render canonical IR events into OpenAI Responses semantic event SSE."""
    model = ""
    response_id = "resp_stream"

    async for event in events:
        if isinstance(event, MessageStartEvent):
            model = event.model
            raw_id = event._raw.get("id") if isinstance(event._raw, dict) else None
            if isinstance(raw_id, str) and raw_id:
                response_id = raw_id
            yield _sse(
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "model": model,
                        "status": "in_progress",
                        "output": [],
                    },
                }
            )
        elif isinstance(event, PartStartEvent):
            if event.part_type == "text":
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": event.index,
                        "item": {
                            "type": "message",
                            "id": f"msg_{event.index}",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    }
                )
            elif event.part_type == "tool_call":
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": event.index,
                        "item": {
                            "type": "function_call",
                            "id": f"fc_{event.index}",
                            "call_id": event.call_id or "",
                            "name": event.name or "",
                            "arguments": "",
                            "status": "in_progress",
                        },
                    }
                )
            elif event.part_type == "reasoning":
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": event.index,
                        "item": {"type": "reasoning", "id": f"rs_{event.index}", "summary": []},
                    }
                )
        elif isinstance(event, PartDeltaEvent):
            if event.delta_type == "text":
                yield _sse(
                    {
                        "type": "response.output_text.delta",
                        "output_index": event.index,
                        "delta": event.text,
                    }
                )
            elif event.delta_type == "json":
                yield _sse(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": event.index,
                        "delta": event.text,
                    }
                )
            elif event.delta_type == "reasoning":
                yield _sse(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "output_index": event.index,
                        "delta": event.text,
                    }
                )
        elif isinstance(event, PartStopEvent):
            yield _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": event.index,
                }
            )
        elif isinstance(event, MessageDeltaEvent):
            status = (
                "completed" if event.stop and event.stop.normalized == "end_turn" else "incomplete"
            )
            usage = event.usage
            yield _sse(
                {
                    "type": f"response.{status}",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "model": model,
                        "status": status,
                        "usage": {
                            "input_tokens": usage.input_tokens if usage else 0,
                            "output_tokens": usage.output_tokens if usage else 0,
                            "total_tokens": (usage.input_tokens + usage.output_tokens)
                            if usage
                            else 0,
                        },
                    },
                }
            )
        elif isinstance(event, MessageStopEvent):
            yield b"data: [DONE]\n\n"
        elif isinstance(event, ErrorEvent):
            yield _sse(
                {"type": "error", "error": {"type": event.error_type, "message": event.message}}
            )
