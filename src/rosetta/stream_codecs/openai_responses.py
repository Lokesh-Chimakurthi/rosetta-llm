"""OpenAI Responses streaming codec — semantic event parse/render via IR events."""

from __future__ import annotations

import uuid
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

_SEARCH_TOOL_NAMES = frozenset({"tool_search_tool_regex", "tool_search_tool_bm25"})


async def parse(chunks: AsyncIterator[bytes]) -> AsyncIterator[CanonicalStreamEvent]:
    """Parse OpenAI Responses semantic event stream into canonical IR events."""
    buffer = b""
    current_item_index = -1
    ws_counters: dict[int, int] = {}
    pending_search_id = ""
    added_items: dict[int, dict[str, Any]] = {}

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
                    elif item_type in ("tool_search_call", "tool_search_output"):
                        # The payload usually arrives with output_item.done;
                        # stash it so a done without an item still emits.
                        added_items[current_item_index] = item
                        continue

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
                    idx = data.get("output_index", current_item_index)
                    stashed = added_items.pop(idx, None)
                    item = data.get("item") or stashed or {}
                    if item.get("type") in ("tool_search_call", "tool_search_output"):
                        execution = item.get("execution")
                        if execution == "client":
                            raise ValueError(
                                "client-executed tool search has no anthropic equivalent; "
                                "refusing to translate"
                            )
                        if execution is None:
                            raise ValueError(
                                "tool search item without execution cannot be translated"
                            )
                    if item.get("type") == "tool_search_call":
                        search_id = item.get("call_id") or f"srvtoolu_{uuid.uuid4().hex[:16]}"
                        pending_search_id = search_id
                        name = (
                            "tool_search_tool_bm25"
                            if item.get("search_variant") == "bm25"
                            else "tool_search_tool_regex"
                        )
                        yield with_raw(
                            PartStartEvent(
                                index=idx,
                                part_type="server_tool_use",
                                call_id=search_id,
                                name=name,
                                payload={"input": item.get("arguments") or {}},
                            ),
                            data,
                        )
                        yield with_raw(PartStopEvent(index=idx), data)
                        continue
                    if item.get("type") == "tool_search_output":
                        search_id = item.get("call_id") or pending_search_id
                        if not search_id:
                            raise ValueError(
                                "tool_search_output without a preceding tool_search_call "
                                "cannot be translated"
                            )
                        pending_search_id = ""
                        tools = [t for t in item.get("tools") or [] if isinstance(t, dict)]
                        references = [
                            {"type": "tool_reference", "tool_name": t.get("name", "")}
                            for t in tools
                        ]
                        yield with_raw(
                            PartStartEvent(
                                index=idx,
                                part_type="tool_search_tool_result",
                                call_id=search_id,
                                # `tools` keeps full definitions for responses
                                # renders; the anthropic wire only has names.
                                payload={"tool_references": references, "tools": tools},
                            ),
                            data,
                        )
                        yield with_raw(PartStopEvent(index=idx), data)
                        continue
                    yield with_raw(PartStopEvent(index=idx), data)

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
    search_items: dict[int, dict[str, Any]] = {}

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
            elif event.part_type == "server_tool_use" and event.name in _SEARCH_TOOL_NAMES:
                item: dict[str, Any] = {
                    "type": "tool_search_call",
                    "execution": "server",
                    "call_id": None,
                    "status": "completed",
                    "arguments": event.payload.get("input") or {},
                }
                search_items[event.index] = item
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": event.index,
                        "item": item,
                    }
                )
            elif event.part_type == "tool_search_tool_result":
                full_tools = event.payload.get("tools")
                if full_tools:
                    tools = full_tools
                else:
                    tools = [
                        {
                            "type": "function",
                            "name": r.get("tool_name", ""),
                            "parameters": {"type": "object", "properties": {}},
                        }
                        for r in event.payload.get("tool_references") or []
                    ]
                item = {
                    "type": "tool_search_output",
                    "execution": "server",
                    "call_id": None,
                    "status": "completed",
                    "tools": tools,
                }
                search_items[event.index] = item
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": event.index,
                        "item": item,
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
            # Search blocks are one-shot: repeat the item on `done` so parsers
            # reading only `done` (this file's own included) keep the payload.
            done_payload: dict[str, Any] = {"output_index": event.index}
            stashed = search_items.pop(event.index, None)
            if stashed is not None:
                done_payload["item"] = stashed
            yield _sse({"type": "response.output_item.done", **done_payload})
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
