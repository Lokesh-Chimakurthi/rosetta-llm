"""Anthropic Messages streaming codec — SSE parse/render via IR events.

`wrap_with_ping` injects a synthetic ping event when the upstream is silent
for `interval` seconds, so Anthropic SDK clients don't disconnect during
slow upstream generation.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    PingEvent,
)
from rosetta.ir.helpers import with_raw
from rosetta.ir.response import StopInfo, Usage
from rosetta.stop_reasons import ANTHROPIC_STOP as _STOP_NORM

_SEARCH_TOOL_NAMES = frozenset({"tool_search_tool_regex", "tool_search_tool_bm25"})


async def parse(chunks: AsyncIterator[bytes]) -> AsyncIterator[CanonicalStreamEvent]:
    """Parse Anthropic SSE byte stream into canonical IR events."""
    buffer = b""
    current_block_index = -1

    async for chunk in chunks:
        buffer += chunk
        while b"\n\n" in buffer:
            line_block, buffer = buffer.split(b"\n\n", 1)
            text = line_block.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            event_type = ""
            data_str = ""
            for line in text.split("\n"):
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].lstrip()

            if not data_str:
                continue
            try:
                data = orjson.loads(data_str)
            except orjson.JSONDecodeError:
                continue

            evt = event_type or data.get("type", "")

            if evt == "message_start":
                msg = data.get("message", {}) or {}
                yield with_raw(MessageStartEvent(model=msg.get("model", "")), msg)
            elif evt == "content_block_start":
                block = data.get("content_block", {}) or {}
                current_block_index = data.get("index", 0)
                btype = block.get("type", "")
                part_type_map = {
                    "text": "text",
                    "thinking": "reasoning",
                    "tool_use": "tool_call",
                    "server_tool_use": "server_tool_use",
                    "tool_search_tool_result": "tool_search_tool_result",
                }
                if btype in ("tool_use", "server_tool_use"):
                    block_id = block.get("id")
                    block_name = block.get("name")
                elif btype == "tool_search_tool_result":
                    block_id = block.get("tool_use_id")
                    block_name = None
                else:
                    block_id = None
                    block_name = None
                payload: dict[str, Any] = {}
                if btype == "server_tool_use" and block.get("name", "") in _SEARCH_TOOL_NAMES:
                    payload = {"input": block.get("input") or {}}
                elif btype == "tool_search_tool_result":
                    references = (block.get("content") or {}).get("tool_references") or []
                    payload = {"tool_references": references}
                yield with_raw(
                    PartStartEvent(
                        index=current_block_index,
                        part_type=part_type_map.get(btype, btype),
                        call_id=block_id,
                        name=block_name,
                        payload=payload,
                    ),
                    block,
                )
            elif evt == "content_block_delta":
                idx = data.get("index", current_block_index)
                delta = data.get("delta", {}) or {}
                dtype = delta.get("type", "")
                if dtype == "text_delta":
                    yield PartDeltaEvent(index=idx, delta_type="text", text=delta.get("text", ""))
                elif dtype == "input_json_delta":
                    yield PartDeltaEvent(
                        index=idx, delta_type="json", text=delta.get("partial_json", "")
                    )
                elif dtype == "thinking_delta":
                    yield PartDeltaEvent(
                        index=idx, delta_type="reasoning", text=delta.get("thinking", "")
                    )
                elif dtype == "signature_delta":
                    yield PartDeltaEvent(
                        index=idx, delta_type="signature", text=delta.get("signature", "")
                    )
            elif evt == "content_block_stop":
                yield PartStopEvent(index=data.get("index", current_block_index))
            elif evt == "message_delta":
                delta = data.get("delta", {}) or {}
                usage_data = data.get("usage", {}) or {}
                stop_reason = delta.get("stop_reason", "") or ""
                yield MessageDeltaEvent(
                    stop=StopInfo(
                        normalized=_STOP_NORM.get(stop_reason, stop_reason),
                        provider_raw=stop_reason,
                        stop_sequence=delta.get("stop_sequence"),
                    ),
                    usage=Usage(
                        input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                        output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                        cache_read_input_tokens=int(
                            usage_data.get("cache_read_input_tokens", 0) or 0
                        ),
                        cache_creation_input_tokens=int(
                            usage_data.get("cache_creation_input_tokens", 0) or 0
                        ),
                    ),
                )
            elif evt == "message_stop":
                yield MessageStopEvent()
            elif evt == "ping":
                yield PingEvent()
            elif evt == "error":
                err = data.get("error", {}) or {}
                yield with_raw(
                    ErrorEvent(
                        error_type=err.get("type", "error"),
                        message=err.get("message", ""),
                    ),
                    err,
                )


def _sse(event_name: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: ".encode() + orjson.dumps(payload) + b"\n\n"


_DELTA_TYPE_MAP = {
    "text": "text_delta",
    "json": "input_json_delta",
    "reasoning": "thinking_delta",
    "signature": "signature_delta",
}


async def render(events: AsyncIterator[CanonicalStreamEvent]) -> AsyncIterator[bytes]:
    """Render canonical IR events into Anthropic SSE bytes."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    started = False

    async for event in events:
        if isinstance(event, MessageStartEvent):
            started = True
            raw_id = event._raw.get("id") if isinstance(event._raw, dict) else None
            if isinstance(raw_id, str) and raw_id:
                message_id = raw_id
            yield _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": event.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        elif isinstance(event, PartStartEvent):
            block: dict[str, Any] = {}
            if event.part_type == "tool_call":
                block = {
                    "type": "tool_use",
                    "id": event.call_id or "",
                    "name": event.name or "",
                    "input": {},
                }
            elif event.part_type == "reasoning":
                block = {"type": "thinking", "thinking": "", "signature": ""}
            elif event.part_type == "server_tool_use":
                # client-executed search is refused at Responses parse time.
                block = {
                    "type": "server_tool_use",
                    "id": event.call_id or "",
                    "name": event.name or "",
                    "input": event.payload.get("input") or {},
                }
            elif event.part_type == "tool_search_tool_result":
                block = {
                    "type": "tool_search_tool_result",
                    "tool_use_id": event.call_id or "",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": event.payload.get("tool_references") or [],
                    },
                }
            elif event.part_type == "text":
                block = {"type": "text", "text": ""}
            else:
                block = {"type": event.part_type}
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": event.index,
                    "content_block": block,
                },
            )
        elif isinstance(event, PartDeltaEvent):
            dt = _DELTA_TYPE_MAP.get(event.delta_type, "text_delta")
            delta: dict[str, Any] = {"type": dt}
            if dt == "text_delta":
                delta["text"] = event.text
            elif dt == "input_json_delta":
                delta["partial_json"] = event.text
            elif dt == "thinking_delta":
                delta["thinking"] = event.text
            elif dt == "signature_delta":
                delta["signature"] = event.text
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": event.index,
                    "delta": delta,
                },
            )
        elif isinstance(event, PartStopEvent):
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": event.index})
        elif isinstance(event, MessageDeltaEvent):
            delta_data: dict[str, Any] = {}
            if event.stop:
                delta_data["stop_reason"] = _STOP_NORM.get(
                    event.stop.normalized, event.stop.provider_raw
                )
                delta_data["stop_sequence"] = event.stop.stop_sequence
            usage_data: dict[str, Any] = {
                "output_tokens": event.usage.output_tokens if event.usage else 0
            }
            if event.usage and event.usage.input_tokens:
                usage_data["input_tokens"] = event.usage.input_tokens
            yield _sse(
                "message_delta",
                {"type": "message_delta", "delta": delta_data, "usage": usage_data},
            )
        elif isinstance(event, MessageStopEvent):
            yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        elif isinstance(event, PingEvent):
            yield b'event: ping\ndata: {"type":"ping"}\n\n'
        elif isinstance(event, ErrorEvent):
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": event.error_type, "message": event.message},
                },
            )

    # Ensure clients see message_stop even if upstream forgot.
    if started:
        # No-op: caller is expected to feed MessageStopEvent. Defensive only.
        pass


async def wrap_with_ping(
    events: AsyncIterator[CanonicalStreamEvent],
    interval: float = 15.0,
) -> AsyncIterator[CanonicalStreamEvent]:
    """Inject ping events when the source is silent for `interval` seconds.

    Uses a producer task feeding a queue so the source generator is never
    cancelled mid-await — only its consumer is parked on the queue.
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def producer() -> None:
        try:
            async for ev in events:
                await queue.put(("event", ev))
            await queue.put(("end", None))
        except Exception as exc:
            await queue.put(("error", exc))

    task = asyncio.create_task(producer())
    try:
        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                yield PingEvent()
                continue
            if kind == "event":
                yield value
            elif kind == "end":
                return
            elif kind == "error":
                raise value
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
