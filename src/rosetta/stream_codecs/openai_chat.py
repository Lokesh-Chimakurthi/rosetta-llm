"""OpenAI Chat Completions streaming codec — SSE parse/render via IR events."""

from __future__ import annotations

import time
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
from rosetta.stop_reasons import OPENAI_CHAT_STOP_IN as _STOP_IN
from rosetta.stop_reasons import OPENAI_CHAT_STOP_OUT as _STOP_OUT

_TEXT_BLOCK_INDEX = 0  # Conventional index for the assistant text block.


async def parse(chunks: AsyncIterator[bytes]) -> AsyncIterator[CanonicalStreamEvent]:
    """Parse OpenAI Chat SSE byte stream into canonical IR events."""
    buffer = b""
    started = False
    text_block_open = False
    reasoning_block_open = False
    tool_seen: set[int] = set()

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
                    if text_block_open:
                        yield PartStopEvent(index=_TEXT_BLOCK_INDEX)
                        text_block_open = False
                    yield MessageStopEvent()
                    continue

                try:
                    data = orjson.loads(data_str)
                except orjson.JSONDecodeError:
                    continue

                if not started:
                    started = True
                    yield with_raw(MessageStartEvent(model=data.get("model", "")), data)

                # Top-level error envelope
                if "error" in data and not data.get("choices"):
                    err = data["error"]
                    yield with_raw(
                        ErrorEvent(
                            error_type=str(err.get("type", "error")),
                            message=str(err.get("message", "")),
                        ),
                        data,
                    )
                    continue

                choices = data.get("choices") or []
                if not choices:
                    # Some providers send a final usage-only chunk.
                    usage_data = data.get("usage")
                    if usage_data:
                        yield MessageDeltaEvent(
                            usage=Usage(
                                input_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
                                output_tokens=int(usage_data.get("completion_tokens", 0) or 0),
                            )
                        )
                    continue

                choice = choices[0]
                delta = choice.get("delta") or {}

                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    if not reasoning_block_open:
                        reasoning_block_open = True
                        yield PartStartEvent(index=_TEXT_BLOCK_INDEX, part_type="reasoning")
                    yield with_raw(
                        PartDeltaEvent(
                            index=_TEXT_BLOCK_INDEX,
                            delta_type="reasoning",
                            text=str(reasoning),
                        ),
                        data,
                    )

                content = delta.get("content")
                if content:
                    if reasoning_block_open:
                        yield PartStopEvent(index=_TEXT_BLOCK_INDEX)
                        reasoning_block_open = False
                    if not text_block_open:
                        text_block_open = True
                        yield PartStartEvent(index=_TEXT_BLOCK_INDEX, part_type="text")
                    yield with_raw(
                        PartDeltaEvent(
                            index=_TEXT_BLOCK_INDEX,
                            delta_type="text",
                            text=content,
                        ),
                        data,
                    )

                for tc in delta.get("tool_calls") or []:
                    idx = (
                        int(tc.get("index", 0)) + 1
                    )  # offset to avoid colliding with text block 0
                    func = tc.get("function") or {}
                    if idx not in tool_seen:
                        tool_seen.add(idx)
                        if text_block_open:
                            yield PartStopEvent(index=_TEXT_BLOCK_INDEX)
                            text_block_open = False
                        if reasoning_block_open:
                            yield PartStopEvent(index=_TEXT_BLOCK_INDEX)
                            reasoning_block_open = False
                        yield with_raw(
                            PartStartEvent(
                                index=idx,
                                part_type="tool_call",
                                call_id=tc.get("id", ""),
                                name=func.get("name", ""),
                            ),
                            data,
                        )
                    args = func.get("arguments")
                    if args:
                        yield with_raw(
                            PartDeltaEvent(index=idx, delta_type="json", text=args), data
                        )

                finish = choice.get("finish_reason")
                if finish:
                    if text_block_open:
                        yield PartStopEvent(index=_TEXT_BLOCK_INDEX)
                        text_block_open = False
                    if reasoning_block_open:
                        yield PartStopEvent(index=_TEXT_BLOCK_INDEX)
                        reasoning_block_open = False
                    for idx in tool_seen:
                        yield PartStopEvent(index=idx)
                    tool_seen.clear()
                    usage_data = data.get("usage") or {}
                    yield with_raw(
                        MessageDeltaEvent(
                            stop=StopInfo(
                                normalized=_STOP_IN.get(finish, finish), provider_raw=finish
                            ),
                            usage=Usage(
                                input_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
                                output_tokens=int(usage_data.get("completion_tokens", 0) or 0),
                            ),
                        ),
                        data,
                    )


def _sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + orjson.dumps(payload) + b"\n\n"


async def render(events: AsyncIterator[CanonicalStreamEvent]) -> AsyncIterator[bytes]:
    """Render canonical IR events into OpenAI Chat SSE bytes."""
    model = ""
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    tool_index_map: dict[int, int] = {}  # IR index → OpenAI tool_calls[].index
    final_usage: Usage | None = None
    final_finish: str | None = None
    sent_done = False

    def base_chunk() -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}}],
        }

    async for event in events:
        if isinstance(event, MessageStartEvent):
            model = event.model

        elif isinstance(event, PartStartEvent):
            if event.part_type == "tool_call":
                next_idx = len(tool_index_map)
                tool_index_map[event.index] = next_idx
                delta: dict[str, Any] = {
                    "tool_calls": [
                        {
                            "index": next_idx,
                            "id": event.call_id or "",
                            "type": "function",
                            "function": {"name": event.name or "", "arguments": ""},
                        }
                    ]
                }
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True
                chunk = base_chunk()
                chunk["choices"][0]["delta"] = delta
                yield _sse(chunk)
            elif event.part_type == "text" and not sent_role:
                # Emit role-only chunk for clients that expect it.
                chunk = base_chunk()
                chunk["choices"][0]["delta"] = {"role": "assistant", "content": ""}
                sent_role = True
                yield _sse(chunk)

        elif isinstance(event, PartDeltaEvent):
            if event.delta_type == "text":
                delta = {"content": event.text}
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True
                chunk = base_chunk()
                chunk["choices"][0]["delta"] = delta
                yield _sse(chunk)
            elif event.delta_type == "json":
                oai_idx = tool_index_map.get(event.index, 0)
                chunk = base_chunk()
                chunk["choices"][0]["delta"] = {
                    "tool_calls": [{"index": oai_idx, "function": {"arguments": event.text}}]
                }
                yield _sse(chunk)
            elif event.delta_type == "reasoning":
                delta = {"reasoning_content": event.text}
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True
                chunk = base_chunk()
                chunk["choices"][0]["delta"] = delta
                yield _sse(chunk)

        elif isinstance(event, MessageDeltaEvent):
            if event.usage:
                final_usage = event.usage
            if event.stop:
                final_finish = _STOP_OUT.get(
                    event.stop.normalized, event.stop.provider_raw or "stop"
                )

        elif isinstance(event, MessageStopEvent):
            chunk = base_chunk()
            chunk["choices"][0]["finish_reason"] = final_finish or "stop"
            if final_usage:
                chunk["usage"] = {
                    "prompt_tokens": final_usage.input_tokens,
                    "completion_tokens": final_usage.output_tokens,
                    "total_tokens": final_usage.input_tokens + final_usage.output_tokens,
                }
            yield _sse(chunk)
            yield b"data: [DONE]\n\n"
            sent_done = True

        elif isinstance(event, ErrorEvent):
            yield _sse({"error": {"message": event.message, "type": event.error_type}})

    # Defensive: ensure [DONE] is sent even if upstream closed without message_stop.
    if not sent_done:
        chunk = base_chunk()
        chunk["choices"][0]["finish_reason"] = final_finish or "stop"
        if final_usage:
            chunk["usage"] = {
                "prompt_tokens": final_usage.input_tokens,
                "completion_tokens": final_usage.output_tokens,
                "total_tokens": final_usage.input_tokens + final_usage.output_tokens,
            }
        yield _sse(chunk)
        yield b"data: [DONE]\n\n"
