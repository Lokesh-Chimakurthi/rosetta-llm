"""Tests for OpenAI Responses streaming codec behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator

import orjson

from rosetta.ir.events import MessageStopEvent, PartStopEvent
from rosetta.stream_codecs.openai_responses import parse


async def _sse_events(events: list[dict[str, object]]) -> AsyncIterator[bytes]:
    for event in events:
        yield b"data: " + orjson.dumps(event) + b"\n\n"
    yield b"data: [DONE]\n\n"


async def test_responses_completed_emits_single_message_stop() -> None:
    """Responses completion should close the Anthropic stream exactly once.

    Some providers send both output_text.done and output_item.done, then
    response.completed, then [DONE]. Rosetta should normalize that into one
    content block stop and one message stop for Anthropic clients such as
    Claude Code.
    """

    events = [
        {"type": "response.created", "response": {"model": "gpt-test"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message"},
        },
        {"type": "response.output_text.delta", "output_index": 0, "delta": "OK"},
        {"type": "response.output_text.done", "output_index": 0},
        {"type": "response.output_item.done", "output_index": 0},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    ]

    parsed = [event async for event in parse(_sse_events(events))]

    assert sum(isinstance(event, PartStopEvent) for event in parsed) == 1
    assert sum(isinstance(event, MessageStopEvent) for event in parsed) == 1
