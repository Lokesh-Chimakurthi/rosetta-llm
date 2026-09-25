"""Anthropic stream render tests — held terminal message_delta and cache fields."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import orjson

from rosetta.ir.events import (
    CanonicalStreamEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PartDeltaEvent,
    PartStartEvent,
    PartStopEvent,
)
from rosetta.ir.response import StopInfo, Usage
from rosetta.stream_codecs import anthropic as ac_stream


async def _feed(*events: CanonicalStreamEvent) -> AsyncIterator[CanonicalStreamEvent]:
    for e in events:
        yield e


async def _render(*events: CanonicalStreamEvent) -> bytes:
    return b"".join([chunk async for chunk in ac_stream.render(_feed(*events))])


def _sse_events(rendered: bytes) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for block in rendered.split(b"\n\n"):
        if not block.strip():
            continue
        name, data_str = "", ""
        for line in block.decode().split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].lstrip()
        out.append((name, orjson.loads(data_str)))
    return out


def _full_sequence(stop_normalized: str = "end_turn") -> list[CanonicalStreamEvent]:
    return [
        MessageStartEvent(model="m1"),
        PartStartEvent(index=0, part_type="text"),
        PartDeltaEvent(index=0, delta_type="text", text="hi"),
        PartStopEvent(index=0),
        MessageDeltaEvent(
            stop=StopInfo(normalized=stop_normalized, provider_raw=stop_normalized),
            usage=Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=7),
        ),
        MessageStopEvent(),
    ]


async def test_anthropic_render_single_terminal_delta_with_cache_fields() -> None:
    """Exactly one message_delta, between the last content_block_stop and message_stop."""
    rendered = await _render(*_full_sequence())
    events = _sse_events(rendered)
    names = [name for name, _ in events]

    assert names.count("message_delta") == 1
    delta_idx = names.index("message_delta")
    assert delta_idx > len(names) - 1 - names[::-1].index("content_block_stop")
    assert names.index("message_stop") == delta_idx + 1

    _, payload = events[delta_idx]
    # Anthropic semantics: 10 - 7 cached = 3 non-cached input tokens.
    assert payload["usage"]["input_tokens"] == 3
    assert payload["usage"]["cache_read_input_tokens"] == 7
    assert "cache_creation_input_tokens" not in payload["usage"]
    assert payload["delta"]["stop_reason"] == "end_turn"


async def test_anthropic_render_message_start_has_four_zeroed_usage_fields() -> None:
    """message_start usage zeros all four fields — signals caching support."""
    rendered = await _render(*_full_sequence())
    start = next(data for name, data in _sse_events(rendered) if name == "message_start")
    assert start["message"]["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


async def test_anthropic_render_truncated_stream_flushes_held_delta() -> None:
    """Delta held with no message_stop: still flushed at generator end."""
    rendered = await _render(
        MessageStartEvent(model="m1"),
        MessageDeltaEvent(
            stop=StopInfo(normalized="max_tokens", provider_raw="max_tokens"),
            usage=Usage(input_tokens=10, output_tokens=5),
        ),
    )
    events = _sse_events(rendered)
    names = [name for name, _ in events]

    assert names.count("message_delta") == 1
    assert names[-1] == "message_delta"
    _, payload = events[-1]
    assert payload["usage"]["input_tokens"] == 10
    assert payload["delta"]["stop_reason"] == "max_tokens"


async def test_anthropic_render_multiple_deltas_merge_last_wins() -> None:
    """Defensive: several deltas must not corrupt — merged, single emission."""
    rendered = await _render(
        MessageStartEvent(model="m1"),
        MessageDeltaEvent(usage=Usage(input_tokens=10, output_tokens=1)),
        MessageDeltaEvent(
            stop=StopInfo(normalized="end_turn", provider_raw="end_turn"),
            usage=Usage(input_tokens=4, output_tokens=2),
        ),
        MessageStopEvent(),
    )
    events = _sse_events(rendered)
    names = [name for name, _ in events]

    assert names.count("message_delta") == 1
    assert names[-1] == "message_stop"
    _, payload = events[names.index("message_delta")]
    assert payload["usage"]["input_tokens"] == 4
    assert payload["usage"]["output_tokens"] == 2
    assert payload["delta"]["stop_reason"] == "end_turn"
