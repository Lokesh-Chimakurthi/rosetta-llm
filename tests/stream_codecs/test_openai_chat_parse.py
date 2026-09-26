"""OpenAI Chat stream parse tests — usage accumulation and single terminal delta."""

from __future__ import annotations

from collections.abc import AsyncIterator

import orjson

from rosetta.ir.events import (
    MessageDeltaEvent,
    MessageStopEvent,
    PartStopEvent,
)
from rosetta.stream_codecs import openai_chat as oc_stream


def _chunk(payload: dict[str, object]) -> bytes:
    return b"data: " + orjson.dumps(payload) + b"\n\n"  # type: ignore[arg-type]


async def _feed(*parts: bytes) -> AsyncIterator[bytes]:
    for p in parts:
        yield p


async def _parse(*parts: bytes) -> list[object]:
    return [ev async for ev in oc_stream.parse(_feed(*parts))]


def _deltas(events: list[object]) -> list[MessageDeltaEvent]:
    return [e for e in events if isinstance(e, MessageDeltaEvent)]


async def test_chat_stream_usage_after_finish_then_done_single_delta() -> None:
    """Usage-only chunk after finish_reason: one delta, terminal, fully populated.

    Fixture from the live octo probe: finish chunk with null usage, then a
    choices-empty usage chunk, then [DONE].
    """
    events = await _parse(
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            }
        ),
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "max_tokens"}],
                "usage": None,
            }
        ),
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [],
                "usage": {
                    "prompt_tokens": 18,
                    "completion_tokens": 40,
                    "prompt_tokens_details": {"cached_tokens": 7},
                },
            }
        ),
        b"data: [DONE]\n\n",
    )

    deltas = _deltas(events)
    assert len(deltas) == 1, "must emit exactly one MessageDeltaEvent"
    # Terminal: emitted after the usage chunk, immediately before message_stop.
    assert isinstance(events[-2], MessageDeltaEvent)
    assert isinstance(events[-1], MessageStopEvent)

    delta = deltas[0]
    assert delta.usage is not None
    assert delta.usage.input_tokens == 18
    assert delta.usage.output_tokens == 40
    assert delta.usage.cache_read_input_tokens == 7
    assert delta.stop is not None
    assert delta.stop.normalized == "max_tokens"
    assert delta.stop.provider_raw == "max_tokens"


async def test_chat_stream_usage_inline_on_finish_chunk_single_delta() -> None:
    """Usage inlined on the finish_reason chunk yields the same single-delta outcome."""
    events = await _parse(
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            }
        ),
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            }
        ),
        b"data: [DONE]\n\n",
    )

    deltas = _deltas(events)
    assert len(deltas) == 1
    assert isinstance(events[-2], MessageDeltaEvent)
    assert isinstance(events[-1], MessageStopEvent)

    delta = deltas[0]
    assert delta.usage is not None
    assert delta.usage.input_tokens == 5
    assert delta.usage.output_tokens == 2
    assert delta.usage.cache_read_input_tokens == 3
    assert delta.stop is not None
    assert delta.stop.normalized == "end_turn"


async def test_chat_stream_usage_zero_when_upstream_sends_none() -> None:
    """Stream ending without any usage still gets one zero-usage delta before stop."""
    events = await _parse(
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            }
        ),
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            }
        ),
        b"data: [DONE]\n\n",
    )

    deltas = _deltas(events)
    assert len(deltas) == 1
    assert isinstance(events[-2], MessageDeltaEvent)
    assert isinstance(events[-1], MessageStopEvent)

    delta = deltas[0]
    assert delta.usage is not None
    assert delta.usage.input_tokens == 0
    assert delta.usage.output_tokens == 0


async def test_chat_stream_usage_stop_none_when_no_finish_reason() -> None:
    """No finish_reason ever, [DONE] arrives: delta carries stop=None."""
    events = await _parse(
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            }
        ),
        b"data: [DONE]\n\n",
    )

    deltas = _deltas(events)
    assert len(deltas) == 1
    assert isinstance(events[-2], MessageDeltaEvent)
    assert isinstance(events[-1], MessageStopEvent)

    delta = deltas[0]
    assert delta.stop is None
    assert delta.usage is not None
    # Open text block is closed before the terminal delta.
    assert isinstance(events[-3], PartStopEvent)


async def test_chat_stream_usage_flushed_on_exhaustion_without_done() -> None:
    """Upstream closes without [DONE]: delta + stop still flush, well-formed."""
    events = await _parse(
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            }
        ),
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            }
        ),
    )

    deltas = _deltas(events)
    assert len(deltas) == 1
    assert isinstance(events[-2], MessageDeltaEvent)
    assert isinstance(events[-1], MessageStopEvent)

    delta = deltas[0]
    assert delta.usage is not None
    assert delta.usage.input_tokens == 9
    assert delta.usage.output_tokens == 4
    assert delta.stop is not None
    assert delta.stop.normalized == "max_tokens"


async def test_chat_stream_usage_double_done_emits_single_terminal_pair() -> None:
    """A pathological double [DONE] must not re-emit the terminal pair."""
    events = await _parse(
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
        b"data: [DONE]\n\n",
    )

    assert len(_deltas(events)) == 1
    assert sum(isinstance(e, MessageStopEvent) for e in events) == 1


async def test_chat_stream_usage_reasoning_block_closed_at_done_without_finish() -> None:
    """Reasoning-only stream with no finish_reason: the block is closed at [DONE]."""
    events = await _parse(
        _chunk(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": "thinking"}, "finish_reason": None}
                ],
            }
        ),
        b"data: [DONE]\n\n",
    )

    stops = [e for e in events if isinstance(e, PartStopEvent)]
    assert len(stops) == 1, "the open reasoning block must be closed at the terminal"
    assert isinstance(events[-3], PartStopEvent)
    assert isinstance(events[-2], MessageDeltaEvent)
    assert isinstance(events[-1], MessageStopEvent)
