"""Wire-contract validators for what Rosetta sends to clients.

Each validator raises AssertionError whose message starts with the violated
rule in brackets, e.g. "[anthropic.duplicate_block_stop] ...", so a negative
test can prove it failed for the intended reason. Schemas come from the
official SDKs; ordering rules come from each API's streaming documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, get_args

import orjson
from anthropic.types import Message, RawMessageStreamEvent
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import BaseModel, ValidationError


def _models_by_type(annotated_union: Any) -> dict[str, type[BaseModel]]:
    """Map each event `type` literal to its SDK model.

    The SDK unions are `Annotated[Union[...], <SDK discriminator>]`, metadata
    pydantic ignores; dispatching on `type` here yields one precise error
    instead of one per union member.
    """
    union, *_ = get_args(annotated_union)
    return {
        literal: model
        for model in get_args(union)
        for literal in get_args(model.model_fields["type"].annotation)
    }


_ANTHROPIC_EVENTS = _models_by_type(RawMessageStreamEvent)
_RESPONSES_EVENTS = _models_by_type(ResponseStreamEvent)
_RESPONSES_TERMINAL = frozenset({"response.completed", "response.incomplete", "response.failed"})


@dataclass(frozen=True, slots=True)
class SseEvent:
    event: str | None
    data: str


def _fail(rule: str, detail: str) -> NoReturn:
    raise AssertionError(f"[{rule}] {detail}")


def parse_sse(raw: bytes) -> list[SseEvent]:
    """Split an SSE body into events per the WHATWG event-stream rules.

    Comment lines are ignored, multiple data lines join with "\\n", and blocks
    without data (comment-only keepalives) dispatch nothing.
    """
    events: list[SseEvent] = []
    name: str | None = None
    data: list[str] = []
    for line in raw.decode().replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line:
            if data:
                events.append(SseEvent(name, "\n".join(data)))
            name = None
            data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
    if data:
        events.append(SseEvent(name, "\n".join(data)))
    return events


def _json(rule: str, text: str, where: str) -> Any:
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError as exc:
        _fail(rule, f"{where}: data is not JSON ({exc}): {text[:200]}")


def _model(rule: str, model: type[BaseModel], body: Any, where: str) -> None:
    try:
        model.model_validate(body)
    except ValidationError as exc:
        _fail(rule, f"{where}: {exc}")


def _event(rule: str, models: dict[str, type[BaseModel]], body: Any, where: str) -> None:
    kind = body.get("type") if isinstance(body, dict) else None
    model = models.get(kind) if isinstance(kind, str) else None
    if model is None:
        _fail(rule, f"{where}: unknown event type {kind!r}")
    _model(rule, model, body, where)


def validate_anthropic_message(body: Any) -> None:
    _model("anthropic.message_schema", Message, body, "message")


def validate_anthropic_stream(raw: bytes) -> None:
    """Anthropic Messages stream lifecycle.

    message_start first; content-block indices dense from 0; one start and
    one stop per block; no event for a block not open; a non-null stop_reason
    before message_stop; message_stop last unless the stream ends in error.
    """
    events = parse_sse(raw)
    if not events:
        _fail("anthropic.terminal", "stream has no events")
    opened: set[int] = set()
    closed: set[int] = set()
    saw_stop_reason = False
    last = len(events) - 1
    for i, ev in enumerate(events):
        where = f"event {i}"
        data = _json("anthropic.json", ev.data, where)
        kind = data.get("type") if isinstance(data, dict) else None
        if ev.event != kind:
            _fail("anthropic.event_name", f"{where}: event {ev.event!r} carries type {kind!r}")
        if (i == 0) != (kind == "message_start"):
            _fail("anthropic.message_start_first", f"{where}: {kind!r}")
        if kind == "ping":
            continue
        if kind == "error":
            if i != last:
                _fail("anthropic.error_last", f"{where}: events follow the error")
            if not isinstance(data.get("error"), dict):
                _fail("anthropic.error_schema", f"{where}: error has no error object")
            return
        _event("anthropic.sdk_schema", _ANTHROPIC_EVENTS, data, where)
        index = data.get("index")
        if kind == "content_block_start":
            if index != len(opened):
                _fail(
                    "anthropic.block_index_dense",
                    f"{where}: start index {index}, expected {len(opened)}",
                )
            opened.add(index)
        elif kind == "content_block_delta":
            if index not in opened or index in closed:
                _fail("anthropic.block_not_open", f"{where}: delta for block {index}")
        elif kind == "content_block_stop":
            if index not in opened:
                _fail("anthropic.block_not_open", f"{where}: stop for unstarted block {index}")
            if index in closed:
                _fail("anthropic.duplicate_block_stop", f"{where}: block {index} stopped twice")
            closed.add(index)
        elif kind == "message_delta":
            saw_stop_reason = saw_stop_reason or data["delta"].get("stop_reason") is not None
        elif kind == "message_stop":
            if i != last:
                _fail("anthropic.terminal", f"{where}: events follow message_stop")
            if not saw_stop_reason:
                _fail("anthropic.stop_reason_before_stop", f"{where}: no stop_reason was sent")
            if opened != closed:
                _fail("anthropic.blocks_closed", f"{where}: open blocks {sorted(opened - closed)}")
            return
    _fail("anthropic.terminal", "stream ended without message_stop or error")


def validate_chat_completion(body: Any) -> None:
    _model("chat.completion_schema", ChatCompletion, body, "completion")


def validate_chat_stream(raw: bytes, *, include_usage: bool) -> None:
    """Chat Completions stream: SDK chunks, one finish_reason, [DONE] last.

    With include_usage, the last JSON chunk has no choices and carries usage.
    """
    events = parse_sse(raw)
    if not events or events[-1].data != "[DONE]":
        _fail("chat.done_last", "stream does not end with data: [DONE]")
    chunks: list[Any] = []
    for i, ev in enumerate(events[:-1]):
        if ev.data == "[DONE]":
            _fail("chat.done_last", f"event {i}: [DONE] before the end")
        chunk = _json("chat.json", ev.data, f"event {i}")
        _model("chat.sdk_schema", ChatCompletionChunk, chunk, f"event {i}")
        chunks.append(chunk)
    finishes = sum(
        1 for c in chunks for choice in c["choices"] if choice.get("finish_reason") is not None
    )
    if finishes != 1:
        _fail("chat.single_finish_reason", f"{finishes} chunks carry finish_reason")
    if include_usage and (not chunks or chunks[-1]["choices"] or not chunks[-1].get("usage")):
        _fail("chat.usage_last", "last chunk must have choices == [] and usage")


def validate_response(body: Any) -> None:
    _model("responses.response_schema", Response, body, "response")


def validate_responses_stream(raw: bytes) -> None:
    """Responses stream: SDK events, sequence_number strictly increasing from 0,
    exactly one terminal event. A trailing data: [DONE] is tolerated.
    """
    events = parse_sse(raw)
    if events and events[-1].data == "[DONE]":
        events = events[:-1]
    previous = -1
    terminals = 0
    for i, ev in enumerate(events):
        where = f"event {i}"
        data = _json("responses.json", ev.data, where)
        _event("responses.sdk_schema", _RESPONSES_EVENTS, data, where)
        if ev.event is not None and ev.event != data["type"]:
            _fail("responses.event_name", f"{where}: event {ev.event!r} carries {data['type']!r}")
        seq = data["sequence_number"]
        if (previous == -1 and seq != 0) or seq <= previous:
            _fail("responses.sequence_number", f"{where}: {seq} after {previous}")
        previous = seq
        terminals += data["type"] in _RESPONSES_TERMINAL
    if terminals != 1:
        _fail("responses.single_terminal", f"{terminals} terminal events")
