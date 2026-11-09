"""Attach `safeguard_results` to an Anthropic response for Claude Code's auto mode.

Wire shape, as Anthropic's API returns it and Claude Code 2.1.319 accepts it
(verified live; Anthropic does not document it, LiteLLM's
`litellm/llms/anthropic/messages/safeguards.py` produces the same):
`[{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses":
{<tool_use id>: <verdict>}}}]`, at the top level of a message, or in the
`delta` of the final `message_delta` event of a stream. A tool call with no
verdict is denied by Claude Code, so a stream cut before its final
`message_delta` denies every call in it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import orjson

from rosetta.auto_mode.classifier import Classifier, Session, ToolCall, Unavailable, Verdict
from rosetta.observability import ms_since, request_log
from rosetta.stream_codecs.anthropic import PING_EVENT, PING_INTERVAL_S, sse_event

_REVIEW_TYPE = "dangerous_tool_use"
# Pairs with the `safeguards` field; dropped with it when Rosetta reviews instead.
_REVIEW_BETA_PREFIX = "dangerous-tool-use-"


def requested_session(payload: Mapping[str, Any]) -> Session | None:
    """The session to review, when the Anthropic request asks for auto mode review."""
    safeguards = payload.get("safeguards")
    if not isinstance(safeguards, list):
        return None
    for entry in safeguards:
        if isinstance(entry, dict) and entry.get("type") == _REVIEW_TYPE:
            messages = payload.get("messages")
            return Session(
                entry.get("classifier_context"), messages if isinstance(messages, list) else []
            )
    return None


def without_review_request(
    body: dict[str, Any], headers: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """The request minus `safeguards` and its beta, for an upstream Rosetta reviews for."""
    stripped = {k: v for k, v in body.items() if k != "safeguards"}
    out = dict(headers)
    if "anthropic-beta" in out:
        betas = [b.strip() for b in out["anthropic-beta"].split(",")]
        kept = [b for b in betas if b and not b.startswith(_REVIEW_BETA_PREFIX)]
        if kept:
            out["anthropic-beta"] = ",".join(kept)
        else:
            del out["anthropic-beta"]
    return stripped, out


@dataclass(frozen=True, slots=True)
class _Truncated:
    id: str


type _Pending = ToolCall | _Truncated


class Review:
    """Reviews the tool calls of one response; one instance per request."""

    def __init__(self, classifier: Classifier, session: Session) -> None:
        self._classifier = classifier
        self._session = session

    async def stamp_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """`message` with `safeguard_results` for its tool_use blocks."""
        content = message.get("content")
        blocks: list[dict[str, Any]] = (
            [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
        )
        cut_at = len(blocks) - 1 if message.get("stop_reason") == "max_tokens" else None
        pending: list[_Pending] = [
            _Truncated(str(b.get("id")))
            if i == cut_at
            else ToolCall(str(b.get("id")), str(b.get("name")), b.get("input"))
            for i, b in enumerate(blocks)
            if b.get("type") == "tool_use"
        ]
        return {**message, "safeguard_results": await self._results(pending)}

    async def stamp_stream(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Relay an Anthropic SSE stream, adding `safeguard_results` to its final `message_delta`.

        Every other event passes through byte for byte. The final
        `message_delta` is held until the classifier answers, with a `ping`
        every PING_INTERVAL_S meanwhile; the classifier call is cancelled if
        the consumer stops reading.
        """
        tracker = _StreamTracker()
        held: dict[str, Any] | None = None
        buffer = b""
        async for chunk in chunks:
            buffer += chunk
            while (end := buffer.find(b"\n\n")) >= 0:
                raw, buffer = buffer[: end + 2], buffer[end + 2 :]
                data = _event_data(raw)
                kind = data.get("type") if data is not None else None
                if data is not None and kind == "message_delta":
                    if held is not None:
                        yield sse_event("message_delta", held)
                    held = data
                    continue
                if kind == "message_stop" and held is not None:
                    async for out in self._stamped_delta(held, tracker):
                        yield out
                    held = None
                elif data is not None:
                    tracker.observe(data)
                yield raw
        if buffer:
            yield buffer
        if held is not None:
            async for out in self._stamped_delta(held, tracker):
                yield out

    async def _stamped_delta(
        self, delta_event: dict[str, Any], tracker: _StreamTracker
    ) -> AsyncIterator[bytes]:
        delta = delta_event.get("delta")
        stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else None
        task = asyncio.create_task(self._results(tracker.pending(stop_reason == "max_tokens")))
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=PING_INTERVAL_S)
                if not done:
                    yield PING_EVENT
            results = task.result()
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        stamped = {**(delta if isinstance(delta, dict) else {}), "safeguard_results": results}
        yield sse_event("message_delta", {**delta_event, "delta": stamped})

    async def _results(self, pending: Sequence[_Pending]) -> list[dict[str, Any]]:
        rec = request_log()
        started = time.perf_counter()
        calls = [p for p in pending if isinstance(p, ToolCall)]
        classified = await self._classifier.classify(self._session, calls)
        verdicts: dict[str, Verdict] = {
            p.id: Unavailable("truncated")
            if isinstance(p, _Truncated)
            else classified.get(p.id, Unavailable("error"))
            for p in pending
        }
        if pending:
            rec.auto_mode_review_ms = ms_since(started)
            rec.auto_mode_review = (
                "unavailable"
                if any(isinstance(v, Unavailable) for v in verdicts.values())
                else "reviewed"
            )
            rec.auto_mode_flagged = sum(
                1 for v in verdicts.values() if not isinstance(v, Unavailable) and v.flagged
            )
        else:
            rec.auto_mode_review = "no_tool_calls"
        tool_uses = {tool_id: v.wire() for tool_id, v in verdicts.items()}
        return [{"type": _REVIEW_TYPE, "status": {"type": "available", "tool_uses": tool_uses}}]


@dataclass(slots=True)
class _StreamedToolUse:
    id: str
    name: str
    start_input: Any
    partial_json: list[str] = field(default_factory=list)
    stopped: bool = False

    def pending(self, cut_off: bool) -> _Pending:
        if cut_off or not self.stopped:
            return _Truncated(self.id)
        if not self.partial_json:
            return ToolCall(self.id, self.name, self.start_input)
        try:
            return ToolCall(self.id, self.name, orjson.loads("".join(self.partial_json)))
        except orjson.JSONDecodeError:
            return _Truncated(self.id)


class _StreamTracker:
    """Rebuilds the tool_use blocks of one streamed message from its events."""

    def __init__(self) -> None:
        self._tool_uses: dict[int, _StreamedToolUse] = {}
        self._last_index: int | None = None

    def observe(self, data: Mapping[str, Any]) -> None:
        index = data.get("index")
        if not isinstance(index, int):
            return
        kind = data.get("type")
        if kind == "content_block_start":
            self._last_index = index
            block = data.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                self._tool_uses[index] = _StreamedToolUse(
                    str(block.get("id")), str(block.get("name")), block.get("input")
                )
        elif (tool_use := self._tool_uses.get(index)) is None:
            return
        elif kind == "content_block_delta":
            delta = data.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                tool_use.partial_json.append(str(delta.get("partial_json", "")))
        elif kind == "content_block_stop":
            tool_use.stopped = True

    def pending(self, hit_max_tokens: bool) -> list[_Pending]:
        cut_at = self._last_index if hit_max_tokens else None
        return [t.pending(i == cut_at) for i, t in sorted(self._tool_uses.items())]


def _event_data(raw: bytes) -> dict[str, Any] | None:
    for line in raw.split(b"\n"):
        if line.startswith(b"data:"):
            try:
                data = orjson.loads(line[5:])
            except orjson.JSONDecodeError:
                return None
            return data if isinstance(data, dict) else None
    return None
