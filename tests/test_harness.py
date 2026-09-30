"""The harness later plans test against: validators, fixtures and FakeUpstream."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rosetta.config import Config, Format
from rosetta.upstream import UpstreamClient
from tests.support.contracts import (
    validate_anthropic_message,
    validate_anthropic_stream,
    validate_chat_completion,
    validate_chat_stream,
    validate_response,
    validate_responses_stream,
)
from tests.support.upstream import FakeUpstream, build_client, load_fixture

type StreamValidator = Callable[[bytes], None]


def _chat(include_usage: bool) -> StreamValidator:
    return lambda raw: validate_chat_stream(raw, include_usage=include_usage)


_STREAM_VALIDATORS: dict[Format, StreamValidator] = {
    "anthropic": validate_anthropic_stream,
    "openai_chat": _chat(include_usage=False),
    "openai_responses": validate_responses_stream,
}


def _replace(fmt: Format, name: str, old: str, new: str, count: int = 1) -> bytes:
    raw = load_fixture(fmt, name).decode()
    assert raw.count(old) >= count, f"mutation anchor {old!r} missing from {fmt}/{name}"
    return raw.replace(old, new, count).encode()


def _drop_line_block(fmt: Format, name: str, needle: str) -> bytes:
    blocks = load_fixture(fmt, name).decode().split("\n\n")
    kept = [b for b in blocks if needle not in b]
    assert len(kept) == len(blocks) - 1, f"{needle!r} must match exactly one event"
    return "\n\n".join(kept).encode()


@pytest.mark.parametrize(
    ("fmt", "name", "validator"),
    [
        ("anthropic", "text", validate_anthropic_stream),
        ("anthropic", "tool_call", validate_anthropic_stream),
        ("anthropic", "reasoning", validate_anthropic_stream),
        ("anthropic", "error_midstream", validate_anthropic_stream),
        ("openai_chat", "text", _chat(include_usage=True)),
        ("openai_chat", "tool_call", _chat(include_usage=False)),
        ("openai_chat", "reasoning", _chat(include_usage=False)),
        ("openai_responses", "text", validate_responses_stream),
        ("openai_responses", "tool_call", validate_responses_stream),
        ("openai_responses", "reasoning", validate_responses_stream),
    ],
)
def test_harness_valid_fixtures_pass(fmt: Format, name: str, validator: StreamValidator) -> None:
    raw = load_fixture(fmt, name)
    assert raw.startswith(b": "), "fixtures open with a comment naming their source"
    assert b"https://" in raw.split(b"\n\n", 1)[0]
    validator(raw)


_ANTHROPIC_TEXT_STOP = (
    'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'
)


@pytest.mark.parametrize(
    ("raw", "validator", "rule"),
    [
        pytest.param(
            _replace("anthropic", "text", _ANTHROPIC_TEXT_STOP, _ANTHROPIC_TEXT_STOP * 2),
            validate_anthropic_stream,
            "anthropic.duplicate_block_stop",
            id="anthropic-duplicate-stop",
        ),
        pytest.param(
            _replace(
                "anthropic",
                "tool_call",
                '"content_block_start","index":1',
                '"content_block_start","index":0',
            ),
            validate_anthropic_stream,
            "anthropic.block_index_dense",
            id="anthropic-reused-index",
        ),
        pytest.param(
            _replace(
                "anthropic",
                "tool_call",
                '"content_block_delta","index":1',
                '"content_block_delta","index":2',
            ),
            validate_anthropic_stream,
            "anthropic.block_not_open",
            id="anthropic-delta-for-unstarted-block",
        ),
        pytest.param(
            _drop_line_block("anthropic", "text", "message_stop"),
            validate_anthropic_stream,
            "anthropic.terminal",
            id="anthropic-missing-terminal",
        ),
        pytest.param(
            _replace("anthropic", "text", '"stop_reason": "end_turn"', '"stop_reason": null'),
            validate_anthropic_stream,
            "anthropic.stop_reason_before_stop",
            id="anthropic-no-stop-reason",
        ),
        pytest.param(
            _drop_line_block("anthropic", "text", "event: message_start"),
            validate_anthropic_stream,
            "anthropic.message_start_first",
            id="anthropic-missing-start",
        ),
        pytest.param(
            _drop_line_block("openai_chat", "text", "[DONE]"),
            _chat(include_usage=True),
            "chat.done_last",
            id="chat-missing-done",
        ),
        pytest.param(
            load_fixture("openai_chat", "error_midstream"),
            _chat(include_usage=False),
            "chat.done_last",
            id="chat-error-midstream-has-no-terminal",
        ),
        pytest.param(
            _replace(
                "openai_chat", "tool_call", '"finish_reason":null}]}', '"finish_reason":"stop"}]}'
            ),
            _chat(include_usage=False),
            "chat.single_finish_reason",
            id="chat-two-finish-reasons",
        ),
        pytest.param(
            load_fixture("openai_chat", "tool_call"),
            _chat(include_usage=True),
            "chat.usage_last",
            id="chat-missing-usage-chunk",
        ),
        pytest.param(
            _replace("openai_responses", "text", '"sequence_number":3', '"sequence_number":1'),
            validate_responses_stream,
            "responses.sequence_number",
            id="responses-out-of-order-sequence",
        ),
        pytest.param(
            load_fixture("openai_responses", "error_midstream"),
            validate_responses_stream,
            "responses.single_terminal",
            id="responses-missing-terminal",
        ),
        pytest.param(
            _replace("openai_responses", "text", '"delta":"Hi",', ""),
            validate_responses_stream,
            "responses.sdk_schema",
            id="responses-delta-without-delta",
        ),
    ],
)
def test_harness_invalid_streams_fail_their_rule(
    raw: bytes, validator: StreamValidator, rule: str
) -> None:
    with pytest.raises(AssertionError, match=rf"^\[{rule}\]"):
        validator(raw)


_MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1694268190,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
}
_RESPONSE: dict[str, Any] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1741290958,
    "model": "gpt-4.1",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


@pytest.mark.parametrize(
    ("body", "validator", "missing", "rule"),
    [
        (_MESSAGE, validate_anthropic_message, "usage", "anthropic.message_schema"),
        (_COMPLETION, validate_chat_completion, "created", "chat.completion_schema"),
        (_RESPONSE, validate_response, "output", "responses.response_schema"),
    ],
)
def test_harness_body_validators_accept_good_and_reject_missing_field(
    body: dict[str, Any], validator: Callable[[Any], None], missing: str, rule: str
) -> None:
    validator(body)
    with pytest.raises(AssertionError, match=rf"^\[{rule}\]"):
        validator({k: v for k, v in body.items() if k != missing})


_CONFIG: dict[str, Any] = {
    "providers": {
        "chat": {
            "format": "openai_chat",
            "base_url": "https://chat.test/v1",
            "api_key": "k",
            "models": [{"id": "m"}],
        },
        "anth": {
            "format": "anthropic",
            "base_url": "https://anth.test/v1",
            "api_key": "k",
            "models": [{"id": "claude"}],
        },
    },
}


async def test_harness_fake_upstream_chunks_delays_and_counts_streams() -> None:
    fake = FakeUpstream()
    cfg = Config.model_validate(_CONFIG)
    fake.bind(cfg)
    fake.add_sse("anth", "/messages", "text", chunk_size=100, delay_s=0.01)
    upstream = UpstreamClient(cfg, transport=fake)
    await upstream.start()
    try:
        started = time.monotonic()
        async with upstream.stream("anth", "/messages", b"{}", {}) as stream:
            assert fake.streams_opened == 1
            assert fake.streams_closed == 0
            chunks = [c async for c in stream.iter_bytes()]
        elapsed = time.monotonic() - started
    finally:
        await upstream.close()
    expected = load_fixture("anthropic", "text")
    assert b"".join(chunks) == expected
    assert all(len(c) == 100 for c in chunks[:-1])
    assert len(chunks) == -(-len(expected) // 100)
    assert elapsed >= 0.01 * len(chunks)
    assert fake.streams_closed == 1
    assert [(r.provider_key, r.method, r.path) for r in fake.requests] == [
        ("anth", "POST", "/messages")
    ]


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(chunk_size=st.integers(min_value=1, max_value=512))
def test_harness_passthrough_stream_is_verbatim_for_any_upstream_chunking(
    chunk_size: int,
) -> None:
    fake = FakeUpstream()
    fake.add_sse("chat", "/chat/completions", "text", chunk_size=chunk_size)
    with (
        build_client(_CONFIG, fake) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "chat/m", "stream": True, "messages": []},
        ) as r,
    ):
        body = b"".join(r.iter_bytes())
    assert body == load_fixture("openai_chat", "text")
    assert fake.streams_opened == fake.streams_closed == 1
