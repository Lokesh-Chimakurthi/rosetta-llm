"""End-to-end smoke tests via FastAPI TestClient with an in-memory upstream."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.support.contracts import parse_sse, validate_anthropic_stream
from tests.support.upstream import FakeUpstream, build_client, load_fixture

_CONFIG: dict[str, Any] = {
    "providers": {
        "abc": {
            "format": "openai_chat",
            "base_url": "https://upstream.test/v1",
            "api_key": "sk-up",
            "models": [{"id": "m1"}],
        },
        "anth": {
            "format": "anthropic",
            "base_url": "https://anth.test/v1",
            "api_key": "sk-up",
            "models": [{"id": "claude"}],
        },
    },
}

_CHAT_COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-up",
    "object": "chat.completion",
    "model": "m1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _parse_sse(raw: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Split a rendered Anthropic SSE byte stream into (event_name, payload) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.split(b"\n\n"):
        if not block.strip():
            continue
        name, data_str = "", ""
        for line in block.decode().split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].lstrip()
        events.append((name, json.loads(data_str)))
    return events


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def client(fake: FakeUpstream) -> Iterator[TestClient]:
    with build_client(_CONFIG, fake) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_static(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    ids = sorted(m["id"] for m in data["data"])
    assert ids == ["abc/m1", "anth/claude"]


def test_count_tokens_local(client: TestClient) -> None:
    body = {"messages": [{"role": "user", "content": "hello world"}]}
    r = client.post("/v1/messages/count_tokens", json=body)
    assert r.status_code == 200
    assert r.json()["input_tokens"] >= 1


def test_chat_completions_passthrough(client: TestClient, fake: FakeUpstream) -> None:
    fake.add_json("abc", "POST", "/chat/completions", 200, _CHAT_COMPLETION)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "abc/m1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json() == _CHAT_COMPLETION
    assert fake.requests[-1].json()["model"] == "m1"


def test_anthropic_to_chat_translation(client: TestClient, fake: FakeUpstream) -> None:
    """Anthropic /v1/messages → openai_chat provider should translate both ways."""
    upstream_resp = {
        **_CHAT_COMPLETION,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello back"},
                "finish_reason": "stop",
            }
        ],
    }
    fake.add_json("abc", "POST", "/chat/completions", 200, upstream_resp)
    r = client.post(
        "/v1/messages",
        json={
            "model": "abc/m1",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert [(q.provider_key, q.path) for q in fake.requests] == [("abc", "/chat/completions")]
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "hello back"
    assert body["stop_reason"] == "end_turn"


def test_unknown_provider_400(client: TestClient) -> None:
    r = client.post("/v1/chat/completions", json={"model": "ghost/missing", "messages": []})
    assert r.status_code == 400
    assert "error" in r.json()


def test_upstream_error_passthrough_format(client: TestClient, fake: FakeUpstream) -> None:
    """Upstream 4xx should be wrapped in inbound-format error envelope when translating."""
    fake.add_json("abc", "POST", "/chat/completions", 429, {"error": {"message": "rate limited"}})
    r = client.post(
        "/v1/messages",
        json={
            "model": "abc/m1",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 429
    body = r.json()
    # Must be Anthropic-shaped because /v1/messages was hit.
    assert body["type"] == "error"
    assert body["error"]["message"] == "rate limited"


def test_auth_required_when_configured(fake: FakeUpstream) -> None:
    config = {**_CONFIG, "proxy": {"api_keys": ["sk-proxy"]}}
    fake.add_json("abc", "POST", "/chat/completions", 200, _CHAT_COMPLETION)
    with build_client(config, fake) as c:
        r = c.post("/v1/chat/completions", json={"model": "abc/m1", "messages": []})
        assert r.status_code == 401
        r = c.post(
            "/v1/chat/completions",
            json={"model": "abc/m1", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-proxy"},
        )
        assert r.status_code == 200


def test_chat_stream_passthrough(client: TestClient, fake: FakeUpstream) -> None:
    upstream_sse = (
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    fake.add_sse("abc", "/chat/completions", upstream_sse)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "abc/m1",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        chunks = b"".join(r.iter_bytes())
    assert chunks == upstream_sse


def test_anthropic_stream_translation_carries_usage(
    client: TestClient, fake: FakeUpstream
) -> None:
    """Anthropic-inbound streaming translation: one usage-bearing terminal message_delta."""
    upstream_sse = (
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":5,"cache_write_tokens":0}}}\n\n'
        b"data: [DONE]\n\n"
    )
    fake.add_sse("abc", "/chat/completions", upstream_sse)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "abc/m1",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        body = b"".join(r.iter_bytes())

    # The upstream saw the ask for usage.
    sent = fake.requests[-1].json()
    assert sent["stream_options"] == {"include_usage": True}

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert names.count("message_delta") == 1
    delta_idx = names.index("message_delta")
    assert names[delta_idx + 1] == "message_stop"

    usage = events[delta_idx][1]["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["cache_read_input_tokens"] == 5

    start = next(data for name, data in events if name == "message_start")
    assert start["message"]["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def test_anthropic_stream_translation_plain_usage_no_cache(
    client: TestClient, fake: FakeUpstream
) -> None:
    """Plain case: no cached tokens — input_tokens passes through unchanged."""
    upstream_sse = (
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","model":"m1","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
        b"data: [DONE]\n\n"
    )
    fake.add_sse("abc", "/chat/completions", upstream_sse)
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "abc/m1",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        body = b"".join(r.iter_bytes())

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert names.count("message_delta") == 1
    delta_idx = names.index("message_delta")
    assert names[delta_idx + 1] == "message_stop"

    usage = events[delta_idx][1]["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3
    assert usage.get("cache_read_input_tokens", 0) == 0


def test_env_resolved_custom_header_reaches_upstream(
    fake: FakeUpstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header declared via extra_headers_env must be sent upstream after env resolution."""
    monkeypatch.setenv("ABC_SECRET_HEADER", "secret-from-env")
    config = {
        "providers": {
            "abc": {
                **_CONFIG["providers"]["abc"],
                "extra_headers": {"X-Literal-Header": "literal-val"},
                "extra_headers_env": {"X-Secret-Header": "ABC_SECRET_HEADER"},
            },
        },
    }
    fake.add_json("abc", "POST", "/chat/completions", 200, _CHAT_COMPLETION)
    with build_client(config, fake) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "abc/m1", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    sent_headers = fake.requests[-1].headers
    assert sent_headers["x-secret-header"] == "secret-from-env"
    assert sent_headers["x-literal-header"] == "literal-val"


# Claude Code's auto mode review request, per
# https://docs.litellm.ai/blog/claude-code-server-side-auto-mode.
_REVIEW_BETA = "dangerous-tool-use-2026-09-03"
_SAFEGUARDS: list[dict[str, Any]] = [
    {"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}
]
_TOOL_USE_ID = "toolu_01V9Z5KXn3SU71Fzr5cquHLi"


def _review_request(model: str, *, stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "safeguards": _SAFEGUARDS,
        "messages": [{"role": "user", "content": "Run: echo hello"}],
    }


def test_auto_mode_review_passthrough_keeps_request_and_verdicts(
    client: TestClient, fake: FakeUpstream
) -> None:
    """Anthropic passthrough: safeguards and its beta reach the upstream; verdicts reach the client."""
    upstream_resp = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [
            {"type": "tool_use", "id": _TOOL_USE_ID, "name": "Bash", "input": {"command": "ls"}}
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 5},
        "safeguard_results": [
            {
                "type": "dangerous_tool_use",
                "status": {
                    "type": "available",
                    "tool_uses": {_TOOL_USE_ID: {"type": "evaluated", "outcome": "not_flagged"}},
                },
            }
        ],
    }
    fake.add_json("anth", "POST", "/messages", 200, upstream_resp)

    r = client.post(
        "/v1/messages",
        json=_review_request("anth/claude"),
        headers={"anthropic-beta": f"oauth-2025-04-20,{_REVIEW_BETA}"},
    )

    sent = fake.requests[-1]
    assert sent.json()["safeguards"] == _SAFEGUARDS
    assert sent.headers["anthropic-beta"] == f"oauth-2025-04-20,{_REVIEW_BETA}"
    assert r.status_code == 200
    assert r.json() == upstream_resp


def test_auto_mode_review_passthrough_stream_relays_final_delta_verbatim(
    client: TestClient, fake: FakeUpstream
) -> None:
    """The verdict rides on the last message_delta; passthrough must not drop or rewrite it."""
    fake.add_sse("anth", "/messages", "tool_call_safeguards", chunk_size=97)
    with client.stream(
        "POST",
        "/v1/messages",
        json=_review_request("anth/claude", stream=True),
        headers={"anthropic-beta": _REVIEW_BETA},
    ) as r:
        body = b"".join(r.iter_bytes())

    assert fake.requests[-1].json()["safeguards"] == _SAFEGUARDS
    assert body == load_fixture("anthropic", "tool_call_safeguards")
    validate_anthropic_stream(body)
    final_delta = [json.loads(ev.data) for ev in parse_sse(body) if ev.event == "message_delta"][
        -1
    ]
    verdicts = final_delta["delta"]["safeguard_results"][0]["status"]["tool_uses"]
    assert verdicts == {
        "toolu_01T1x1fJ34qAmk2tNTrN7Up6": {"type": "evaluated", "outcome": "not_flagged"}
    }


def test_auto_mode_review_dropped_on_translation_so_claude_code_falls_back(
    client: TestClient, fake: FakeUpstream
) -> None:
    """A non-Anthropic upstream cannot review: the field is withheld and no verdict is invented."""
    fake.add_json("abc", "POST", "/chat/completions", 200, _CHAT_COMPLETION)

    r = client.post(
        "/v1/messages",
        json=_review_request("claude-code/abc/m1"),
        headers={"anthropic-beta": _REVIEW_BETA},
    )

    assert "safeguards" not in fake.requests[-1].json()
    assert r.status_code == 200
    assert "safeguard_results" not in r.json()


def test_auto_mode_review_dropped_on_stream_translation_and_stream_completes(
    client: TestClient, fake: FakeUpstream
) -> None:
    """Claude Code reads a completed stream without verdicts as "not reviewed" and falls back."""
    fake.add_sse("abc", "/chat/completions", load_fixture("openai_chat", "tool_call"))
    with client.stream(
        "POST",
        "/v1/messages",
        json=_review_request("claude-code/abc/m1", stream=True),
        headers={"anthropic-beta": _REVIEW_BETA},
    ) as r:
        body = b"".join(r.iter_bytes())

    assert "safeguards" not in fake.requests[-1].json()
    validate_anthropic_stream(body)
    assert b"safeguard_results" not in body


@pytest.mark.parametrize(
    ("fmt", "path", "fixture", "stop_reason"),
    [
        (fmt, path, name, stop)
        for fmt, path in (("openai_chat", "/chat/completions"), ("openai_responses", "/responses"))
        for name, stop in (
            ("text", "end_turn"),
            ("tool_call", "tool_use"),
            ("reasoning", "end_turn"),
        )
    ],
)
def test_translated_anthropic_stream_is_valid_for_every_upstream_fixture(
    fake: FakeUpstream, fmt: str, path: str, fixture: str, stop_reason: str
) -> None:
    """Dense block indices, one start and stop each, and the stop reason Claude Code acts on."""
    config = {
        "providers": {
            "up": {
                "format": fmt,
                "base_url": "https://up.test/v1",
                "api_key": "sk-up",
                "models": [{"id": "m"}],
            }
        }
    }
    fake.add_sse("up", path, fixture)
    with (
        build_client(config, fake) as c,
        c.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "up/m",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r,
    ):
        body = b"".join(r.iter_bytes())
    validate_anthropic_stream(body)
    final_delta = [json.loads(ev.data) for ev in parse_sse(body) if ev.event == "message_delta"][
        -1
    ]
    assert final_delta["delta"]["stop_reason"] == stop_reason


def test_responses_server_state_to_other_format_is_a_400_with_the_remedy(
    client: TestClient, fake: FakeUpstream
) -> None:
    r = client.post(
        "/v1/responses",
        json={"model": "abc/m1", "input": "and then?", "previous_response_id": "resp_1"},
    )
    assert r.status_code == 400
    assert "full conversation in 'input'" in r.json()["error"]["message"]
    assert fake.requests == []
