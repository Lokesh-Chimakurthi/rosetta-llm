"""Request logging contract: one line per request at info, stage events at debug."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from rosetta.config import LogLevel
from rosetta.observability import setup_logging
from tests.support.upstream import FakeUpstream, build_client

_CONFIG: dict[str, Any] = {
    "providers": {
        "chat": {
            "format": "openai_chat",
            "base_url": "https://chat.test/v1",
            "api_key": "sk-provider-secret",
            "models": [{"id": "m"}],
        },
    },
}
_LIFECYCLE_EVENTS = frozenset({"startup", "models_refreshed"})
_BASE_KEYS = {"event", "level", "timestamp", "request_id"}
_REQUEST_KEYS = _BASE_KEYS | {"method", "path", "status", "duration_ms"}
_ROUTED_KEYS = _REQUEST_KEYS | {"route", "provider", "model", "stream"}
_COMPLETION = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    yield
    setup_logging("info")


def _lines(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def _run(
    level: LogLevel,
    capsys: pytest.CaptureFixture[str],
    fake: FakeUpstream,
    method: str,
    path: str,
    **kwargs: Any,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (request_id, that request's lines, unrelated lines)."""
    setup_logging(level)
    with build_client(_CONFIG, fake) as client:
        capsys.readouterr()
        r = client.request(method, path, **kwargs)
        r.read()
    lines = _lines(capsys.readouterr().out)
    request_id = r.headers["x-request-id"]
    mine = [line for line in lines if line.get("request_id") == request_id]
    others = [line for line in lines if "request_id" not in line]
    return request_id, mine, others


@pytest.mark.parametrize(
    ("case", "expected_keys"),
    [
        ("passthrough", _ROUTED_KEYS),
        ("translated_stream", _ROUTED_KEYS | {"ttft_ms", "input_tokens", "output_tokens"}),
        ("not_found", _REQUEST_KEYS),
    ],
)
def test_info_emits_exactly_one_line_per_request(
    case: str, expected_keys: set[str], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeUpstream()
    if case == "translated_stream":
        fake.add_sse("chat", "/chat/completions", "text")
    else:
        fake.add_json("chat", "POST", "/chat/completions", 200, _COMPLETION)
    requests: dict[str, tuple[str, str, dict[str, Any]]] = {
        "passthrough": (
            "POST",
            "/v1/chat/completions",
            {"json": {"model": "chat/m", "messages": []}},
        ),
        "translated_stream": (
            "POST",
            "/v1/messages",
            {
                "json": {
                    "model": "chat/m",
                    "max_tokens": 8,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            },
        ),
        "not_found": ("GET", "/nope", {}),
    }
    method, path, kwargs = requests[case]

    _, mine, others = _run("info", capsys, fake, method, path, **kwargs)

    assert [line["event"] for line in mine] == ["request_completed"]
    assert set(mine[0]) == expected_keys
    assert {line["event"] for line in others} <= _LIFECYCLE_EVENTS
    if case == "translated_stream":
        assert (mine[0]["input_tokens"], mine[0]["output_tokens"]) == (9, 3)


def test_debug_emits_stage_events_with_redacted_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeUpstream()
    fake.add_sse("chat", "/chat/completions", "text")
    body = {
        "model": "chat/m",
        "max_tokens": 8,
        "stream": True,
        "messages": [{"role": "user", "content": "x" * 5000}],
    }
    secrets = {
        "authorization": "Bearer sk-client-secret",
        "x-api-key": "sk-other-secret",
        "cookie": "session=cookie-secret",
    }

    _, mine, _ = _run("debug", capsys, fake, "POST", "/v1/messages", json=body, headers=secrets)

    events = [line["event"] for line in mine]
    assert events == [
        "request_received",
        "upstream_request",
        "upstream_response",
        "stream_summary",
        "request_completed",
    ]
    received, upstream_request, upstream_response, summary, completed = mine
    assert all(received["headers"][name] == "***" for name in secrets)
    raw = json.dumps(mine)
    assert not any(
        secret in raw for secret in ("sk-client-secret", "sk-other-secret", "cookie-secret")
    )
    assert "sk-provider-secret" not in raw
    assert received["body_bytes"] == len(json.dumps(body, separators=(",", ":")))
    assert len(received["body_preview"]) == 2048
    assert upstream_request["provider"] == "chat"
    assert upstream_request["upstream_model"] == "m"
    assert "authorization" not in upstream_request["headers"]
    assert upstream_response["status"] == 200
    assert summary["chunks_in"] >= 1
    assert summary["chunks_out"] >= 1
    assert completed["inbound_format"] == "anthropic"
    assert completed["provider_format"] == "openai_chat"
    assert completed["upstream_status"] == 200


def test_unexpected_exception_still_emits_request_completed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeUpstream()  # no route registered: the transport raises AssertionError
    with pytest.raises(AssertionError):
        _run(
            "info",
            capsys,
            fake,
            "POST",
            "/v1/chat/completions",
            json={"model": "chat/m", "messages": []},
        )
    completed = [
        line
        for line in _lines(capsys.readouterr().out)
        if line.get("event") == "request_completed"
    ]
    assert len(completed) == 1
    assert (completed[0]["status"], completed[0]["error_type"]) == (500, "AssertionError")


@pytest.mark.parametrize(
    ("inbound", "reused"),
    [
        ("req_ABC-123.x", True),
        ("has space", False),
        ("x" * 129, False),
        ("inject\\nline", False),
    ],
)
def test_inbound_request_id_reused_only_when_safe(
    inbound: str, reused: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    request_id, mine, _ = _run(
        "info", capsys, FakeUpstream(), "GET", "/v1/models", headers={"x-request-id": inbound}
    )
    assert (request_id == inbound) is reused
    assert len(request_id) <= 128
    assert [line["event"] for line in mine] == ["request_completed"]


def test_startup_event_names_providers_without_urls_or_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging("info")
    with build_client(_CONFIG, FakeUpstream()):
        pass
    startup = [line for line in _lines(capsys.readouterr().out) if line["event"] == "startup"]
    assert len(startup) == 1
    assert startup[0]["providers"] == [{"key": "chat", "format": "openai_chat", "discover": False}]
    assert "chat.test" not in json.dumps(startup)
    assert "sk-provider-secret" not in json.dumps(startup)
