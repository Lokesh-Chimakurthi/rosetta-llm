"""Request logging contract.

Info: request_started per request; request_completed only with log_completed.
Debug: upstream_request/upstream_response; headers and bodies only behind
log_headers/log_bodies.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx2
import pytest

from rosetta.config import LogLevel
from rosetta.observability import BODY_LOG_LIMIT, setup_logging
from tests.support.upstream import FakeUpstream, build_client

_PROVIDER_KEY = "sk-provider-secret"
_PROVIDERS: dict[str, Any] = {
    "chat": {
        "format": "openai_chat",
        "base_url": "https://chat.test/v1",
        "api_key": _PROVIDER_KEY,
        "models": [{"id": "m"}],
    },
}
_LIFECYCLE_EVENTS = frozenset({"startup", "models_refreshed"})
_BASE = {"ts", "level", "event", "request_id", "path"}
_ROUTED = _BASE | {"model", "route", "stream"}
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
_CHAT: tuple[str, str, dict[str, Any]] = (
    "POST",
    "/v1/chat/completions",
    {"json": {"model": "chat/m", "messages": []}},
)
_MESSAGES = {"model": "chat/m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
_MESSAGES_STREAM = ("POST", "/v1/messages", {"json": {**_MESSAGES, "stream": True}})
_CLIENT_SECRETS = {
    "authorization": "Bearer sk-client-secret",
    "x-api-key": "sk-other-secret",
    "cookie": "session=cookie-secret",
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
    request: tuple[str, str, dict[str, Any]],
    **config: Any,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (request_id, that request's lines, lines outside any request)."""
    setup_logging(level)
    method, path, kwargs = request
    with build_client({"providers": _PROVIDERS, **config}, fake) as client:
        capsys.readouterr()
        r = client.request(method, path, **kwargs)
        r.read()
    lines = _lines(capsys.readouterr().out)
    request_id = r.headers["x-request-id"]
    mine = [line for line in lines if line.get("request_id") == request_id]
    others = [line for line in lines if "request_id" not in line]
    return request_id, mine, others


def _fake(case: str) -> FakeUpstream:
    fake = FakeUpstream()
    if case == "stream":
        fake.add_sse("chat", "/chat/completions", "text")
    elif case == "rate_limited":
        fake.add_json("chat", "POST", "/chat/completions", 429, {"error": {"message": "slow"}})
    elif case == "timeout":
        fake.add_error("chat", "POST", "/chat/completions", httpx2.ReadTimeout("chat.test"))
    else:
        fake.add_json("chat", "POST", "/chat/completions", 200, _COMPLETION)
    return fake


@pytest.mark.parametrize(
    ("case", "request_", "expected_keys"),
    [
        ("ok", _CHAT, _ROUTED),
        ("stream", _MESSAGES_STREAM, _ROUTED),
        ("ok", ("GET", "/nope", {}), _BASE),
        (
            "ok",
            ("POST", "/v1/chat/completions", {"json": {"model": "ghost/x"}}),
            _BASE | {"model"},
        ),
    ],
    ids=["passthrough", "translated-stream", "not-found", "unknown-provider"],
)
def test_info_logs_only_request_started_by_default(
    case: str,
    request_: tuple[str, str, dict[str, Any]],
    expected_keys: set[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, mine, others = _run("info", capsys, _fake(case), request_)
    assert [line["event"] for line in mine] == ["request_started"]
    assert set(mine[0]) == expected_keys
    assert mine[0]["level"] == "info"
    assert {line["event"] for line in others} <= _LIFECYCLE_EVENTS


@pytest.mark.parametrize(
    ("case", "request_", "level", "extra_keys", "values"),
    [
        ("ok", _CHAT, "info", set(), {"status": 200, "route": "passthrough"}),
        (
            "stream",
            _MESSAGES_STREAM,
            "info",
            {"ttft_ms", "input_tokens", "output_tokens", "stop_reason"},
            {"status": 200, "input_tokens": 9, "output_tokens": 3, "stop_reason": "end_turn"},
        ),
        (
            "rate_limited",
            ("POST", "/v1/messages", {"json": _MESSAGES}),
            "warning",
            {"error", "upstream_status"},
            {"status": 429, "error": "upstream_status", "upstream_status": 429},
        ),
        (
            "timeout",
            ("POST", "/v1/messages", {"json": _MESSAGES}),
            "error",
            {"error"},
            {"status": 502, "error": "upstream_timeout"},
        ),
    ],
    ids=["passthrough", "translated-stream", "upstream-429", "upstream-timeout"],
)
def test_log_completed_adds_one_completion_line_levelled_by_status(
    case: str,
    request_: tuple[str, str, dict[str, Any]],
    level: str,
    extra_keys: set[str],
    values: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, mine, _ = _run("info", capsys, _fake(case), request_, log_completed=True)
    assert [line["event"] for line in mine] == ["request_started", "request_completed"]
    completed = mine[1]
    assert completed["level"] == level
    assert set(completed) == _ROUTED | {"status", "duration_ms"} | extra_keys
    assert values.items() <= completed.items()


def test_debug_adds_upstream_events_without_headers_or_bodies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    method, path, kwargs = _MESSAGES_STREAM
    _, mine, _ = _run(
        "debug", capsys, _fake("stream"), (method, path, {**kwargs, "headers": _CLIENT_SECRETS})
    )
    assert [line["event"] for line in mine] == [
        "request_started",
        "upstream_request",
        "upstream_response",
    ]
    _, upstream_request, upstream_response = mine
    assert upstream_request["upstream_model"] == "m"
    assert upstream_response["status"] == 200
    assert isinstance(upstream_response["ttfb_ms"], int)
    assert not any({"headers", "body"} & set(line) for line in mine)


def test_debug_log_headers_adds_redacted_headers(capsys: pytest.CaptureFixture[str]) -> None:
    method, path, kwargs = _MESSAGES_STREAM
    _, mine, _ = _run(
        "debug",
        capsys,
        _fake("stream"),
        (method, path, {**kwargs, "headers": _CLIENT_SECRETS}),
        log_headers=True,
    )
    started, upstream_request, upstream_response = mine
    assert all(started["headers"][name] == "***" for name in _CLIENT_SECRETS)
    assert upstream_request["headers"]["authorization"] == "***"
    assert upstream_response["headers"]["content-type"] == "text/event-stream"
    raw = json.dumps(mine)
    for secret in ("sk-client-secret", "sk-other-secret", "cookie-secret", _PROVIDER_KEY):
        assert secret not in raw
    assert not any("body" in line for line in mine)


def test_debug_log_bodies_adds_capped_bodies(capsys: pytest.CaptureFixture[str]) -> None:
    big = {**_MESSAGES, "messages": [{"role": "user", "content": "x" * (2 * BODY_LOG_LIMIT)}]}
    _, mine, _ = _run(
        "debug",
        capsys,
        _fake("ok"),
        ("POST", "/v1/messages", {"json": big}),
        log_bodies=True,
    )
    started, upstream_request, upstream_response = mine
    assert len(started["body"]) == BODY_LOG_LIMIT
    assert started["body_truncated"] is True
    assert started["body_bytes"] == len(json.dumps(big, separators=(",", ":")))
    assert upstream_request["body_truncated"] is True
    assert json.loads(upstream_response["body"]) == _COMPLETION
    assert "body_truncated" not in upstream_response
    assert not any("headers" in line for line in mine)


def test_header_and_body_flags_have_no_effect_at_info(capsys: pytest.CaptureFixture[str]) -> None:
    method, path, kwargs = _CHAT
    _, mine, _ = _run(
        "info",
        capsys,
        _fake("ok"),
        (method, path, {**kwargs, "headers": _CLIENT_SECRETS}),
        log_headers=True,
        log_bodies=True,
    )
    assert [line["event"] for line in mine] == ["request_started"]
    assert not {"headers", "body"} & set(mine[0])


def test_translation_failure_logs_one_error_line_without_traceback_at_info(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeUpstream()
    fake.add_sse("chat", "/chat/completions", b"not json")
    _, mine, _ = _run("info", capsys, fake, ("POST", "/v1/messages", {"json": _MESSAGES}))
    assert [line["event"] for line in mine] == ["request_started", "translation_failed"]
    failed = mine[1]
    assert (failed["level"], failed["direction"]) == ("error", "response")
    assert failed["error"].startswith("JSONDecodeError")
    assert "exception" not in failed


def test_unexpected_exception_still_logs_completion(capsys: pytest.CaptureFixture[str]) -> None:
    # No route registered: the fake transport raises AssertionError inside the app.
    with pytest.raises(AssertionError):
        _run("info", capsys, FakeUpstream(), _CHAT, log_completed=True)
    lines = _lines(capsys.readouterr().out)
    completed = [line for line in lines if line["event"] == "request_completed"]
    assert [line["event"] for line in lines if "request_id" in line] == [
        "request_started",
        "request_completed",
    ]
    assert (completed[0]["status"], completed[0]["error"]) == (500, "AssertionError")
    assert completed[0]["level"] == "error"


@pytest.mark.parametrize(
    ("inbound", "reused"),
    [("req_ABC-123.x", True), ("has space", False), ("x" * 129, False), ("a\\b", False)],
)
def test_inbound_request_id_reused_only_when_safe(
    inbound: str, reused: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    request_id, mine, _ = _run(
        "info",
        capsys,
        FakeUpstream(),
        ("GET", "/v1/models", {"headers": {"x-request-id": inbound}}),
    )
    assert (request_id == inbound) is reused
    if not reused:
        assert len(request_id) == 16
    assert [line["event"] for line in mine] == ["request_started"]


def test_health_is_never_logged(capsys: pytest.CaptureFixture[str]) -> None:
    _, mine, _ = _run("debug", capsys, FakeUpstream(), ("GET", "/health", {}), log_completed=True)
    assert mine == []


def test_startup_event_names_providers_without_urls_or_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging("info")
    with build_client({"providers": _PROVIDERS}, FakeUpstream()):
        pass
    startup = [line for line in _lines(capsys.readouterr().out) if line["event"] == "startup"]
    assert len(startup) == 1
    assert startup[0]["providers"] == [{"key": "chat", "format": "openai_chat", "discover": False}]
    assert "chat.test" not in json.dumps(startup)
    assert _PROVIDER_KEY not in json.dumps(startup)


def test_debug_completion_adds_detail_fields(capsys: pytest.CaptureFixture[str]) -> None:
    method, path, kwargs = _MESSAGES_STREAM
    session = {"x-claude-code-session-id": "sess-1"}
    _, mine, _ = _run(
        "debug",
        capsys,
        _fake("stream"),
        (method, path, {**kwargs, "headers": session}),
        log_completed=True,
    )
    completed = mine[-1]
    assert completed["event"] == "request_completed"
    assert {
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "upstream_stop_reason",
    } <= set(completed)
    assert (completed["upstream_status"], completed["session_id"]) == (200, "sess-1")


def test_debug_completion_names_fields_dropped_by_translation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    method, path, kwargs = _MESSAGES_STREAM
    extras = {
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        "safeguards": [{"secret": "sk-in-body"}],
    }
    _, mine, _ = _run(
        "debug",
        capsys,
        _fake("stream"),
        (method, path, {"json": {**kwargs["json"], **extras}}),
        log_completed=True,
    )
    completed = mine[-1]
    assert completed["dropped_fields"] == ["context_management", "safeguards"]
    assert "sk-in-body" not in json.dumps(mine)
