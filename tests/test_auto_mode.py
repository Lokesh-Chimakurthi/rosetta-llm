"""Auto mode review: Rosetta answers Claude Code's `safeguards` with a configured classifier.

Wire shapes follow what Claude Code 2.1.319 sends and accepts (captured live):
`safeguards[0].classifier_context` in, `safeguard_results` out.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rosetta.app import create_app
from rosetta.auto_mode.instructions import BASE_PROMPT, AutoModeSetupError
from rosetta.config import Config
from rosetta.observability import setup_logging
from tests.support.contracts import parse_sse, validate_anthropic_stream
from tests.support.upstream import FakeUpstream, build_client, load_fixture

_BETA = "dangerous-tool-use-2026-09-03"
_CONTEXT: dict[str, Any] = {
    "v": 1,
    "permission_mode": "auto",
    "live_cwd": "/work/repo",
    "auto_mode": {
        "allow": [],
        "soft_deny": ["$defaults", "Never touch /etc/hosts"],
        "hard_deny": [],
        "environment": ["Trusted repo: github.example.com/acme"],
    },
}
_SAFEGUARDS = [{"type": "dangerous_tool_use", "classifier_context": _CONTEXT}]
_TOOL_ID = "call_abc123"
_PROVIDERS: dict[str, Any] = {
    "abc": {
        "format": "openai_chat",
        "base_url": "https://abc.test/v1",
        "api_key": "k",
        "models": [{"id": "m1"}],
    },
    "cls": {
        "format": "openai_chat",
        "base_url": "https://cls.test/v1",
        "api_key": "k",
        "models": [{"id": "judge"}],
    },
    "anth": {
        "format": "anthropic",
        "base_url": "https://anth.test/v1",
        "api_key": "k",
        "models": [{"id": "claude"}],
    },
    "mirror": {
        "format": "anthropic",
        "base_url": "https://mirror.test/v1",
        "api_key": "k",
        "models": [{"id": "glm"}],
        "auto_mode_review": "rosetta",
    },
}
_AUTO_MODE: dict[str, Any] = {"classifier_model": "cls/judge", "builtin_rules": "none"}
_CONFIG: dict[str, Any] = {"providers": _PROVIDERS, "auto_mode": _AUTO_MODE}


def _messages(model: str, *, stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "safeguards": _SAFEGUARDS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>CLAUDE.md: tests first</system-reminder>",
                    },
                    {"type": "text", "text": "What's the weather in SF?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "AGENT PROSE"},
                    {
                        "type": "tool_use",
                        "id": "toolu_prev",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_prev", "content": "TOOL OUTPUT"}
                ],
            },
        ],
    }


def _chat_tool_call(finish_reason: str = "tool_calls") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "m1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": _TOOL_ID,
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                        }
                    ],
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def _classifier_reply(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-cls",
        "object": "chat.completion",
        "model": "judge",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _verdicts(**by_id: dict[str, Any]) -> str:
    return json.dumps({"verdicts": by_id})


def _results(body: dict[str, Any]) -> dict[str, Any]:
    results = body["safeguard_results"]
    assert [r["type"] for r in results] == ["dangerous_tool_use"]
    assert results[0]["status"]["type"] == "available"
    return results[0]["status"]["tool_uses"]


def _classifier_prompt(fake: FakeUpstream) -> tuple[str, str]:
    """(system, user) text of the one request the classifier received."""
    [request] = [r for r in fake.requests if r.provider_key == "cls"]
    messages = request.json()["messages"]
    return messages[0]["content"], messages[1]["content"]


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


def _client(fake: FakeUpstream, config: dict[str, Any] = _CONFIG) -> TestClient:
    return build_client(config, fake)


def test_translated_reply_carries_classifier_verdicts(fake: FakeUpstream) -> None:
    """The verdict comes from the classifier, keyed by the tool_use id Claude Code sees."""
    fake.add_json("abc", "POST", "/chat/completions", 200, _chat_tool_call())
    reason = "[Data Exfiltration] Sends the location to an unknown host."
    fake.add_json(
        "cls",
        "POST",
        "/chat/completions",
        200,
        _classifier_reply(_verdicts(**{_TOOL_ID: {"flagged": True, "explanation": reason}})),
    )
    with _client(fake) as c:
        r = c.post(
            "/v1/messages", json=_messages("claude-code/abc/m1"), headers={"anthropic-beta": _BETA}
        )

    assert r.status_code == 200
    body = r.json()
    assert [b["id"] for b in body["content"] if b["type"] == "tool_use"] == [_TOOL_ID]
    assert _results(body) == {
        _TOOL_ID: {"type": "evaluated", "outcome": "flagged", "explanation": reason}
    }
    assert "safeguards" not in next(r for r in fake.requests if r.provider_key == "abc").json()


def test_classifier_sees_claude_code_inputs_but_no_tool_output_or_agent_prose(
    fake: FakeUpstream,
) -> None:
    """classifier_context verbatim, user text and tool calls; tool results and assistant prose withheld."""
    fake.add_json("abc", "POST", "/chat/completions", 200, _chat_tool_call())
    fake.add_json(
        "cls",
        "POST",
        "/chat/completions",
        200,
        _classifier_reply(_verdicts(**{_TOOL_ID: {"flagged": False}})),
    )
    with _client(fake) as c:
        c.post("/v1/messages", json=_messages("abc/m1"))

    system, user = _classifier_prompt(fake)
    assert system.startswith(BASE_PROMPT)
    environment = user.split("<environment>\n", 1)[1].split("\n</environment>", 1)[0]
    assert json.loads(environment) == _CONTEXT
    transcript = user.split("<transcript>\n", 1)[1].split("\n</transcript>", 1)[0].splitlines()
    assert [json.loads(line) for line in transcript] == [
        {"user": "<system-reminder>CLAUDE.md: tests first</system-reminder>"},
        {"user": "What's the weather in SF?"},
        {"Bash": {"command": "ls"}},
    ]
    assert "TOOL OUTPUT" not in user
    assert "AGENT PROSE" not in user
    calls = json.loads(user.split("<tool_calls>\n", 1)[1].split("\n</tool_calls>", 1)[0])
    assert calls == [{"id": _TOOL_ID, "name": "get_weather", "input": {"location": "SF"}}]


def test_translated_stream_stamps_final_message_delta(fake: FakeUpstream) -> None:
    fake.add_sse(
        "abc", "/chat/completions", load_fixture("openai_chat", "tool_call"), chunk_size=37
    )
    fake.add_json(
        "cls",
        "POST",
        "/chat/completions",
        200,
        _classifier_reply(_verdicts(**{_TOOL_ID: {"flagged": False}})),
    )
    with (
        _client(fake) as c,
        c.stream("POST", "/v1/messages", json=_messages("abc/m1", stream=True)) as r,
    ):
        body = b"".join(r.iter_bytes())

    validate_anthropic_stream(body)
    events = [(ev.event, json.loads(ev.data)) for ev in parse_sse(body)]
    deltas = [data for name, data in events if name == "message_delta"]
    assert len(deltas) == 1
    assert _results(deltas[0]["delta"]) == {
        _TOOL_ID: {"type": "evaluated", "outcome": "not_flagged"}
    }
    assert [name for name, _ in events][-2:] == ["message_delta", "message_stop"]
    _, user = _classifier_prompt(fake)
    assert '"location":"San Francisco, CA"' in user.replace(": ", ":")


def test_reply_without_tool_calls_is_reviewed_without_calling_the_classifier(
    fake: FakeUpstream,
) -> None:
    reply = _chat_tool_call("stop")
    reply["choices"][0]["message"] = {"role": "assistant", "content": "Sunny."}
    fake.add_json("abc", "POST", "/chat/completions", 200, reply)
    with _client(fake) as c:
        r = c.post("/v1/messages", json=_messages("abc/m1"))

    assert _results(r.json()) == {}
    assert [r.provider_key for r in fake.requests] == ["abc"]


_CLS_PATH = "/chat/completions"


@pytest.mark.parametrize(
    ("classifier", "reason"),
    [
        ((500, {"error": {"message": "boom"}}), "error"),
        ((413, {"error": {"message": "too big"}}), "input_too_long"),
        (
            (400, {"error": {"message": "long", "code": "context_length_exceeded"}}),
            "input_too_long",
        ),
        ((200, _classifier_reply("I think it is fine")), "error"),
        ((200, _classifier_reply(_verdicts(other_id={"flagged": False}))), "error"),
        ((200, _classifier_reply(_verdicts(**{_TOOL_ID: {"flagged": "no"}}))), "error"),
        ((200, _classifier_reply("", "content_filter")), "refused"),
        (httpx2.ReadTimeout("slow"), "timeout"),
        (httpx2.ConnectError("down"), "error"),
    ],
    ids=[
        "status_500",
        "status_413",
        "context_length_code",
        "prose",
        "missing_id",
        "non_bool",
        "refusal",
        "timeout",
        "connect",
    ],
)
def test_classifier_failure_is_an_unavailable_verdict_never_an_allow(
    fake: FakeUpstream, classifier: tuple[int, dict[str, Any]] | Exception, reason: str
) -> None:
    """Claude Code denies a call whose verdict is unavailable, so failures fail closed."""
    fake.add_json("abc", "POST", "/chat/completions", 200, _chat_tool_call())
    if isinstance(classifier, Exception):
        fake.add_error("cls", "POST", _CLS_PATH, classifier)
    else:
        fake.add_json("cls", "POST", _CLS_PATH, *classifier)
    with _client(fake) as c:
        r = c.post("/v1/messages", json=_messages("abc/m1"))

    assert r.status_code == 200
    assert _results(r.json()) == {_TOOL_ID: {"type": "unavailable", "reason": reason}}


def test_tool_call_cut_off_by_max_tokens_is_truncated_and_not_classified(
    fake: FakeUpstream,
) -> None:
    fake.add_json("abc", "POST", "/chat/completions", 200, _chat_tool_call("length"))
    with _client(fake) as c:
        r = c.post("/v1/messages", json=_messages("abc/m1"))

    assert _results(r.json()) == {_TOOL_ID: {"type": "unavailable", "reason": "truncated"}}
    assert [r.provider_key for r in fake.requests] == ["abc"]


def test_without_classifier_translated_review_is_refused_with_remedy(fake: FakeUpstream) -> None:
    """Anthropic's gateway contract: a 400 naming `safeguards` makes Claude Code retry without it."""
    providers = {k: v for k, v in _PROVIDERS.items() if k != "mirror"}
    with _client(fake, {"providers": providers}) as c:
        r = c.post("/v1/messages", json=_messages("abc/m1"), headers={"anthropic-beta": _BETA})

    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"].startswith("safeguards: ")
    assert "auto_mode.classifier_model" in error["message"]
    assert fake.requests == []


def test_anthropic_provider_reviews_itself_by_default(fake: FakeUpstream) -> None:
    """auto_mode_review defaults to upstream: passthrough stays byte for byte and the classifier is unused."""
    fake.add_sse("anth", "/messages", "tool_call_safeguards", chunk_size=97)
    with (
        _client(fake) as c,
        c.stream(
            "POST",
            "/v1/messages",
            json=_messages("anth/claude", stream=True),
            headers={"anthropic-beta": _BETA},
        ) as r,
    ):
        body = b"".join(r.iter_bytes())

    assert body == load_fixture("anthropic", "tool_call_safeguards")
    assert fake.requests[-1].json()["safeguards"] == _SAFEGUARDS
    assert [r.provider_key for r in fake.requests] == ["anth"]


def test_rosetta_reviewed_anthropic_provider_gets_no_review_request_and_relays_other_events_verbatim(
    fake: FakeUpstream,
) -> None:
    upstream_sse = load_fixture("anthropic", "tool_call")
    fake.add_sse("mirror", "/messages", upstream_sse, chunk_size=53)
    tool_id = "toolu_01T1x1fJ34qAmk2tNTrN7Up6"
    fake.add_json(
        "cls",
        "POST",
        "/chat/completions",
        200,
        _classifier_reply(_verdicts(**{tool_id: {"flagged": False}})),
    )
    with (
        _client(fake) as c,
        c.stream(
            "POST",
            "/v1/messages",
            json=_messages("mirror/glm", stream=True),
            headers={"anthropic-beta": f"oauth-2025-04-20,{_BETA}"},
        ) as r,
    ):
        body = b"".join(r.iter_bytes())

    sent = next(r for r in fake.requests if r.provider_key == "mirror")
    assert "safeguards" not in sent.json()
    assert sent.headers["anthropic-beta"] == "oauth-2025-04-20"
    validate_anthropic_stream(body)
    upstream_events = [ev for ev in parse_sse(upstream_sse) if ev.event != "message_delta"]
    relayed = [ev for ev in parse_sse(body) if ev.event != "message_delta"]
    assert relayed == upstream_events
    [delta] = [json.loads(ev.data) for ev in parse_sse(body) if ev.event == "message_delta"]
    upstream_delta = next(
        json.loads(ev.data) for ev in parse_sse(upstream_sse) if ev.event == "message_delta"
    )
    assert {k: v for k, v in delta["delta"].items() if k != "safeguard_results"} == upstream_delta[
        "delta"
    ]
    assert _results(delta["delta"]) == {tool_id: {"type": "evaluated", "outcome": "not_flagged"}}


def _fake_claude(tmp_path: Path, defaults: str = "", *, exit_code: int = 0) -> Path:
    rules = defaults or json.dumps(
        {
            "allow": ["BUILTIN-ALLOW"],
            "soft_deny": ["BUILTIN-SOFT"],
            "hard_deny": ["BUILTIN-HARD"],
            "environment": ["BUILTIN-ENV"],
        }
    )
    (tmp_path / "defaults.json").write_text(rules)
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "--version" ]; then echo "2.1.999 (Claude Code)"; exit {exit_code}; fi\n'
        f'if [ "$1 $2" = "auto-mode defaults" ]; then cat "{tmp_path / "defaults.json"}"; exit {exit_code}; fi\n'
        "echo unexpected >&2; exit 64\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_builtin_rules_and_operator_instructions_reach_the_classifier(
    fake: FakeUpstream, tmp_path: Path
) -> None:
    """Built-in rules come from `claude auto-mode defaults`; the user's settings merge per section."""
    instructions = tmp_path / "auto-mode.md"
    instructions.write_text("Never deploy on Fridays.\n")
    auto_mode = {
        "classifier_model": "cls/judge",
        "claude_cli": str(_fake_claude(tmp_path)),
        "instructions_file": str(instructions),
    }
    fake.add_json("abc", "POST", "/chat/completions", 200, _chat_tool_call())
    fake.add_json(
        "cls",
        "POST",
        "/chat/completions",
        200,
        _classifier_reply(_verdicts(**{_TOOL_ID: {"flagged": False}})),
    )
    with _client(fake, {"providers": _PROVIDERS, "auto_mode": auto_mode}) as c:
        c.post("/v1/messages", json=_messages("abc/m1"))

    system, user = _classifier_prompt(fake)
    assert "Claude Code 2.1.999" in system
    for rule in (
        "BUILTIN-ALLOW",
        "BUILTIN-SOFT",
        "BUILTIN-HARD",
        "BUILTIN-ENV",
        "Never deploy on Fridays.",
    ):
        assert rule in system
    rules = user.split("<rules>\n", 1)[1].split("\n</rules>", 1)[0].splitlines()
    assert rules == [
        "hard_deny: the built-in hard_deny rules.",
        "soft_deny: the built-in soft_deny rules, plus:",
        "- Never touch /etc/hosts",
        "allow: the built-in allow rules.",
        "environment: only these rules; the built-in environment rules do not apply:",
        "- Trusted repo: github.example.com/acme",
    ]


@pytest.mark.parametrize(
    ("auto_mode", "message"),
    [
        ({"builtin_rules": "claude_cli", "claude_cli": "/nonexistent/claude"}, "was not found"),
        ({"instructions_file": "/nonexistent/auto-mode.md"}, "does not exist"),
        ({"classifier_model": "nope/judge"}, "Unknown provider 'nope'"),
    ],
    ids=["missing_cli", "missing_instructions", "unknown_classifier_provider"],
)
def test_unusable_auto_mode_setup_stops_startup_with_remedy(
    auto_mode: dict[str, Any], message: str
) -> None:
    config = Config.model_validate(
        {"providers": _PROVIDERS, "auto_mode": {**_AUTO_MODE, **auto_mode}}
    )
    with pytest.raises(AutoModeSetupError, match=message):
        create_app(config)


@pytest.mark.parametrize(
    ("defaults", "exit_code"),
    [("not json", 0), ('{"allow": []}', 0), ("{}", 1)],
    ids=["not_json", "missing_sections", "exit_status"],
)
def test_unexpected_claude_cli_output_stops_startup(
    tmp_path: Path, defaults: str, exit_code: int
) -> None:
    auto_mode = {
        "classifier_model": "cls/judge",
        "claude_cli": str(_fake_claude(tmp_path, defaults, exit_code=exit_code)),
    }
    with pytest.raises(AutoModeSetupError, match="builtin_rules"):
        create_app(Config.model_validate({"providers": _PROVIDERS, "auto_mode": auto_mode}))


@pytest.mark.parametrize(
    ("providers", "auto_mode", "message"),
    [
        (
            {"abc": {**_PROVIDERS["abc"], "auto_mode_review": "rosetta"}},
            _AUTO_MODE,
            "applies only to 'anthropic'",
        ),
        ({"mirror": _PROVIDERS["mirror"]}, None, "needs auto_mode.classifier_model"),
    ],
    ids=["non_anthropic_provider", "no_classifier"],
)
def test_auto_mode_review_setting_is_validated(
    providers: dict[str, Any], auto_mode: dict[str, Any] | None, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Config.model_validate({"providers": providers, "auto_mode": auto_mode})


def test_streamed_tool_call_that_never_closes_is_truncated_and_not_classified(
    fake: FakeUpstream,
) -> None:
    tool_id = "toolu_cut"
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'},
            },
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    sse = b"".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events
    )
    fake.add_sse("mirror", "/messages", sse)
    with (
        _client(fake) as c,
        c.stream("POST", "/v1/messages", json=_messages("mirror/glm", stream=True)) as r,
    ):
        body = b"".join(r.iter_bytes())

    [delta] = [json.loads(ev.data) for ev in parse_sse(body) if ev.event == "message_delta"]
    assert _results(delta["delta"]) == {tool_id: {"type": "unavailable", "reason": "truncated"}}
    assert [r.provider_key for r in fake.requests] == ["mirror"]


def test_review_outcome_is_logged_and_classifier_failure_warns(
    fake: FakeUpstream, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.add_json("abc", "POST", "/chat/completions", 200, _chat_tool_call())
    fake.add_json("cls", "POST", "/chat/completions", 503, {"error": {"message": "overloaded"}})
    setup_logging("debug")
    try:
        with _client(fake, {**_CONFIG, "log_completed": True}) as c:
            capsys.readouterr()
            r = c.post("/v1/messages", json=_messages("abc/m1"))
        lines = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
    finally:
        setup_logging("info")

    mine = [line for line in lines if line.get("request_id") == r.headers["x-request-id"]]
    [failed] = [line for line in mine if line["event"] == "auto_mode_review_failed"]
    assert {
        k: failed[k]
        for k in ("level", "reason", "classifier_model", "tool_calls", "upstream_status")
    } == {
        "level": "warning",
        "reason": "error",
        "classifier_model": "cls/judge",
        "tool_calls": 1,
        "upstream_status": 503,
    }
    [completed] = [line for line in mine if line["event"] == "request_completed"]
    assert (completed["auto_mode_review"], completed["auto_mode_flagged"]) == ("unavailable", 0)
    assert completed["upstream_status"] == 200
    assert {line["upstream_model"] for line in mine if line["event"] == "upstream_request"} == {
        "m1",
        "judge",
    }
