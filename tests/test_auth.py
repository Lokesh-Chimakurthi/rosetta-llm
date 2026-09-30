"""Authentication and exposure defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rosetta.__main__ import main
from rosetta.app import create_app
from rosetta.auth import InsecureBindError
from rosetta.config import Config
from tests.support.upstream import FakeUpstream, build_client

_PROVIDERS: dict[str, Any] = {
    "chat": {
        "format": "openai_chat",
        "base_url": "https://chat.test/v1",
        "api_key": "sk-up",
        "models": [{"id": "m"}],
    },
}
_KEYED: dict[str, Any] = {"proxy": {"api_keys": ["sk-proxy"]}, "providers": _PROVIDERS}


@pytest.mark.parametrize(
    ("host", "proxy", "starts"),
    [
        ("127.0.0.1", {}, True),
        ("localhost", {}, True),
        ("::1", {}, True),
        ("0.0.0.0", {}, False),  # noqa: S104 - the case under test
        ("::", {}, False),
        ("10.0.0.5", {}, False),
        ("0.0.0.0", {"api_keys": ["sk-proxy"]}, True),  # noqa: S104 - the case under test
        ("0.0.0.0", {"allow_unauthenticated": True}, True),  # noqa: S104 - the case under test
    ],
)
def test_startup_refuses_unauthenticated_non_loopback_bind(
    host: str, proxy: dict[str, Any], starts: bool
) -> None:
    config = Config.model_validate({"host": host, "proxy": proxy, "providers": _PROVIDERS})
    if starts:
        create_app(config)
        return
    with pytest.raises(InsecureBindError) as info:
        create_app(config)
    message = str(info.value)
    assert "--host 127.0.0.1" in message
    assert "api_keys" in message
    assert "--allow-unauthenticated" in message


def test_default_host_is_loopback() -> None:
    assert Config().host == "127.0.0.1"


def test_cli_exits_2_with_remedies_on_insecure_bind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"providers": _PROVIDERS}))
    with pytest.raises(SystemExit) as info:
        main(["--config", str(config_path), "--host", "0.0.0.0"])  # noqa: S104 - the case under test
    assert info.value.code == 2
    assert "--allow-unauthenticated" in capsys.readouterr().err


@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": "Bearer sk-proxy"},
        {"x-api-key": "sk-proxy"},
        {"authorization": "Bearer wrong", "x-api-key": "sk-proxy"},
    ],
)
def test_either_credential_header_authenticates(headers: dict[str, str]) -> None:
    with build_client(_KEYED, FakeUpstream()) as client:
        assert client.get("/v1/models", headers=headers).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {b"authorization": b"Bearer wrong"},
        {b"x-api-key": b"sk-prox"},
        {b"authorization": "Bearer ü".encode("latin-1")},
    ],
)
def test_missing_or_wrong_credentials_are_rejected(headers: dict[bytes, bytes]) -> None:
    with build_client(_KEYED, FakeUpstream()) as client:
        assert client.get("/v1/models", headers=headers).status_code == 401


@pytest.mark.parametrize(
    ("method", "path", "headers", "anthropic_envelope"),
    [
        ("POST", "/v1/messages", {}, True),
        ("POST", "/v1/messages/count_tokens", {}, True),
        ("POST", "/v1/chat/completions", {}, False),
        ("POST", "/v1/responses", {}, False),
        ("GET", "/v1/models", {"anthropic-version": "2023-06-01"}, True),
        ("GET", "/v1/models", {}, False),
        ("GET", "/providers", {}, False),
    ],
)
def test_401_uses_the_inbound_format_envelope(
    method: str, path: str, headers: dict[str, str], anthropic_envelope: bool
) -> None:
    with build_client(_KEYED, FakeUpstream()) as client:
        r = client.request(method, path, headers=headers)
    assert r.status_code == 401
    body = r.json()
    if anthropic_envelope:
        assert body == {
            "type": "error",
            "error": {"type": "authentication_error", "message": "Invalid or missing API key"},
        }
    else:
        assert body["error"]["type"] == "authentication_error"
        assert "type" not in body


@pytest.mark.parametrize("path", ["/health", "/openapi.json", "/docs", "/docs/oauth2-redirect"])
def test_public_paths_need_no_key(path: str) -> None:
    with build_client(_KEYED, FakeUpstream()) as client:
        assert client.get(path).status_code == 200
