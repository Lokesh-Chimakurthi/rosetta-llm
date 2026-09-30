"""CLI contract: commands, the bare-flags default, overrides and exit statuses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rosetta import __version__
from rosetta.__main__ import main

_PROVIDERS: dict[str, Any] = {
    "chat": {
        "format": "openai_chat",
        "base_url": "https://chat.secret-host.test/v1",
        "api_key": "sk-provider-secret",
        "models": [{"id": "m1"}, {"id": "m2"}],
    },
    "disc": {
        "format": "anthropic",
        "base_url": "https://disc.test/v1",
        "api_key": "sk-provider-secret",
        "models": ["*"],
    },
}


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Captures uvicorn.run kwargs instead of starting a server."""
    calls: list[dict[str, Any]] = []

    def run(app: Any, **kwargs: Any) -> None:
        calls.append({**kwargs, "config": app.state.config})

    monkeypatch.setattr("rosetta.__main__.uvicorn.run", run)
    return calls


def _config(tmp_path: Path, **overrides: Any) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"providers": _PROVIDERS, **overrides}))
    return str(path)


@pytest.mark.parametrize("command", [[], ["serve"]])
def test_bare_flags_and_serve_both_run_the_server(
    command: list[str], tmp_path: Path, served: list[dict[str, Any]]
) -> None:
    main([*command, "--config", _config(tmp_path), "--port", "9999"])
    assert len(served) == 1
    assert (served[0]["host"], served[0]["port"]) == ("127.0.0.1", 9999)
    assert served[0]["ws"] == "none"


@pytest.mark.parametrize(
    ("config", "flags", "kwarg", "expected"),
    [
        ({}, [], "host", "127.0.0.1"),
        ({"host": "::1"}, [], "host", "::1"),
        ({"host": "::1"}, ["--host", "localhost"], "host", "localhost"),
        ({}, [], "port", 7860),
        ({"port": 8000}, [], "port", 8000),
        ({"port": 8000}, ["-p", "8001"], "port", 8001),
        ({}, [], "log_level", "info"),
        ({"log_level": "warning"}, [], "log_level", "warning"),
        ({"log_level": "warning"}, ["--log-level", "error"], "log_level", "error"),
        ({}, [], "proxy_headers", False),
        ({"proxy": {"trust_proxy_headers": True}}, [], "proxy_headers", True),
        ({"proxy": {"trust_proxy_headers": True}}, ["--no-proxy-headers"], "proxy_headers", False),
        ({}, ["--proxy-headers"], "proxy_headers", True),
        ({}, [], "forwarded_allow_ips", "127.0.0.1"),
        ({"proxy": {"forwarded_allow_ips": "10.0.0.1"}}, [], "forwarded_allow_ips", "10.0.0.1"),
        (
            {"proxy": {"forwarded_allow_ips": "10.0.0.1"}},
            ["--forwarded-allow-ips", "10.0.0.2"],
            "forwarded_allow_ips",
            "10.0.0.2",
        ),
        ({}, [], "root_path", ""),
        ({}, ["--root-path", "/rosetta"], "root_path", "/rosetta"),
        ({}, [], "timeout_keep_alive", 5),
        ({}, ["--timeout-keep-alive", "30"], "timeout_keep_alive", 30),
        ({}, [], "limit_concurrency", None),
        ({}, ["--limit-concurrency", "64"], "limit_concurrency", 64),
    ],
)
def test_flag_overrides_config_overrides_default(
    config: dict[str, Any],
    flags: list[str],
    kwarg: str,
    expected: Any,
    tmp_path: Path,
    served: list[dict[str, Any]],
) -> None:
    main(["serve", "--config", _config(tmp_path, **config), *flags])
    assert served[0][kwarg] == expected


@pytest.mark.parametrize("name", ["completed", "headers", "bodies"])
@pytest.mark.parametrize(
    ("config_value", "flag", "expected"),
    [
        (None, None, False),
        (True, None, True),
        (True, "--no-log-{}", False),
        (False, "--log-{}", True),
    ],
)
def test_log_flags_override_config_override_default(
    name: str,
    config_value: bool | None,
    flag: str | None,
    expected: bool,
    tmp_path: Path,
    served: list[dict[str, Any]],
) -> None:
    config = {} if config_value is None else {f"log_{name}": config_value}
    flags = [] if flag is None else [flag.format(name)]
    main(["serve", "--config", _config(tmp_path, **config), *flags])
    assert getattr(served[0]["config"], f"log_{name}") is expected


@pytest.mark.parametrize(
    ("config", "flags", "starts"),
    [
        ({"host": "0.0.0.0"}, [], False),  # noqa: S104 - the case under test
        ({"host": "0.0.0.0", "proxy": {"allow_unauthenticated": True}}, [], True),  # noqa: S104
        ({"host": "0.0.0.0"}, ["--allow-unauthenticated"], True),  # noqa: S104
    ],
)
def test_allow_unauthenticated_flag_and_config(
    config: dict[str, Any],
    flags: list[str],
    starts: bool,
    tmp_path: Path,
    served: list[dict[str, Any]],
) -> None:
    argv = ["serve", "--config", _config(tmp_path, **config), *flags]
    if starts:
        main(argv)
        assert len(served) == 1
    else:
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == 2
        assert served == []


def test_check_prints_a_secret_free_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _config(tmp_path, proxy={"api_keys": ["sk-proxy-secret"]})
    main(["check", "--config", path, "--port", "9001"])
    out = capsys.readouterr().out
    assert "bind:          127.0.0.1:9001" in out
    assert "1 API key" in out
    assert "chat  openai_chat       2 models" in out
    assert "disc  anthropic         discovers models" in out
    assert out.rstrip().endswith("ok")
    for secret in ("sk-proxy-secret", "sk-provider-secret", "secret-host"):
        assert secret not in out


@pytest.mark.parametrize(
    ("content", "flags", "reason"),
    [
        (None, [], "config file not found"),
        ("{not json", [], "not valid JSON"),
        (json.dumps({"providers": {"a/b": _PROVIDERS["chat"]}}), [], "is invalid"),
        (json.dumps({"providers": _PROVIDERS}), ["--host", "0.0.0.0"], "--allow-unauthenticated"),  # noqa: S104
    ],
)
def test_check_exits_2_with_the_reason(
    content: str | None,
    flags: list[str],
    reason: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "config.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(SystemExit) as info:
        main(["check", "--config", str(path), *flags])
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert reason in captured.err
    assert "Traceback" not in captured.err
    assert "ok" not in captured.out


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    main(["version"])
    assert capsys.readouterr().out.strip() == f"rosetta-llm {__version__}"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"rosetta-llm {__version__}"
