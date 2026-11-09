"""Entrypoint for `python -m rosetta` and the `rosetta-llm` CLI.

Commands: `serve` (the default when none is given), `check`, `version`.
Precedence for every setting a flag can override: CLI flag > config > default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, get_args

import uvicorn
from pydantic import ValidationError

from rosetta import __version__
from rosetta.app import create_app
from rosetta.auth import InsecureBindError, ensure_safe_bind
from rosetta.auto_mode.classifier import describe, prepare
from rosetta.auto_mode.instructions import SECTIONS, AutoModeSetupError
from rosetta.config import Config, LogLevel, load_config
from rosetta.observability import setup_logging

DEFAULT_CONFIG = str(Path.home() / ".rosetta-llm" / "config.json")
_COMMANDS = frozenset({"serve", "check", "version"})
_ROOT_FLAGS = frozenset({"-h", "--help", "--version"})


def _config_options() -> argparse.ArgumentParser:
    """Options shared by `serve` and `check`: the config file and its overrides."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--config",
        "-c",
        default=os.environ.get("ROSETTA_CONFIG", DEFAULT_CONFIG),
        help=f"Config file (default: $ROSETTA_CONFIG, else {DEFAULT_CONFIG})",
    )
    parent.add_argument("--host", help="Bind address (overrides config; default: 127.0.0.1)")
    parent.add_argument(
        "--port", "-p", type=int, help="Bind port (overrides config; default: 7860)"
    )
    parent.add_argument(
        "--log-level",
        choices=get_args(LogLevel),
        help="Log level (overrides config; default: info)",
    )
    for name, what in (
        ("completed", "also log request_completed (status, latency, tokens) per request"),
        ("headers", "add headers (credentials redacted) to debug events"),
        ("bodies", "add bodies (capped at 64 KiB) to debug events"),
    ):
        parent.add_argument(
            f"--log-{name}",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"{what.capitalize()} (overrides config; default: off)",
        )
    parent.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        default=None,
        help="Serve without API keys on a non-loopback host (overrides config)",
    )
    parent.add_argument(
        "--proxy-headers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Trust X-Forwarded-* from --forwarded-allow-ips (overrides config; default: off)",
    )
    parent.add_argument(
        "--forwarded-allow-ips",
        help="Comma-separated IPs trusted for proxy headers (overrides config; default: 127.0.0.1)",
    )
    return parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosetta-llm",
        description=(
            "Multi-format bidirectional LLM proxy. Without a command, runs `serve` "
            "(so `rosetta-llm --port 9000` still works)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rosetta-llm {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    options = _config_options()

    serve = commands.add_parser(
        "serve", parents=[options], help="Run the proxy (default)", description="Run the proxy."
    )
    serve.add_argument(
        "--root-path",
        default="",
        help="Path prefix when mounted behind a reverse proxy under a sub-path (default: none)",
    )
    serve.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=5,
        metavar="SECONDS",
        help="Close idle client keep-alive connections after this long (default: 5)",
    )
    serve.add_argument(
        "--limit-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="Answer 503 beyond N concurrent connections (default: unlimited)",
    )
    serve.set_defaults(handler=_serve)

    check = commands.add_parser(
        "check",
        parents=[options],
        help="Validate the config and print a summary",
        description=(
            "Load and validate the config with the same overrides as `serve`, run the "
            "exposure check, and print a summary without secrets or URLs. Exit status 0 "
            "when serve would start, 2 otherwise."
        ),
    )
    check.set_defaults(handler=_check)

    version = commands.add_parser("version", help="Print the version")
    version.set_defaults(handler=_version)
    return parser


def _fail(message: str) -> NoReturn:
    print(f"rosetta-llm: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load(args: argparse.Namespace) -> Config:
    """Load the config and apply CLI overrides; exits 2 with the reason on failure."""
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        _fail(f"config file not found: {args.config} (pass --config or set ROSETTA_CONFIG)")
    except OSError as exc:
        _fail(f"cannot read config file {args.config}: {exc.strerror}")
    except json.JSONDecodeError as exc:
        _fail(f"config file {args.config} is not valid JSON: {exc}")
    except ValidationError as exc:
        _fail(f"config file {args.config} is invalid:\n{exc}")
    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.log_completed is not None:
        config.log_completed = args.log_completed
    if args.log_headers is not None:
        config.log_headers = args.log_headers
    if args.log_bodies is not None:
        config.log_bodies = args.log_bodies
    if args.allow_unauthenticated is not None:
        config.proxy.allow_unauthenticated = args.allow_unauthenticated
    if args.proxy_headers is not None:
        config.proxy.trust_proxy_headers = args.proxy_headers
    if args.forwarded_allow_ips is not None:
        config.proxy.forwarded_allow_ips = args.forwarded_allow_ips
    return config


def _serve(args: argparse.Namespace) -> None:
    config = _load(args)
    setup_logging(config.log_level)
    try:
        app = create_app(config)
    except (InsecureBindError, AutoModeSetupError) as exc:
        _fail(str(exc))
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        access_log=False,
        log_config=None,
        ws="none",
        proxy_headers=config.proxy.trust_proxy_headers,
        forwarded_allow_ips=config.proxy.forwarded_allow_ips,
        root_path=args.root_path,
        timeout_keep_alive=args.timeout_keep_alive,
        limit_concurrency=args.limit_concurrency,
    )


def _auth_summary(config: Config) -> str:
    proxy = config.proxy
    if proxy.api_keys:
        count = len(proxy.api_keys)
        return f"{count} API key{'s' if count != 1 else ''} (Bearer or x-api-key)"
    if proxy.allow_unauthenticated:
        return "none (allow_unauthenticated)"
    return "none (loopback only)"


def _log_extras(config: Config) -> str:
    extras = [
        name
        for name, on in (
            ("completed", config.log_completed),
            ("headers", config.log_headers),
            ("bodies", config.log_bodies),
        )
        if on
    ]
    return f" (+{', '.join(extras)})" if extras else ""


def _check(args: argparse.Namespace) -> None:
    config = _load(args)
    try:
        ensure_safe_bind(config)
    except InsecureBindError as exc:
        _fail(str(exc))
    proxy = config.proxy
    headers = f"trusted from {proxy.forwarded_allow_ips}" if proxy.trust_proxy_headers else "off"
    lines = [
        f"config:        {args.config}",
        f"bind:          {config.host}:{config.port}",
        f"auth:          {_auth_summary(config)}",
        f"proxy headers: {headers}",
        f"log level:     {config.log_level}{_log_extras(config)}",
        f"providers:     {len(config.providers)}",
    ]
    width = max((len(key) for key in config.providers), default=0)
    for key, prov in config.providers.items():
        count = len(prov.static_models)
        models = (
            "discovers models"
            if prov.is_auto_discover
            else f"{count} model{'s' if count != 1 else ''}"
        )
        lines.append(f"  {key:<{width}}  {prov.format:<16}  {models}")
    lines.append(f"auto mode:     {_auto_mode_summary(config)}")
    lines.append("ok")
    print("\n".join(lines))


def _auto_mode_summary(config: Config) -> str:
    """Sets the classifier up as `serve` would (running the Claude Code CLI if configured)."""
    if config.auto_mode is None:
        return "off (auto_mode not set)"
    try:
        summary = describe(*prepare(config, config.auto_mode))
    except AutoModeSetupError as exc:
        _fail(str(exc))
    rules = summary["builtin_rules"]
    builtin = (
        "no built-in rules"
        if rules is None
        else f"Claude Code {rules['claude_code']} built-in rules ("
        + ", ".join(f"{rules[s]} {s}" for s in SECTIONS)
        + ")"
    )
    instructions = summary["instructions_file"] or "no instructions file"
    return f"classifier {summary['classifier_model']}; {builtin}; {instructions}"


def _version(_args: argparse.Namespace) -> None:
    print(f"rosetta-llm {__version__}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI. Failures print the reason to stderr and exit with status 2."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or (args_list[0] not in _COMMANDS and args_list[0] not in _ROOT_FLAGS):
        args_list.insert(0, "serve")
    args = _build_parser().parse_args(args_list)
    args.handler(args)


if __name__ == "__main__":
    main()
