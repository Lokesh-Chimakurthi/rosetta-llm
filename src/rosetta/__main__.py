"""Entrypoint for `python -m rosetta` and the `rosetta-llm` CLI."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

import uvicorn

from rosetta.app import create_app
from rosetta.auth import InsecureBindError
from rosetta.config import LogLevel, load_config
from rosetta.observability import setup_logging

DEFAULT_CONFIG = str(Path.home() / ".rosetta-llm" / "config.json")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rosetta-llm",
        description="Multi-format bidirectional LLM proxy",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=os.environ.get("ROSETTA_CONFIG", DEFAULT_CONFIG),
        help=f"Path to config.json (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address (overrides config)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="Bind port (overrides config)",
    )
    parser.add_argument(
        "--log-level",
        choices=get_args(LogLevel),
        default=None,
        help="Log level (overrides config; default: info)",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        default=None,
        help="Serve without API keys on a non-loopback host (overrides config)",
    )
    parser.add_argument(
        "--proxy-headers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Trust X-Forwarded-* from --forwarded-allow-ips (overrides config; default: off)",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default=None,
        help="Comma-separated IPs trusted for proxy headers (overrides config; default: 127.0.0.1)",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)
    # Precedence: CLI flag > config > default.
    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.allow_unauthenticated is not None:
        config.proxy.allow_unauthenticated = args.allow_unauthenticated
    if args.proxy_headers is not None:
        config.proxy.trust_proxy_headers = args.proxy_headers
    if args.forwarded_allow_ips is not None:
        config.proxy.forwarded_allow_ips = args.forwarded_allow_ips
    setup_logging(config.log_level)

    try:
        app = create_app(config)
    except InsecureBindError as exc:
        print(f"rosetta-llm: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

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
    )


if __name__ == "__main__":
    main()
