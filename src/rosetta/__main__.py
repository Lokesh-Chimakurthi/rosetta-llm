"""Entrypoint for `python -m rosetta` and the `rosetta-llm` CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from rosetta.app import create_app
from rosetta.config import load_config
from rosetta.observability import setup_logging

DEFAULT_CONFIG = str(Path.home() / ".rosetta-llm" / "config.json")


def main() -> None:
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
        "--proxy-headers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trust X-Forwarded-* headers from proxies (default: on)",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default="*",
        help="Allowed forwarded IPs (default: *)",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)

    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        access_log=False,
        log_config=None,
        proxy_headers=args.proxy_headers,
        forwarded_allow_ips=args.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()
