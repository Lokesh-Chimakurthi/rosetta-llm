"""Structured JSON logging with request-scoped structlog contextvars."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog


def setup_logging(level: str) -> None:
    """Configure structlog once, before the app starts serving requests."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    processors.append(
        structlog.dev.ConsoleRenderer()
        if level == "debug"
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(**ctx: Any) -> Any:
    return structlog.get_logger().bind(**ctx) if ctx else structlog.get_logger()


def bind_request_context(**kv: Any) -> None:
    """Bind keys onto structlog's contextvars for the duration of a request."""
    structlog.contextvars.bind_contextvars(**kv)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


Send = Callable[[dict[str, Any]], Awaitable[None]]
Receive = Callable[[], Awaitable[dict[str, Any]]]
ASGIApp = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]


class RequestContextMiddleware:
    """Pure ASGI middleware that sets a request_id contextvar and echoes the header."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        request_id = headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=scope.get("method", ""),
            path=scope.get("path", ""),
        )

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers_list = list(message.get("headers", []))
                headers_list.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers_list
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            structlog.contextvars.clear_contextvars()
