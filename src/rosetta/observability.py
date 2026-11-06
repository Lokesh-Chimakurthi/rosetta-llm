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

# Header values masked in debug logs so secrets never hit disk.
_SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "cookie"})
# Cap logged body size to avoid flooding logs with large payloads.
_MAX_LOGGED_BODY = 8192


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("***" if k in _SENSITIVE_HEADERS else v) for k, v in headers.items()}


async def _buffer_body(receive: Receive) -> tuple[bytes, Receive]:
    """Drain the request body, then return it plus a receive() that replays it.

    Buffering the full body keeps streaming intact: the wrapped receive replays
    the captured chunks so downstream handlers read the same bytes. Once the
    replay is exhausted, calls forward to the original receive — per ASGI, the
    only message left there is a real http.disconnect. Forwarding (rather than
    fabricating a synchronous disconnect) preserves `Request.is_disconnected()`
    semantics: starlette probes with a pre-cancelled scope, so a genuine
    blocking await reads as "still connected" while a real disconnect still
    arrives. Fabricating one made every streaming response die on its first
    chunk under debug logging.
    """
    chunks: list[bytes] = []
    more = True
    while more:
        message = await receive()
        if message["type"] != "http.request":
            # Re-emit non-body messages (e.g. http.disconnect) verbatim.
            replayed = [message]
            body = b"".join(chunks)

            async def _replay_other(
                replayed: list[dict[str, Any]] = replayed,
            ) -> dict[str, Any]:
                if replayed:
                    return replayed.pop(0)
                return await receive()

            return body, _replay_other
        chunks.append(message.get("body", b""))
        more = message.get("more_body", False)

    body = b"".join(chunks)
    sent = False

    async def _replay() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        # Body fully replayed; the only remaining message on the original
        # channel is a real http.disconnect — forward to it, don't invent one.
        return await receive()

    return body, _replay


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

        log = get_logger()
        if log.is_enabled_for(logging.DEBUG):
            body, receive = await _buffer_body(receive)
            log.debug(
                "request_received",
                headers=_redact_headers(headers),
                query=scope.get("query_string", b"").decode(),
                body=body[:_MAX_LOGGED_BODY].decode("utf-8", "replace"),
                body_bytes=len(body),
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
