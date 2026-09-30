"""Structured JSON logging: one `request_completed` line per request.

At info, each HTTP request (except /health) produces exactly one
`request_completed` event built from its `RequestLog`. At debug, four stage
events add detail: `request_received`, `upstream_request`, `upstream_response`
and, for streams, `stream_summary`. Debug detail is only computed when debug
is enabled.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from rosetta.config import LogLevel

Send = Callable[[dict[str, Any]], Awaitable[None]]
Receive = Callable[[], Awaitable[dict[str, Any]]]
ASGIApp = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]

_UNLOGGED_PATHS = frozenset({"/health"})
_REDACTED_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"}
)
# Inbound ids are echoed in headers and logs, so only a conservative charset is reused.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_BODY_PREVIEW_CHARS = 2048
_ERROR_PREVIEW_CHARS = 1024
# Enough bytes for 2048 characters of 4-byte UTF-8.
_BODY_PREVIEW_BYTES = _BODY_PREVIEW_CHARS * 4

type Route = Literal["passthrough", "translate"]

_INFO_FIELDS = (
    "method",
    "path",
    "status",
    "duration_ms",
    "route",
    "provider",
    "model",
    "stream",
    "ttft_ms",
    "input_tokens",
    "output_tokens",
    "error_type",
    "client_disconnected",
)
_DEBUG_FIELDS = (
    "inbound_format",
    "provider_format",
    "upstream_model",
    "upstream_status",
    "upstream_request_id",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
    "raw_stop_reason",
    "dropped_params",
    "session_id",
    "client",
)
_STREAM_COUNTERS = ("chunks_in", "chunks_out", "dropped_events", "keepalives")


@dataclass(slots=True)
class RequestLog:
    """Mutable record of one request, emitted once as `request_completed`.

    Fields left as None are omitted from the event. `debug` tells producers
    whether debug-only detail (previews, header copies, counters) is wanted.
    """

    request_id: str
    method: str
    path: str
    debug: bool
    started: float = field(default_factory=time.perf_counter)
    status: int | None = None
    duration_ms: float | None = None
    route: Route | None = None
    provider: str | None = None
    model: str | None = None
    stream: bool | None = None
    ttft_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_type: str | None = None
    client_disconnected: bool | None = None
    inbound_format: str | None = None
    provider_format: str | None = None
    upstream_model: str | None = None
    upstream_status: int | None = None
    upstream_request_id: str | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw_stop_reason: str | None = None
    dropped_params: list[str] | None = None
    session_id: str | None = None
    client: str | None = None
    chunks_in: int = 0
    chunks_out: int = 0
    dropped_events: int = 0
    keepalives: int = 0

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 1)

    def completed_fields(self) -> dict[str, Any]:
        names = _INFO_FIELDS + _DEBUG_FIELDS if self.debug else _INFO_FIELDS
        return {n: v for n in names if (v := getattr(self, n)) is not None}

    def stream_counters(self) -> dict[str, int]:
        return {n: getattr(self, n) for n in _STREAM_COUNTERS}


_REQUEST_LOG: ContextVar[RequestLog | None] = ContextVar("rosetta_request_log", default=None)


def request_log() -> RequestLog:
    """The current request's record; outside a request, a detached throwaway.

    The throwaway keeps background work (model refresh) from failing on a
    logging call; nothing ever emits it.
    """
    rec = _REQUEST_LOG.get()
    if rec is None:
        return RequestLog(request_id="", method="", path="", debug=False)
    return rec


def setup_logging(level: LogLevel) -> None:
    """Configure structlog once, before the app starts serving requests."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(**ctx: Any) -> Any:
    return structlog.get_logger().bind(**ctx) if ctx else structlog.get_logger()


def debug_enabled() -> bool:
    return bool(structlog.get_logger().is_enabled_for(logging.DEBUG))


def redact_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {k: ("***" if k in _REDACTED_HEADERS else v) for k, v in headers}


def upstream_response_fields(
    status: int, headers: Mapping[str, str], ttfb_ms: float, body: bytes | None
) -> dict[str, Any]:
    """Debug fields for `upstream_response`; `body` is only read when status >= 400."""
    out: dict[str, Any] = {"status": status, "ttfb_ms": ttfb_ms}
    if "request-id" in headers:
        out["request-id"] = headers["request-id"]
    out.update({k: v for k, v in headers.items() if "ratelimit" in k})
    if status >= 400 and body is not None:
        out["error_preview"] = body.decode(errors="replace")[:_ERROR_PREVIEW_CHARS]
    return out


def _request_id(inbound: str | None) -> str:
    if inbound is not None and _REQUEST_ID_RE.fullmatch(inbound):
        return inbound
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Pure ASGI middleware: request id, the `RequestLog`, and its emission.

    `request_completed` is emitted when the final response body is sent, when
    the app returns without sending it (the client went away), or when the
    app raises.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in raw_headers}
        request_id = _request_id(headers.get("x-request-id"))
        path = scope.get("path", "")
        logged = path not in _UNLOGGED_PATHS
        debug = logged and debug_enabled()
        rec = RequestLog(
            request_id=request_id, method=scope.get("method", ""), path=path, debug=debug
        )
        rec.session_id = headers.get("x-claude-code-session-id")
        token = _REQUEST_LOG.set(rec)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        log = get_logger()

        emitted = False
        response_done = False

        def emit() -> None:
            nonlocal emitted
            if emitted or not logged:
                return
            emitted = True
            rec.duration_ms = rec.elapsed_ms()
            if debug and rec.stream:
                log.debug("stream_summary", **rec.stream_counters())
            log.info("request_completed", **rec.completed_fields())

        tap = _BodyTap(receive, headers, scope, log) if debug else None
        if tap is not None:
            receive = tap

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_done
            if message["type"] == "http.response.start":
                if tap is not None:
                    tap.report()
                rec.status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                ]
            elif message["type"] == "http.response.body":
                if rec.stream and rec.ttft_ms is None and message.get("body"):
                    rec.ttft_ms = rec.elapsed_ms()
                if not message.get("more_body", False):
                    response_done = True
            await send(message)
            if response_done:
                emit()

        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as exc:
            if rec.status is None:
                rec.status = 500
            rec.error_type = rec.error_type or type(exc).__name__
            emit()
            raise
        finally:
            if not emitted:
                rec.client_disconnected = True
                emit()
            structlog.contextvars.clear_contextvars()
            _REQUEST_LOG.reset(token)


class _BodyTap:
    """receive() passed through unchanged; logs `request_received` once.

    It reports when the body ends or, for requests the app never reads, when
    the response starts. Only a bounded preview is kept, so large bodies are
    never buffered and disconnect semantics are untouched.
    """

    def __init__(
        self, receive: Receive, headers: dict[str, str], scope: dict[str, Any], log: Any
    ) -> None:
        self._receive = receive
        self._headers = headers
        self._query: bytes = scope.get("query_string", b"")
        self._log = log
        self._preview = bytearray()
        self._size = 0
        self._reported = False

    async def __call__(self) -> dict[str, Any]:
        message = await self._receive()
        if not self._reported and message["type"] == "http.request":
            chunk: bytes = message.get("body", b"")
            self._size += len(chunk)
            room = _BODY_PREVIEW_BYTES - len(self._preview)
            if room > 0:
                self._preview.extend(chunk[:room])
            if not message.get("more_body", False):
                self.report()
        return message

    def report(self) -> None:
        if self._reported:
            return
        self._reported = True
        self._log.debug(
            "request_received",
            headers=redact_headers(self._headers.items()),
            query=self._query.decode("latin-1"),
            body_bytes=self._size,
            body_preview=self._preview.decode(errors="replace")[:_BODY_PREVIEW_CHARS],
        )
