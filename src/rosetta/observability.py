"""Structured JSON request logging.

Info: one `request_started` line per request (except /health); a
`request_completed` line too when `log_completed` is on, at info for 2xx/3xx,
warning for 4xx and error for 5xx. Debug adds `upstream_request` and
`upstream_response`; headers and bodies join the debug events only when
`log_headers` / `log_bodies` are on. Nothing is logged per chunk.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from rosetta.config import Config, LogLevel

Send = Callable[[dict[str, Any]], Awaitable[None]]
Receive = Callable[[], Awaitable[dict[str, Any]]]
ASGIApp = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]

_UNLOGGED_PATHS = frozenset({"/health"})
_REDACTED_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"}
)
# Inbound ids are echoed in headers and logs, so only a conservative charset is reused.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# Logged bodies are capped so one large prompt cannot flood the log.
BODY_LOG_LIMIT = 64 * 1024

type Route = Literal["passthrough", "translate"]
type AutoModeReviewOutcome = Literal["no_tool_calls", "reviewed", "unavailable"]

_STARTED_FIELDS = ("path", "model", "route", "stream")
_COMPLETED_FIELDS = (
    "path",
    "status",
    "duration_ms",
    "model",
    "route",
    "stream",
    "ttft_ms",
    "input_tokens",
    "output_tokens",
    "stop_reason",
    "error",
    "client_disconnected",
    "auto_mode_review",
    "auto_mode_flagged",
    "auto_mode_review_ms",
)
_COMPLETED_DEBUG_FIELDS = (
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "upstream_stop_reason",
    "upstream_status",
    "session_id",
    "dropped_fields",
)


def ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def redact_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {k.lower(): ("***" if k.lower() in _REDACTED_HEADERS else v) for k, v in headers}


def body_fields(prefix: bytes, size: int) -> dict[str, Any]:
    """`body` (at most BODY_LOG_LIMIT bytes of `prefix`), `body_bytes`, `body_truncated`."""
    out: dict[str, Any] = {
        "body": prefix[:BODY_LOG_LIMIT].decode(errors="replace"),
        "body_bytes": size,
    }
    if size > BODY_LOG_LIMIT:
        out["body_truncated"] = True
    return out


class _BodyTap:
    """receive() passed through unchanged, keeping at most BODY_LOG_LIMIT bytes.

    Nothing beyond the cap is kept, so large bodies are never buffered and
    disconnect semantics are untouched.
    """

    def __init__(self, receive: Receive) -> None:
        self._receive = receive
        self.prefix = bytearray()
        self.size = 0

    async def __call__(self) -> dict[str, Any]:
        message = await self._receive()
        if message["type"] == "http.request":
            chunk: bytes = message.get("body", b"")
            room = BODY_LOG_LIMIT - len(self.prefix)
            if room > 0:
                self.prefix.extend(chunk[:room])
            self.size += len(chunk)
        return message


@dataclass(slots=True)
class RequestLog:
    """Mutable record of one request. Fields left as None are not logged.

    `debug`, `log_headers` and `log_bodies` tell producers which optional
    detail to build; the middleware only sets the flags when debug is on.
    """

    request_id: str
    path: str
    debug: bool = False
    log_headers: bool = False
    log_bodies: bool = False
    started: float = field(default_factory=time.perf_counter)
    status: int | None = None
    duration_ms: int | None = None
    model: str | None = None
    route: Route | None = None
    stream: bool | None = None
    ttft_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    error: str | None = None
    client_disconnected: bool | None = None
    upstream_model: str | None = None
    upstream_status: int | None = None
    upstream_stop_reason: str | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_tokens: int | None = None
    session_id: str | None = None
    dropped_fields: list[str] | None = None
    auto_mode_review: AutoModeReviewOutcome | None = None
    auto_mode_flagged: int | None = None
    auto_mode_review_ms: int | None = None
    inbound_headers: Mapping[str, str] | None = None
    body_tap: _BodyTap | None = None
    started_logged: bool = False

    def log_started(self) -> None:
        """Emit `request_started` once; later calls are no-ops.

        The pipeline calls it once the model is resolved; the middleware calls
        it when the response starts, for requests that never reach the pipeline.
        """
        if self.started_logged or not self.request_id:
            return
        self.started_logged = True
        event = _present(self, _STARTED_FIELDS)
        if self.log_headers and self.inbound_headers is not None:
            event["headers"] = redact_headers(self.inbound_headers.items())
        if self.log_bodies and self.body_tap is not None:
            event.update(body_fields(bytes(self.body_tap.prefix), self.body_tap.size))
        get_logger().info("request_started", **event)

    def log_completed(self) -> None:
        names = _COMPLETED_FIELDS + _COMPLETED_DEBUG_FIELDS if self.debug else _COMPLETED_FIELDS
        event = _present(self, names)
        if self.error and self.upstream_status is not None:
            event["upstream_status"] = self.upstream_status
        status = self.status or 0
        log = get_logger()
        if status >= 500:
            log.error("request_completed", **event)
        elif status >= 400:
            log.warning("request_completed", **event)
        else:
            log.info("request_completed", **event)


def _present(rec: RequestLog, names: tuple[str, ...]) -> dict[str, Any]:
    return {n: v for n in names if (v := getattr(rec, n)) is not None}


_REQUEST_LOG: ContextVar[RequestLog | None] = ContextVar("rosetta_request_log", default=None)


@contextmanager
def sub_request_log(upstream_model: str) -> Generator[RequestLog]:
    """Give an extra upstream call made for this request its own record.

    Upstream events then name `upstream_model` and the request's own upstream
    fields stay untouched; log lines keep the request id. The record is never
    emitted as a request event.
    """
    parent = request_log()
    child = RequestLog(
        request_id="",
        path=parent.path,
        debug=parent.debug,
        log_headers=parent.log_headers,
        log_bodies=parent.log_bodies,
        upstream_model=upstream_model,
    )
    token = _REQUEST_LOG.set(child)
    try:
        yield child
    finally:
        _REQUEST_LOG.reset(token)


def request_log() -> RequestLog:
    """The current request's record; outside a request, a detached throwaway.

    The throwaway keeps background work (model refresh) from failing on a
    logging call; its request events are never emitted.
    """
    rec = _REQUEST_LOG.get()
    if rec is None:
        return RequestLog(request_id="", path="", debug=debug_enabled())
    return rec


_LEADING_KEYS = ("ts", "level", "event", "request_id")


def _leading_keys_first(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """Put ts, level, event and request_id first so lines scan well by eye."""
    head = {k: event_dict.pop(k) for k in _LEADING_KEYS if k in event_dict}
    return {**head, **event_dict}


def setup_logging(level: LogLevel) -> None:
    """Configure structlog once, before the app starts serving requests."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
            _leading_keys_first,
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


def log_translation_failed(
    direction: Literal["request", "response", "stream"], exc: Exception
) -> None:
    """One error line for a codec failure; the traceback only at debug."""
    rec = request_log()
    rec.error = "translation_error"
    get_logger().error(
        "translation_failed",
        direction=direction,
        error=f"{type(exc).__name__}: {exc}",
        exc_info=exc if rec.debug else None,
    )


def _request_id(inbound: str | None) -> str:
    if inbound is not None and _REQUEST_ID_RE.fullmatch(inbound):
        return inbound
    return uuid.uuid4().hex[:16]


class RequestContextMiddleware:
    """Pure ASGI middleware: request id, the `RequestLog`, and its events.

    With `log_completed`, `request_completed` is emitted when the final body is
    sent, when the app returns without sending it (the client went away), or
    when the app raises.
    """

    def __init__(self, app: ASGIApp, *, config: Config) -> None:
        self.app = app
        self._log_completed = config.log_completed
        self._log_headers = config.log_headers
        self._log_bodies = config.log_bodies

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
            request_id=request_id if logged else "",
            path=path,
            debug=debug,
            log_headers=debug and self._log_headers,
            log_bodies=debug and self._log_bodies,
        )
        if debug:
            rec.session_id = headers.get("x-claude-code-session-id")
        if rec.log_headers:
            rec.inbound_headers = headers
        if rec.log_bodies:
            rec.body_tap = _BodyTap(receive)
            receive = rec.body_tap
        token = _REQUEST_LOG.set(rec)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        pending = logged and self._log_completed
        response_done = False

        def complete() -> None:
            nonlocal pending
            if not pending:
                return
            pending = False
            rec.duration_ms = ms_since(rec.started)
            rec.log_completed()

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_done
            if message["type"] == "http.response.start":
                rec.status = message["status"]
                rec.log_started()
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                ]
            elif message["type"] == "http.response.body":
                if rec.stream and rec.ttft_ms is None and message.get("body"):
                    rec.ttft_ms = ms_since(rec.started)
                if not message.get("more_body", False):
                    response_done = True
            await send(message)
            if response_done:
                complete()

        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as exc:
            if rec.status is None:
                rec.status = 500
            rec.error = rec.error or type(exc).__name__
            rec.log_started()
            complete()
            raise
        finally:
            if not response_done and pending:
                rec.client_disconnected = True
            complete()
            structlog.contextvars.clear_contextvars()
            _REQUEST_LOG.reset(token)
