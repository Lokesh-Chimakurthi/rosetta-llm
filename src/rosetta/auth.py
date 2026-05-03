"""Bearer-token authentication as a pure ASGI middleware.

BaseHTTPMiddleware buffers streaming responses, which would break SSE.
Implementing as raw ASGI keeps streams flowing chunk-by-chunk.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import orjson

ASGIApp = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[dict[str, Any]]],
        Callable[[dict[str, Any]], Awaitable[None]],
    ],
    Awaitable[None],
]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

_PUBLIC_PATHS = frozenset({"/health", "/providers"})


class AuthMiddleware:
    """Strict ASGI middleware that rejects requests lacking a valid bearer token."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        config = scope["app"].state.config
        keys = set(config.proxy.api_keys)
        if not keys or scope["path"] in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        token = ""
        auth_header = headers.get("authorization", "")
        if auth_header[:7].lower() == "bearer ":
            token = auth_header[7:].strip()
        if not token:
            token = headers.get("x-api-key", "").strip()

        if token in keys:
            await self.app(scope, receive, send)
            return

        body = _error_body(scope["path"])
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _error_body(path: str) -> bytes:
    if path.startswith("/v1/messages"):
        return orjson.dumps(
            {
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid or missing API key"},
            }
        )
    return orjson.dumps(
        {
            "error": {
                "message": "Invalid or missing API key",
                "type": "authentication_error",
                "code": None,
                "param": None,
            },
        }
    )
