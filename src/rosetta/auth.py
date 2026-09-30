"""API-key authentication as a pure ASGI middleware, and the exposure check.

BaseHTTPMiddleware buffers streaming responses, which would break SSE.
Implementing as raw ASGI keeps streams flowing chunk-by-chunk.
"""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Awaitable, Callable, Collection
from typing import Any

import orjson

from rosetta.config import Config
from rosetta.errors import format_error, inbound_format_for

Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]

# Never require a key: liveness, and the API docs (which carry no secrets).
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/docs/oauth2-redirect", "/openapi.json"})


class InsecureBindError(RuntimeError):
    """Serving would expose an unauthenticated proxy beyond this machine."""


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def ensure_safe_bind(config: Config) -> None:
    """Refuse a non-loopback bind with no API keys unless explicitly allowed.

    Raises:
        InsecureBindError: naming every remedy.
    """
    proxy = config.proxy
    if _is_loopback(config.host) or proxy.api_keys or proxy.allow_unauthenticated:
        return
    raise InsecureBindError(
        f"Refusing to serve on {config.host!r} without authentication: anyone who can reach "
        "it could spend your provider credits. Fix one of: bind to a loopback address "
        '(--host 127.0.0.1); set "proxy": {"api_keys": [...]} in the config; or, if another '
        'layer authenticates, pass --allow-unauthenticated (or "proxy": '
        '{"allow_unauthenticated": true}).'
    )


def _matches(candidate: str, keys: Collection[bytes]) -> bool:
    token = candidate.encode()
    # Every key is compared so the time taken does not depend on which one matched.
    matched = False
    for key in keys:
        matched |= hmac.compare_digest(token, key)
    return matched


class AuthMiddleware:
    """Rejects requests lacking a configured key, unless no keys are configured.

    A request passes when `Authorization: Bearer <k>` or `x-api-key: <k>`
    matches any key.
    """

    def __init__(self, app: ASGIApp, *, api_keys: Collection[str]) -> None:
        self.app = app
        self._keys = frozenset(k.encode() for k in api_keys)

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._keys or scope["path"] in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        auth_header = headers.get("authorization", "")
        bearer = auth_header[7:].strip() if auth_header[:7].lower() == "bearer " else ""
        api_key = headers.get("x-api-key", "").strip()
        if (bearer and _matches(bearer, self._keys)) or (
            api_key and _matches(api_key, self._keys)
        ):
            await self.app(scope, receive, send)
            return

        fmt = inbound_format_for(scope["path"], headers)
        body = orjson.dumps(
            format_error(fmt, 401, "authentication_error", "Invalid or missing API key")
        )
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
