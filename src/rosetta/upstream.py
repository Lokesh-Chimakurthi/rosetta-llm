"""Provider HTTP client. The only module that touches the HTTP library.

Callers see `UpstreamResponse`, `UpstreamStream` and the two exceptions below;
no httpx2 type crosses this module's boundary.
"""

from __future__ import annotations

import ssl
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import httpx2
import orjson

from rosetta import __version__
from rosetta.config import Config
from rosetta.observability import body_fields, get_logger, ms_since, redact_headers, request_log

type TransportErrorKind = Literal["connect", "timeout", "protocol"]

# Error bodies on streams are only ever shown to clients as a message; a
# misbehaving upstream must not make us buffer an unbounded body.
_STREAM_ERROR_BODY_LIMIT = 1024 * 1024
_POOL_TIMEOUT_S = 10.0
_USER_AGENT = f"rosetta/{__version__}"
_KIND_MESSAGE: dict[TransportErrorKind, str] = {
    "connect": "connection failed",
    "timeout": "timed out",
    "protocol": "connection broke",
}


@dataclass(frozen=True, slots=True)
class UpstreamResponse:
    """A fully read upstream response. Header names are lower-case."""

    status: int
    headers: Mapping[str, str]
    body: bytes

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        """Decode the body; raises `orjson.JSONDecodeError` (a `ValueError`)."""
        return orjson.loads(self.body)


class UpstreamError(Exception):
    """The upstream could not be reached or broke the connection.

    The message names only the provider key — never the URL or host, because
    it may be shown to clients.
    """

    def __init__(self, provider_key: str, kind: TransportErrorKind) -> None:
        self.provider_key = provider_key
        self.kind: TransportErrorKind = kind
        super().__init__(f"upstream provider '{provider_key}' {_KIND_MESSAGE[kind]}")


class UpstreamStatusError(Exception):
    """A stream request was answered with HTTP status >= 400."""

    def __init__(self, provider_key: str, response: UpstreamResponse) -> None:
        self.provider_key = provider_key
        self.response = response
        super().__init__(f"upstream provider '{provider_key}' returned HTTP {response.status}")


def _lower_headers(headers: httpx2.Headers) -> Mapping[str, str]:
    return MappingProxyType({k.lower(): v for k, v in headers.items()})


def _transport_error(provider_key: str, exc: httpx2.RequestError) -> UpstreamError:
    if isinstance(exc, httpx2.TimeoutException):
        return UpstreamError(provider_key, "timeout")
    if isinstance(exc, httpx2.ConnectError | httpx2.ProxyError | httpx2.UnsupportedProtocol):
        return UpstreamError(provider_key, "connect")
    return UpstreamError(provider_key, "protocol")


def _log_request(provider_key: str, request: httpx2.Request, *, stream: bool) -> bool:
    """Emit `upstream_request` at debug; returns whether debug is on."""
    rec = request_log()
    if not rec.debug:
        return False
    event: dict[str, Any] = {
        "provider": provider_key,
        "path": request.url.path,
        "upstream_model": rec.upstream_model,
        "stream": stream,
        "body_bytes": len(request.content),
    }
    if rec.log_headers:
        event["headers"] = redact_headers(request.headers.items())
    if rec.log_bodies:
        event.update(body_fields(request.content, len(request.content)))
    get_logger().debug("upstream_request", **event)
    return True


def _record_response(
    status: int, headers: Mapping[str, str], ttfb_ms: int, body: bytes | None, *, debug: bool
) -> None:
    """Record the upstream status; at debug emit `upstream_response`.

    `body` is None for a successful stream, whose body is never buffered.
    """
    rec = request_log()
    rec.upstream_status = status
    if not debug:
        return
    event: dict[str, Any] = {"status": status, "ttfb_ms": ttfb_ms}
    upstream_request_id = headers.get("request-id") or headers.get("x-request-id")
    if upstream_request_id:
        event["upstream_request_id"] = upstream_request_id
    if rec.log_headers:
        event["headers"] = redact_headers(headers.items())
    if rec.log_bodies and body is not None:
        event.update(body_fields(body, len(body)))
    get_logger().debug("upstream_response", **event)


class UpstreamStream:
    """An open streaming response whose status is below 400."""

    def __init__(self, provider_key: str, response: httpx2.Response) -> None:
        self._provider_key = provider_key
        self._response = response
        self.status = response.status_code
        self.headers = _lower_headers(response.headers)

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        """Yield the body as received, without content decoding."""
        try:
            async for chunk in self._response.aiter_raw():
                yield chunk
        except httpx2.RequestError as exc:
            raise _transport_error(self._provider_key, exc) from exc

    async def aclose(self) -> None:
        """Close the connection; safe to call more than once."""
        await self._response.aclose()

    async def _read_capped(self, limit: int) -> bytes:
        parts: list[bytes] = []
        size = 0
        async for chunk in self.iter_bytes():
            parts.append(chunk[: limit - size])
            size += len(parts[-1])
            if size >= limit:
                break
        return b"".join(parts)


class UpstreamClient:
    """One pooled HTTP client per provider, created in `start()`.

    Args:
        config: Provider connection settings are read from `config.providers`.
        transport: Replaces the network for every provider (tests use an
            in-memory transport). When set, TLS settings are not used.
    """

    def __init__(self, config: Config, transport: httpx2.AsyncBaseTransport | None = None) -> None:
        self._config = config
        self._transport = transport
        self._clients: dict[str, httpx2.AsyncClient] = {}

    async def start(self) -> None:
        # One context for all providers: building it loads the trust store,
        # which is the expensive part of client construction.
        ssl_context: ssl.SSLContext | None = None
        if self._transport is None:
            ssl_context = httpx2.create_ssl_context()
        for key, prov in self._config.providers.items():
            headers = dict(prov.extra_headers)
            if prov.format == "anthropic":
                headers["x-api-key"] = prov.api_key or ""
                headers.setdefault("anthropic-version", "2023-06-01")
            else:
                headers["Authorization"] = f"Bearer {prov.api_key or ''}"
            headers.setdefault("user-agent", _USER_AGENT)

            self._clients[key] = httpx2.AsyncClient(
                base_url=prov.base_url.rstrip("/"),
                headers=headers,
                timeout=httpx2.Timeout(
                    connect=prov.timeout.connect,
                    read=prov.timeout.read,
                    write=prov.timeout.read,
                    pool=_POOL_TIMEOUT_S,
                ),
                verify=ssl_context if ssl_context is not None else True,
                transport=self._transport,
            )

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    def _client(self, provider_key: str) -> httpx2.AsyncClient:
        client = self._clients.get(provider_key)
        if client is None:
            raise ValueError(f"Unknown provider: {provider_key}")
        return client

    async def post(
        self, provider_key: str, path: str, body: bytes, headers: Mapping[str, str]
    ) -> UpstreamResponse:
        """POST a JSON body and read the whole response, whatever its status.

        Raises:
            UpstreamError: on connect, timeout or protocol failure.
        """
        request = self._client(provider_key).build_request(
            "POST", path, content=body, headers={"content-type": "application/json", **headers}
        )
        return await self._send_and_read(provider_key, request)

    async def get(
        self,
        provider_key: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - per-phase HTTP timeout, not a call deadline
    ) -> UpstreamResponse:
        """GET and read the whole response, whatever its status.

        Args:
            timeout: Overrides every phase of the provider's timeout when set.

        Raises:
            UpstreamError: on connect, timeout or protocol failure.
        """
        client = self._client(provider_key)
        if timeout is None:
            request = client.build_request("GET", path, params=params)
        else:
            request = client.build_request("GET", path, params=params, timeout=timeout)
        return await self._send_and_read(provider_key, request)

    async def _send_and_read(self, provider_key: str, request: httpx2.Request) -> UpstreamResponse:
        client = self._client(provider_key)
        debug = _log_request(provider_key, request, stream=False)
        started = time.perf_counter()
        try:
            resp = await client.send(request, stream=True)
            ttfb_ms = ms_since(started)
            try:
                body = await resp.aread()
            finally:
                await resp.aclose()
        except httpx2.RequestError as exc:
            raise _transport_error(provider_key, exc) from exc
        result = UpstreamResponse(resp.status_code, _lower_headers(resp.headers), body)
        _record_response(result.status, result.headers, ttfb_ms, body, debug=debug)
        return result

    def stream(
        self, provider_key: str, path: str, body: bytes, headers: Mapping[str, str]
    ) -> AbstractAsyncContextManager[UpstreamStream]:
        """POST a JSON body and open the response as a stream.

        Entering sends the request and reads the status and headers; exiting
        closes the connection.

        Raises (on enter):
            UpstreamError: on connect, timeout or protocol failure.
            UpstreamStatusError: when the status is >= 400, carrying at most
                1 MiB of the error body.
        """
        return self._open_stream(provider_key, path, body, headers)

    @asynccontextmanager
    async def _open_stream(
        self, provider_key: str, path: str, body: bytes, headers: Mapping[str, str]
    ) -> AsyncGenerator[UpstreamStream]:
        client = self._client(provider_key)
        request = client.build_request(
            "POST",
            path,
            content=body,
            headers={
                "content-type": "application/json",
                "accept": "text/event-stream",
                # iter_bytes() yields raw bytes, so the body must not be compressed.
                "accept-encoding": "identity",
                **headers,
            },
        )
        debug = _log_request(provider_key, request, stream=True)
        started = time.perf_counter()
        try:
            response = await client.send(request, stream=True)
        except httpx2.RequestError as exc:
            raise _transport_error(provider_key, exc) from exc
        ttfb_ms = ms_since(started)
        stream = UpstreamStream(provider_key, response)
        try:
            if response.status_code >= 400:
                err_body = await stream._read_capped(_STREAM_ERROR_BODY_LIMIT)
                _record_response(stream.status, stream.headers, ttfb_ms, err_body, debug=debug)
                raise UpstreamStatusError(
                    provider_key, UpstreamResponse(stream.status, stream.headers, err_body)
                )
            _record_response(stream.status, stream.headers, ttfb_ms, None, debug=debug)
            yield stream
        finally:
            await stream.aclose()
