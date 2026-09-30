"""In-memory upstream: routes provider requests to canned responses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import httpx2
import orjson
from fastapi.testclient import TestClient

from rosetta.app import create_app
from rosetta.config import Config, Format
from rosetta.upstream import UpstreamClient

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sse"


def load_fixture(fmt: Format, name: str) -> bytes:
    """Raw SSE bytes of `tests/fixtures/sse/<fmt>/<name>.sse`."""
    return (_FIXTURES / fmt / f"{name}.sse").read_bytes()


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    provider_key: str
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        return orjson.loads(self.body)


@dataclass(frozen=True, slots=True)
class _Canned:
    status: int
    headers: Mapping[str, str]
    body: bytes | str
    stream: bool = False
    chunk_size: int | None = None
    delay_s: float = 0.0


class _BodyStream(httpx2.AsyncByteStream):
    """Unread response body, so the client streams it as it would from a socket."""

    def __init__(
        self,
        body: bytes,
        chunk_size: int | None,
        delay_s: float,
        on_close: Callable[[], None] | None,
    ) -> None:
        self._body = body
        self._chunk_size = chunk_size or max(len(body), 1)
        self._delay_s = delay_s
        self._on_close = on_close

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for start in range(0, len(self._body), self._chunk_size):
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            yield self._body[start : start + self._chunk_size]

    @override
    async def aclose(self) -> None:
        if self._on_close is not None:
            self._on_close()
            self._on_close = None


class FakeUpstream(httpx2.AsyncBaseTransport):
    """Transport for `UpstreamClient` that never touches the network.

    Requests are attributed to a provider by matching the URL against each
    provider's `base_url`, so `path` is relative to it (e.g. "/chat/completions").
    An unregistered request fails the test with an AssertionError. Registering
    the same route again replaces it.

    Attributes:
        requests: Every request received, in order.
        streams_opened: SSE responses handed to the client.
        streams_closed: SSE responses the client closed (each counted once).
    """

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.streams_opened = 0
        self.streams_closed = 0
        self._routes: dict[tuple[str, str, str], _Canned | Exception] = {}
        self._bases: list[tuple[str, str]] = []
        self._formats: dict[str, Format] = {}

    def bind(self, config: Config) -> None:
        """Learn each provider's base URL and format; `build_client` calls this."""
        self._formats = {key: prov.format for key, prov in config.providers.items()}
        self._bases = sorted(
            ((key, prov.base_url.rstrip("/")) for key, prov in config.providers.items()),
            key=lambda kb: len(kb[1]),
            reverse=True,
        )

    def add_json(
        self,
        provider_key: str,
        method: str,
        path: str,
        status: int,
        body: Any,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._routes[(provider_key, method.upper(), path)] = _Canned(
            status, {"content-type": "application/json", **(headers or {})}, orjson.dumps(body)
        )

    def add_sse(
        self,
        provider_key: str,
        path: str,
        fixture: str | bytes,
        *,
        chunk_size: int | None = None,
        delay_s: float = 0.0,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Serve an SSE body on POST `path`.

        Args:
            fixture: Raw bytes, or a fixture name loaded for the provider's
                format when the request arrives (see `load_fixture`).
            chunk_size: Split the body into pieces of this many bytes; by
                default it arrives as one piece.
            delay_s: Sleep before each piece.
        """
        self._routes[(provider_key, "POST", path)] = _Canned(
            status,
            {"content-type": "text/event-stream", **(headers or {})},
            fixture,
            stream=True,
            chunk_size=chunk_size,
            delay_s=delay_s,
        )

    def add_error(self, provider_key: str, method: str, path: str, exc: Exception) -> None:
        """Fail the request in the transport, as a dropped connection or timeout would."""
        self._routes[(provider_key, method.upper(), path)] = exc

    def _attribute(self, url: str) -> tuple[str, str]:
        bare = url.split("?", 1)[0]
        for key, base in self._bases:
            if bare == base or bare.startswith(base + "/"):
                return key, bare[len(base) :]
        raise AssertionError(f"request to {url} matches no configured provider")

    def _closed(self) -> None:
        self.streams_closed += 1

    @override
    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = await request.aread()
        provider_key, path = self._attribute(str(request.url))
        self.requests.append(
            RecordedRequest(
                provider_key,
                request.method,
                path,
                {k.lower(): v for k, v in request.headers.items()},
                body,
            )
        )
        canned = self._routes.get((provider_key, request.method, path))
        if canned is None:
            raise AssertionError(
                f"unexpected upstream request {request.method} {provider_key}{path}"
            )
        if isinstance(canned, Exception):
            raise canned
        payload = canned.body
        if isinstance(payload, str):
            payload = load_fixture(self._formats[provider_key], payload)
        on_close = None
        if canned.stream:
            self.streams_opened += 1
            on_close = self._closed
        return httpx2.Response(
            canned.status,
            headers=dict(canned.headers),
            stream=_BodyStream(payload, canned.chunk_size, canned.delay_s, on_close),
        )


def build_client(config: Mapping[str, Any], fake: FakeUpstream) -> TestClient:
    """App wired to `fake`; enter it (`with build_client(...) as c:`) to run the lifespan."""
    cfg = Config.model_validate(config)
    fake.bind(cfg)
    return TestClient(create_app(cfg, upstream=UpstreamClient(cfg, transport=fake)))
