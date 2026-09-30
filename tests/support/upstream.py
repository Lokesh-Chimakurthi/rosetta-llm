"""In-memory upstream: routes provider requests to canned responses."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, override

import httpx2
import orjson
from fastapi.testclient import TestClient

from rosetta.app import create_app
from rosetta.config import Config
from rosetta.upstream import UpstreamClient


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
    body: bytes


class _BodyStream(httpx2.AsyncByteStream):
    """Unread response body, so the client streams it as it would from a socket."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


class FakeUpstream(httpx2.AsyncBaseTransport):
    """Transport for `UpstreamClient` that never touches the network.

    Requests are attributed to a provider by matching the URL against each
    provider's `base_url`, so `path` is relative to it (e.g. "/chat/completions").
    An unregistered request fails the test with an AssertionError.
    """

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self._routes: dict[tuple[str, str, str], _Canned | Exception] = {}
        self._bases: list[tuple[str, str]] = []

    def bind(self, config: Config) -> None:
        """Learn each provider's base URL; `build_client` calls this."""
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
        fixture: bytes,
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._routes[(provider_key, "POST", path)] = _Canned(
            status, {"content-type": "text/event-stream", **(headers or {})}, fixture
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
        return httpx2.Response(
            canned.status, headers=dict(canned.headers), stream=_BodyStream(canned.body)
        )


def build_client(config: Mapping[str, Any], fake: FakeUpstream) -> TestClient:
    """App wired to `fake`; enter it (`with build_client(...) as c:`) to run the lifespan."""
    cfg = Config.model_validate(config)
    fake.bind(cfg)
    return TestClient(create_app(cfg, upstream=UpstreamClient(cfg, transport=fake)))
