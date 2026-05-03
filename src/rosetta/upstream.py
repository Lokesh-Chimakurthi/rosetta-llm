"""httpx.AsyncClient pool per provider with auth header injection."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import orjson

from rosetta.config import ProviderConfig


class UpstreamClient:
    """Manages one httpx.AsyncClient per provider, keyed by provider key."""

    def __init__(self, providers: dict[str, ProviderConfig]) -> None:
        self._providers = providers
        self._clients: dict[str, httpx.AsyncClient] = {}

    async def start(self) -> None:
        for key, prov in self._providers.items():
            headers = dict(prov.extra_headers)
            if prov.format == "anthropic":
                headers["x-api-key"] = prov.api_key or ""
                headers.setdefault("anthropic-version", "2023-06-01")
            else:
                headers["Authorization"] = f"Bearer {prov.api_key or ''}"
            headers.setdefault("user-agent", "rosetta/0.1")

            self._clients[key] = httpx.AsyncClient(
                base_url=prov.base_url.rstrip("/"),
                headers=headers,
                timeout=httpx.Timeout(
                    connect=prov.timeout.connect,
                    read=prov.timeout.read,
                    write=prov.timeout.read,
                    pool=10.0,
                ),
                http2=False,
            )

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    def get_client(self, provider_key: str) -> httpx.AsyncClient:
        client = self._clients.get(provider_key)
        if client is None:
            raise ValueError(f"Unknown provider: {provider_key}")
        return client

    async def request_json(
        self,
        provider_key: str,
        path: str,
        body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        client = self.get_client(provider_key)
        content = orjson.dumps(body) if body is not None else None
        headers = {"content-type": "application/json"} if content else {}
        if extra_headers:
            headers.update(extra_headers)
        return await client.post(path, content=content, headers=headers)

    async def stream(
        self,
        provider_key: str,
        path: str,
        body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        client = self.get_client(provider_key)
        content = orjson.dumps(body) if body is not None else None
        headers = {"content-type": "application/json", "accept": "text/event-stream"}
        if extra_headers:
            headers.update(extra_headers)
        async with client.stream("POST", path, content=content, headers=headers) as response:
            if response.status_code >= 400:
                # Surface upstream error body as a single chunk; the stream
                # codecs will translate it into an error event downstream.
                err_body = await response.aread()
                raise httpx.HTTPStatusError(
                    f"Upstream returned {response.status_code}: {err_body.decode(errors='replace')[:512]}",
                    request=response.request,
                    response=response,
                )
            async for chunk in response.aiter_raw():
                yield chunk
