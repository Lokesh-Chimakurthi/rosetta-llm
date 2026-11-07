"""UpstreamClient contract: the seam every pipeline path and later plan relies on."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pytest

from rosetta import __version__
from rosetta.config import Config
from rosetta.upstream import UpstreamClient, UpstreamError, UpstreamStatusError
from tests.support.upstream import FakeUpstream, build_client

_CONFIG: dict[str, Any] = {
    "providers": {
        "abc": {
            "format": "openai_chat",
            "base_url": "https://secret-host.test/v1",
            "api_key": "sk-up",
            "models": [{"id": "m1"}],
        },
    },
}


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
async def upstream(fake: FakeUpstream) -> AsyncIterator[UpstreamClient]:
    cfg = Config.model_validate(_CONFIG)
    fake.bind(cfg)
    client = UpstreamClient(cfg, transport=fake)
    await client.start()
    yield client
    await client.close()


async def test_stream_error_status_raises_on_enter_with_capped_body(
    upstream: UpstreamClient, fake: FakeUpstream
) -> None:
    fake.add_sse("abc", "/chat/completions", b"x" * (2 * 1024 * 1024), status=503)
    with pytest.raises(UpstreamStatusError) as info:
        async with upstream.stream("abc", "/chat/completions", b"{}", {}):
            pytest.fail("body must not be reachable for an error status")
    assert info.value.response.status == 503
    assert len(info.value.response.body) == 1024 * 1024


async def test_stream_requests_identity_encoding_and_identify_rosetta(
    upstream: UpstreamClient, fake: FakeUpstream
) -> None:
    fake.add_sse("abc", "/chat/completions", b"data: [DONE]\n\n")
    async with upstream.stream("abc", "/chat/completions", b"{}", {}) as stream:
        assert [chunk async for chunk in stream.iter_bytes()] == [b"data: [DONE]\n\n"]
    sent = fake.requests[-1].headers
    assert sent["accept-encoding"] == "identity"
    assert sent["user-agent"] == f"rosetta/{__version__}"


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (httpx2.ConnectError("connect to https://secret-host.test failed"), "connect"),
        (httpx2.ReadTimeout("read from secret-host.test timed out"), "timeout"),
        (httpx2.RemoteProtocolError("secret-host.test closed the connection"), "protocol"),
    ],
)
async def test_transport_failures_map_to_kind_without_leaking_host(
    upstream: UpstreamClient, fake: FakeUpstream, exc: Exception, kind: str
) -> None:
    fake.add_error("abc", "POST", "/chat/completions", exc)
    with pytest.raises(UpstreamError) as info:
        await upstream.post("abc", "/chat/completions", b"{}", {})
    assert info.value.kind == kind
    assert info.value.provider_key == "abc"
    assert "secret-host" not in str(info.value)


def test_transport_failure_reaches_client_as_502_without_host(fake: FakeUpstream) -> None:
    fake.add_error("abc", "POST", "/chat/completions", httpx2.ConnectError("secret-host.test"))
    with build_client(_CONFIG, fake) as c:
        r = c.post("/v1/chat/completions", json={"model": "abc/m1", "messages": []})
    assert r.status_code == 502
    assert "secret-host" not in r.text


def test_translated_stream_upstream_error_status_becomes_error_event(
    fake: FakeUpstream,
) -> None:
    fake.add_sse("abc", "/chat/completions", b'{"error":{"message":"slow down"}}', status=429)
    with (
        build_client(_CONFIG, fake) as c,
        c.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "abc/m1",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r,
    ):
        body = b"".join(r.iter_bytes())
    assert r.status_code == 200
    assert b"event: error" in body
    assert b"Upstream returned 429" in body
    assert b"slow down" in body
