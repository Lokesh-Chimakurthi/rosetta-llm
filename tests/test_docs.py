"""OpenAPI documentation contract."""

from __future__ import annotations

from typing import Any

import pytest

from rosetta import __version__
from tests.support.upstream import FakeUpstream, build_client

_CONFIG: dict[str, Any] = {
    "providers": {
        "chat": {
            "format": "openai_chat",
            "base_url": "https://chat.test/v1",
            "api_key": "k",
            "models": [{"id": "m"}],
        },
    },
}
# /health cannot fail, so it is the one operation without error responses.
_NO_ERROR_RESPONSES = frozenset({("get", "/health")})


@pytest.fixture
def schema() -> dict[str, Any]:
    with build_client(_CONFIG, FakeUpstream()) as client:
        return client.get("/openapi.json").json()


def test_api_is_titled_and_versioned(schema: dict[str, Any]) -> None:
    assert schema["info"]["title"] == "Rosetta"
    assert schema["info"]["version"] == __version__
    for needle in ("<provider>/<model>", "/v1/messages", "x-api-key", "platform.openai.com"):
        assert needle in schema["info"]["description"]


def test_every_operation_has_summary_description_and_errors(schema: dict[str, Any]) -> None:
    operations = [
        (method, path, op) for path, item in schema["paths"].items() for method, op in item.items()
    ]
    assert operations
    for method, path, op in operations:
        assert op.get("summary"), (method, path)
        assert op.get("description"), (method, path)
        errors = {code for code in op["responses"] if code[0] in "45" and code != "422"}
        if (method, path) not in _NO_ERROR_RESPONSES:
            assert errors, (method, path)


def test_error_responses_reference_their_formats_envelope(schema: dict[str, Any]) -> None:
    def envelope(path: str, code: str) -> str:
        ref = schema["paths"][path]["post"]["responses"][code]["content"]["application/json"]
        return ref["schema"]["$ref"].rsplit("/", 1)[-1]

    assert envelope("/v1/messages", "401") == "AnthropicErrorResponse"
    assert envelope("/v1/chat/completions", "401") == "OpenAIErrorResponse"
    assert envelope("/v1/responses", "502") == "OpenAIErrorResponse"


def test_redoc_is_disabled() -> None:
    with build_client(_CONFIG, FakeUpstream()) as client:
        assert client.get("/redoc").status_code == 404
