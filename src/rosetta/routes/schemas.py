"""Request and response Pydantic models for route endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from rosetta.config import Format


class ProxyRequest(BaseModel):
    """Shared request model for chat/completions, messages, and responses endpoints.

    Only fields needed for routing are declared explicitly; all format-specific
    fields are preserved via extra="allow" for passthrough fidelity.
    """

    model: str | None = None
    stream: bool = False
    model_config = ConfigDict(extra="allow")


class HealthResponse(BaseModel):
    status: str
    message: str | None = None


class ProviderStatusItem(BaseModel):
    key: str
    format: str
    last_status: str
    last_check_ts: float | None


class ProvidersResponse(BaseModel):
    providers: list[ProviderStatusItem]


class CountTokensResponse(BaseModel):
    input_tokens: int


class OpenAIModelListResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]


class AnthropicModelPageResponse(BaseModel):
    data: list[dict[str, Any]]
    has_more: bool
    first_id: str
    last_id: str


class AnthropicErrorDetail(BaseModel):
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error envelope for Anthropic Messages endpoints."""

    type: Literal["error"] = "error"
    error: AnthropicErrorDetail


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None
    param: str | None = None


class OpenAIErrorResponse(BaseModel):
    """Error envelope for OpenAI Chat Completions and Responses endpoints."""

    error: OpenAIErrorDetail


_STATUS_DESCRIPTIONS: dict[int, str] = {
    400: "Model id is not `<provider>/<model>`, names an unknown provider, or the request could not be translated.",
    401: "Proxy API keys are configured and neither `Authorization: Bearer` nor `x-api-key` matched.",
    429: "Relayed from the upstream provider; any upstream error status is relayed the same way.",
    502: "The upstream provider could not be reached, or its response could not be translated.",
}


def error_responses(fmt: Format, *statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI `responses` entries documenting `statuses` with `fmt`'s envelope."""
    model = AnthropicErrorResponse if fmt == "anthropic" else OpenAIErrorResponse
    return {s: {"model": model, "description": _STATUS_DESCRIPTIONS[s]} for s in statuses}
