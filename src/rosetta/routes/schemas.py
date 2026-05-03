"""Request and response Pydantic models for route endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


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
