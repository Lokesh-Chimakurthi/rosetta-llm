"""POST /v1/messages — Anthropic Messages endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from rosetta.pipeline import handle
from rosetta.routes.deps import get_config
from rosetta.routes.schemas import CountTokensResponse, ProxyRequest, error_responses
from rosetta.tokens import count_tokens_anthropic

router = APIRouter(prefix="/v1", tags=["messages"], dependencies=[Depends(get_config)])


@router.post(
    "/messages",
    response_model=None,
    summary="Create a message",
    description=(
        "Anthropic Messages request body, forwarded verbatim to `anthropic` providers "
        "(only `model` is rewritten) and translated for others. With `stream: true` the reply "
        "is Server-Sent Events. Errors use the Anthropic envelope."
    ),
    responses=error_responses("anthropic", 400, 401, 429, 502),
)
async def create_message(body: ProxyRequest, request: Request) -> Response:
    return await handle("anthropic", body.model_dump(), request)


@router.post(
    "/messages/count_tokens",
    response_model=CountTokensResponse,
    summary="Count input tokens",
    description=(
        "Approximate input token count for an Anthropic Messages body, computed locally with "
        "tiktoken `o200k_base`; real counts can differ by roughly 5-15%. No upstream call."
    ),
    responses=error_responses("anthropic", 401),
)
def count_tokens(body: ProxyRequest) -> CountTokensResponse:
    return CountTokensResponse(input_tokens=count_tokens_anthropic(body.model_dump()))
