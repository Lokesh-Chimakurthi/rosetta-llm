"""POST /v1/messages — Anthropic Messages endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from rosetta.pipeline import handle
from rosetta.routes.deps import get_config
from rosetta.routes.schemas import CountTokensResponse, ProxyRequest
from rosetta.tokens import count_tokens_anthropic

router = APIRouter(prefix="/v1", tags=["messages"], dependencies=[Depends(get_config)])


@router.post("/messages", response_model=None)
async def create_message(body: ProxyRequest, request: Request) -> Response:
    return await handle("anthropic", body.model_dump(), request)


@router.post("/messages/count_tokens", response_model=CountTokensResponse)
def count_tokens(body: ProxyRequest) -> CountTokensResponse:
    return CountTokensResponse(input_tokens=count_tokens_anthropic(body.model_dump()))
