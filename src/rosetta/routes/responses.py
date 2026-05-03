"""POST /v1/responses — OpenAI Responses endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from rosetta.pipeline import handle
from rosetta.routes.deps import get_config
from rosetta.routes.schemas import ProxyRequest

router = APIRouter(prefix="/v1", tags=["responses"], dependencies=[Depends(get_config)])


@router.post("/responses", response_model=None)
async def responses(body: ProxyRequest, request: Request) -> Response:
    return await handle("openai_responses", body.model_dump(), request)
