"""POST /v1/chat/completions — OpenAI Chat Completions endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from rosetta.pipeline import handle
from rosetta.routes.deps import get_config
from rosetta.routes.schemas import ProxyRequest

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(get_config)])


@router.post("/chat/completions", response_model=None)
async def chat_completions(body: ProxyRequest, request: Request) -> Response:
    return await handle("openai_chat", body.model_dump(), request)
