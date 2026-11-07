"""POST /v1/chat/completions — OpenAI Chat Completions endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from rosetta.pipeline import handle
from rosetta.routes.deps import get_config
from rosetta.routes.schemas import ProxyRequest, error_responses

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(get_config)])


@router.post(
    "/chat/completions",
    response_model=None,
    summary="Create a chat completion",
    description=(
        "OpenAI Chat Completions request body, forwarded verbatim to `openai_chat` providers "
        "(only `model` is rewritten) and translated for others. With `stream: true` the reply "
        "is Server-Sent Events ending in `data: [DONE]`. Errors use the OpenAI envelope."
    ),
    responses=error_responses("openai_chat", 400, 401, 429, 502),
)
async def chat_completions(body: ProxyRequest, request: Request) -> Response:
    return await handle("openai_chat", body.model_dump(), request)
