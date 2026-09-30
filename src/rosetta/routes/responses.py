"""POST /v1/responses — OpenAI Responses endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from rosetta.pipeline import handle
from rosetta.routes.deps import get_config
from rosetta.routes.schemas import ProxyRequest, error_responses

router = APIRouter(prefix="/v1", tags=["responses"], dependencies=[Depends(get_config)])


@router.post(
    "/responses",
    response_model=None,
    summary="Create a model response",
    description=(
        "OpenAI Responses request body, forwarded verbatim to `openai_responses` providers "
        "(only `model` is rewritten) and translated for others. With `stream: true` the reply "
        "is Server-Sent Events. Errors use the OpenAI envelope."
    ),
    responses=error_responses("openai_responses", 400, 401, 429, 502),
)
async def responses(body: ProxyRequest, request: Request) -> Response:
    return await handle("openai_responses", body.model_dump(), request)
