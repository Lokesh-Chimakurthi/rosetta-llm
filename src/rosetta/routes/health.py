"""GET /health and GET /providers — health check and provider status."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from rosetta.routes.schemas import HealthResponse, ProvidersResponse, ProviderStatusItem

router = APIRouter(tags=["health"])


def record_provider_status(status_dict: dict[str, Any], provider_key: str, ok: bool) -> None:
    status_dict[provider_key] = {
        "last_status": "ok" if ok else "error",
        "last_check_ts": time.time(),
    }


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", message="Status: healthy. Cause: you checked on me.")


@router.get("/providers", response_model=ProvidersResponse)
async def providers(request: Request) -> ProvidersResponse:
    config = request.app.state.config
    status = request.app.state.provider_status
    items: list[ProviderStatusItem] = []
    for key, prov in config.providers.items():
        pstatus = status.get(key, {"last_status": "unknown", "last_check_ts": None})
        items.append(
            ProviderStatusItem(
                key=key,
                format=prov.format,
                last_status=pstatus["last_status"],
                last_check_ts=pstatus["last_check_ts"],
            )
        )
    return ProvidersResponse(providers=items)
