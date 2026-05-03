"""GET /v1/models — Merged model list with TTL cache for auto-discovery.

Models are pre-fetched on startup and refreshed in the background so the
endpoint responds instantly (< 5ms) — critical for Claude Code's gateway
discovery which has a tight timeout.

Serves both OpenAI and Anthropic response shapes. Format auto-detected by
the presence of the `anthropic-version` request header.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse

from rosetta.config import ProviderConfig
from rosetta.observability import get_logger
from rosetta.routes.deps import ConfigDep
from rosetta.routes.schemas import AnthropicModelPageResponse, OpenAIModelListResponse
from rosetta.upstream import UpstreamClient

router = APIRouter(prefix="/v1", tags=["models"])


def _pick[T](*values: T | None) -> T | None:
    for v in values:
        if v is not None:
            return v
    return None


def _synthesize_caps(m: dict[str, Any]) -> dict[str, Any]:
    arch = m.get("architecture", {}) or {}
    input_mods = set(arch.get("input_modalities", []) or [])
    supported = set(m.get("supported_parameters", []) or [])
    has_image = "image" in input_mods
    has_thinking = any(k in supported for k in ("reasoning", "include_reasoning", "thinking"))
    has_structured = "structured_outputs" in supported or "response_format" in supported
    return {
        "image_input": {"supported": has_image},
        "pdf_input": {"supported": False},
        "thinking": {
            "supported": has_thinking,
            "types": {
                "enabled": {"supported": has_thinking},
                "adaptive": {"supported": has_thinking},
            },
        },
        "effort": {
            "supported": has_thinking,
            "low": {"supported": has_thinking},
            "medium": {"supported": has_thinking},
            "high": {"supported": has_thinking},
            "xhigh": {"supported": has_thinking},
            "max": {"supported": False},
        },
        "structured_outputs": {"supported": has_structured},
    }


async def _fetch_page(
    client: Any,
    path: str,
    after_id: str | None = None,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    params: dict[str, str] = {}
    if after_id:
        params["after_id"] = after_id
    resp = await client.get(path, params=params or None)
    resp.raise_for_status()
    data = resp.json()
    entries: list[dict[str, Any]] = []
    for m in data.get("data", []):
        if not isinstance(m, dict) or not m.get("id"):
            continue
        entry: dict[str, Any] = {"id": m["id"]}
        display_name = m.get("display_name") or m.get("name")
        if isinstance(display_name, str):
            entry["display_name"] = display_name
        entry["type"] = "model"
        max_tok = m.get("max_tokens")
        if max_tok is None and isinstance(m.get("top_provider"), dict):
            max_tok = m["top_provider"].get("max_completion_tokens")
        if max_tok is not None:
            entry["max_tokens"] = max_tok
        max_in = m.get("max_input_tokens") or m.get("context_length")
        if max_in is None and isinstance(m.get("top_provider"), dict):
            max_in = m["top_provider"].get("context_length")
        if max_in is not None:
            entry["max_input_tokens"] = max_in
            entry["context_length"] = max_in
        created_at = m.get("created_at")
        created_epoch = m.get("created")
        if created_at is None and isinstance(created_epoch, (int, float)):
            created_at = datetime.fromtimestamp(created_epoch, tz=UTC).isoformat()
        if created_at is not None:
            entry["created_at"] = str(created_at)
        if isinstance(created_epoch, (int, float)):
            entry["created"] = int(created_epoch)
        caps = m.get("capabilities")
        if isinstance(caps, dict):
            entry["capabilities"] = caps
        elif isinstance(m.get("architecture"), dict) or isinstance(
            m.get("supported_parameters"), list
        ):
            entry["capabilities"] = _synthesize_caps(m)
        entries.append(entry)
    return entries, bool(data.get("has_more", False)), data.get("last_id")


async def _try_fetch(client: Any, path: str) -> list[dict[str, Any]] | None:
    try:
        entries, has_more, last_id = await _fetch_page(client, path)
        while has_more and last_id:
            page, has_more, last_id = await _fetch_page(client, path, after_id=last_id)
            entries.extend(page)
        return entries if entries else None
    except Exception:  # noqa: BLE001
        return None


def _anthropic_capabilities(entry: dict[str, Any], prov: ProviderConfig) -> dict[str, Any]:
    caps: dict[str, Any] = entry.get("capabilities", {}) or {}
    mc = next((m for m in prov.static_models if m.id == entry["id"]), None)

    def _supported(key: str) -> bool:
        if caps.get(key, {}).get("supported") is True:
            return True
        if mc and mc.supports:
            return getattr(mc.supports, key, True)
        return True

    return {
        "batch": {"supported": False},
        "citations": {"supported": False},
        "code_execution": {"supported": False},
        "context_management": {
            "supported": False,
            "clear_thinking_20251015": {"supported": False},
            "clear_tool_uses_20250919": {"supported": False},
            "compact_20260112": {"supported": False},
        },
        "effort": {
            "supported": _supported("thinking"),
            "low": {"supported": _supported("thinking")},
            "medium": {"supported": _supported("thinking")},
            "high": {"supported": _supported("thinking")},
            "xhigh": {"supported": _supported("thinking")},
            "max": {"supported": False},
        },
        "image_input": {"supported": _supported("vision")},
        "pdf_input": {"supported": _supported("vision")},
        "structured_outputs": {"supported": _supported("structured_output")},
        "thinking": {
            "supported": _supported("thinking"),
            "types": {
                "enabled": {"supported": _supported("thinking")},
                "adaptive": {"supported": _supported("thinking")},
            },
        },
    }


def _build_snapshot(
    config: Any, entries_by_provider: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pre-build all response shapes from raw upstream entries. Runs at refresh time only."""
    openai_models: list[dict[str, Any]] = []
    anthropic_models: list[dict[str, Any]] = []

    for key, provider in config.providers.items():
        entries = entries_by_provider.get(key, [])
        if not entries:
            # Static fallback when upstream fetch failed or provider uses static models.
            if provider.is_auto_discover:
                continue
            entries = [
                {
                    "id": m.id,
                    "display_name": m.id,
                    "max_tokens": m.thinking_budget_default * 4,
                    "max_input_tokens": 200000,
                }
                for m in provider.static_models
            ]
        for entry in entries:
            caps = entry.get("capabilities", {}) or {}
            raw_id = f"{key}/{entry['id']}"
            anthropic_id = raw_id

            anthropic_models.append(
                {
                    "id": anthropic_id,
                    "type": "model",
                    "display_name": _pick(entry.get("display_name"), entry["id"]),
                    "created_at": _pick(entry.get("created_at"), "2000-01-01T00:00:00Z"),
                    "max_input_tokens": _pick(entry.get("max_input_tokens"), 200000),
                    "max_tokens": _pick(entry.get("max_tokens"), 32000),
                    "capabilities": _anthropic_capabilities(entry, provider),
                }
            )

            oai_model: dict[str, Any] = {"id": raw_id, "object": "model", "owned_by": key}
            if "display_name" in entry:
                oai_model["display_name"] = entry["display_name"]
            if "created" in entry:
                oai_model["created"] = entry["created"]
            if "context_length" in entry:
                oai_model["context_length"] = entry["context_length"]
            elif "max_input_tokens" in entry:
                oai_model["context_length"] = entry["max_input_tokens"]
            if "max_tokens" in entry:
                oai_model["max_tokens"] = entry["max_tokens"]
            if caps:
                input_mods = ["text"]
                if caps.get("image_input", {}).get("supported"):
                    input_mods.append("image")
                oai_model["architecture"] = {
                    "modality": "+".join(input_mods) + "->text",
                    "input_modalities": input_mods,
                    "output_modalities": ["text"],
                }
                supported_params = ["tools", "tool_choice", "stop", "seed"]
                if caps.get("structured_outputs", {}).get("supported"):
                    supported_params.extend(["structured_outputs", "response_format"])
                if caps.get("thinking", {}).get("supported"):
                    supported_params.extend(["reasoning", "include_reasoning"])
                if caps.get("effort", {}).get("supported"):
                    supported_params.extend(["max_tokens", "temperature", "top_p"])
                oai_model["supported_parameters"] = sorted(set(supported_params))
            openai_models.append(oai_model)

    claude_code_models = [
        {**m, "id": f"claude-code/{m['id']}"}
        if not m["id"].startswith(("claude", "anthropic"))
        else m
        for m in anthropic_models
    ]

    return openai_models, anthropic_models, claude_code_models


async def _fetch_all_providers(
    config: Any,
    upstream: UpstreamClient,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch models from all auto-discovery providers in parallel."""
    entries_by_provider: dict[str, list[dict[str, Any]]] = {}

    async def _fetch_one(key: str, provider: ProviderConfig) -> None:
        if provider.is_auto_discover:
            try:
                client = upstream.get_client(key)
                entries = await _try_fetch(client, "/models") or await _try_fetch(
                    client, "/v1/models"
                )
                if entries:
                    entries_by_provider[key] = entries
            except Exception:
                get_logger().exception("provider_fetch_failed", provider=key)
        else:
            entries_by_provider[key] = []

    tasks = [asyncio.create_task(_fetch_one(k, p)) for k, p in config.providers.items()]
    await asyncio.gather(*tasks, return_exceptions=True)
    return entries_by_provider


async def refresh_models(config: Any, upstream: UpstreamClient, snapshot: dict[str, Any]) -> None:
    """Fetch from all upstreams and rebuild the in-memory snapshot."""
    log = get_logger()
    started = time.monotonic()
    try:
        entries_by_provider = await _fetch_all_providers(config, upstream)
        oai, ant, cc = _build_snapshot(config, entries_by_provider)
        snapshot["openai"] = oai
        snapshot["anthropic"] = ant
        snapshot["claude_code"] = cc
        snapshot["last_refresh"] = time.monotonic()
        total = len(oai)
        elapsed = time.monotonic() - started
        log.info("models_refreshed", total=total, elapsed_ms=round(elapsed * 1000))
    except Exception:  # noqa: BLE001
        log.exception("models_refresh_failed")


def _snapshot(
    snapshot: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        cast(list[dict[str, Any]], snapshot.get("openai", [])),
        cast(list[dict[str, Any]], snapshot.get("anthropic", [])),
        cast(list[dict[str, Any]], snapshot.get("claude_code", [])),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/models")
async def get_models(
    request: Request,
    response: Response,
    config: ConfigDep,
    after_id: Annotated[str | None, Query()] = None,
    before_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 20,
) -> AnthropicModelPageResponse | OpenAIModelListResponse:
    response.headers["Cache-Control"] = "public, max-age=60"
    is_anthropic = "anthropic-version" in request.headers
    is_claude_code = "x-claude-code-session-id" in request.headers
    snapshot = request.app.state.model_snapshot
    oai, ant, cc = _snapshot(snapshot)

    if is_anthropic and is_claude_code:
        models = cc
    elif is_anthropic:
        models = ant
    else:
        models = oai

    if is_anthropic:
        return _anthropic_page(list(models), after_id, before_id, limit)

    return OpenAIModelListResponse(object="list", data=models)


@router.post("/models/refresh")
async def refresh(config: ConfigDep, request: Request) -> JSONResponse:
    upstream = cast(UpstreamClient, request.app.state.upstream)
    async with request.app.state.model_refresh_lock:
        await refresh_models(config, upstream, request.app.state.model_snapshot)
    *_, cc = _snapshot(request.app.state.model_snapshot)
    return JSONResponse(content={"refreshed": True, "models": len(cc)})


def _anthropic_page(
    models: list[dict[str, Any]],
    after_id: str | None,
    before_id: str | None,
    limit: int,
) -> AnthropicModelPageResponse:
    paged = models
    if after_id:
        for i, m in enumerate(paged):
            if m["id"] == after_id:
                paged = paged[i + 1 :]
                break
    if before_id:
        for i, m in enumerate(paged):
            if m["id"] == before_id:
                paged = paged[:i]
                break
    has_more = len(paged) > limit
    paged = paged[:limit]
    return AnthropicModelPageResponse(
        data=paged,
        has_more=has_more,
        first_id=paged[0]["id"] if paged else "",
        last_id=paged[-1]["id"] if paged else "",
    )
