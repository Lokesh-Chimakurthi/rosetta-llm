"""Pipeline dispatcher — passthrough fast path vs translate slow path.

Translation invariants:
  - Inbound request bytes are JSON-decoded once.
  - When inbound format == provider format, we forward verbatim (only the
    `model` field is rewritten on a copied dict; original payload untouched).
  - Otherwise we run inbound parser → IR → provider renderer → upstream call →
    provider parser → IR → inbound renderer.
  - Streaming uses per-format state machines that emit IR events; outbound
    state machine consumes IR events and emits target SSE.
  - Ping injection is added only when outbound format == anthropic.
  - Cancellation: a client disconnect aborts the streaming generator,
    which exits the upstream `httpx.stream` context and closes the
    connection.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from rosetta.codecs import anthropic as ac_codec
from rosetta.codecs import openai_chat as oc_codec
from rosetta.codecs import openai_responses as or_codec
from rosetta.config import Config, ModelConfig, ProviderConfig
from rosetta.errors import format_error, format_stream_error
from rosetta.ir.request import CanonicalRequest
from rosetta.ir.response import CanonicalResponse
from rosetta.observability import bind_request_context, get_logger
from rosetta.routes.health import record_provider_status
from rosetta.stream_codecs import anthropic as ac_stream
from rosetta.stream_codecs import openai_chat as oc_stream
from rosetta.stream_codecs import openai_responses as or_stream
from rosetta.upstream import UpstreamClient

type ParseRequest = Callable[[dict[str, Any]], CanonicalRequest]
type RenderRequest = Callable[[CanonicalRequest], dict[str, Any]]
type ParseResponse = Callable[[dict[str, Any]], CanonicalResponse]
type RenderResponse = Callable[[CanonicalResponse], dict[str, Any]]
type StreamParse = Callable[[AsyncIterator[bytes]], AsyncIterator[Any]]
type StreamRender = Callable[[AsyncIterator[Any]], AsyncIterator[bytes]]

_PARSE_REQ: dict[str, ParseRequest] = {
    "anthropic": ac_codec.parse_request,
    "openai_chat": oc_codec.parse_request,
    "openai_responses": or_codec.parse_request,
}
_RENDER_REQ: dict[str, RenderRequest] = {
    "anthropic": ac_codec.render_request,
    "openai_chat": oc_codec.render_request,
    "openai_responses": or_codec.render_request,
}
_PARSE_RESP: dict[str, ParseResponse] = {
    "anthropic": ac_codec.parse_response,
    "openai_chat": oc_codec.parse_response,
    "openai_responses": or_codec.parse_response,
}
_RENDER_RESP: dict[str, RenderResponse] = {
    "anthropic": ac_codec.render_response,
    "openai_chat": oc_codec.render_response,
    "openai_responses": or_codec.render_response,
}
_STREAM_PARSE: dict[str, StreamParse] = {
    "anthropic": ac_stream.parse,
    "openai_chat": oc_stream.parse,
    "openai_responses": or_stream.parse,
}
_STREAM_RENDER: dict[str, StreamRender] = {
    "anthropic": ac_stream.render,
    "openai_chat": oc_stream.render,
    "openai_responses": or_stream.render,
}
_FORMAT_PATH: dict[str, str] = {
    "openai_chat": "/chat/completions",
    "openai_responses": "/responses",
    "anthropic": "/messages",
}

# Anthropic headers that MUST be forwarded upstream per the Claude Code LLM Gateway spec.
_ANTHROPIC_FORWARD_HEADERS = frozenset(
    {"anthropic-beta", "anthropic-version", "x-claude-code-session-id"}
)

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _forwarded_headers(request: Request) -> dict[str, str]:
    return {k: request.headers[k] for k in _ANTHROPIC_FORWARD_HEADERS if k in request.headers}


_CLAUDE_CODE_GW_PREFIX = "claude-code/"


def _resolve_model(model_id: str, config: Config) -> tuple[ProviderConfig, str, str, str]:
    """Resolve `model_id` to (provider, provider_key, upstream_name, match_tier).

    `match_tier` is one of: "strict", "strict_passthrough", "exact_id",
    "alias", "regex", "substring". Used for diagnostic logging.
    """
    # Claude Code gateway prefix: strip and re-resolve the underlying name (which may
    # itself be either a provider/model pair or a bare shorthand).
    if model_id.startswith(_CLAUDE_CODE_GW_PREFIX):
        return _resolve_model(model_id[len(_CLAUDE_CODE_GW_PREFIX) :], config)

    # Strict path: explicit provider/model always wins.
    if "/" in model_id:
        provider_key, model_name = model_id.split("/", 1)
        provider = config.providers.get(provider_key)
        if provider is not None:
            for m in provider.static_models:
                if m.id == model_name:
                    return provider, provider_key, m.effective_upstream_name, "strict"
            # Unknown model under a known provider: pass through (auto-discover providers
            # rely on this so any upstream-listed model id resolves without static config).
            return provider, provider_key, model_name, "strict_passthrough"
        # Provider key didn't match — fall through to shorthand search so users can name
        # an alias that contains a slash (rare, but cheap to support).

    # Shorthand path: tier 1 = exact id/alias/regex, tier 2 = case-insensitive substring.
    exact: list[tuple[str, ProviderConfig, ModelConfig, str]] = []
    fuzzy: list[tuple[str, ProviderConfig, ModelConfig, str]] = []
    needle = model_id.casefold()
    for key, prov in config.providers.items():
        for m in prov.static_models:
            if m.id == model_id:
                exact.append((key, prov, m, "exact_id"))
                continue
            if model_id in m.aliases:
                exact.append((key, prov, m, "alias"))
                continue
            if m.alias_pattern is not None and re.fullmatch(m.alias_pattern, model_id):
                exact.append((key, prov, m, "regex"))
                continue
            haystacks = [m.id.casefold(), *(a.casefold() for a in m.aliases)]
            if any(needle in h for h in haystacks):
                fuzzy.append((key, prov, m, "substring"))

    matches = exact or fuzzy
    if not matches:
        raise ValueError(
            f"Model id '{model_id}' did not match any provider/model, alias, "
            f"alias_pattern, or substring of any configured model. "
            f"Use '<provider>/<model>' or configure an alias."
        )
    if len(matches) > 1:
        candidates = ", ".join(f"{k}/{m.id}" for k, _, m, _ in matches)
        tier_label = "exactly" if exact else "as a substring"
        raise ValueError(
            f"Model id '{model_id}' matches {tier_label} more than one configured model: "
            f"{candidates}. Disambiguate by using one of those explicit ids."
        )
    key, prov, m, match_tier = matches[0]
    return prov, key, m.effective_upstream_name, match_tier


async def handle(
    inbound_format: str,
    payload: dict[str, Any],
    request: Request,
) -> Response:
    config: Config = request.app.state.config
    upstream: UpstreamClient = request.app.state.upstream
    status_dict: dict[str, Any] = request.app.state.provider_status
    log = get_logger()

    model_id = payload.get("model", "")
    try:
        provider, provider_key, upstream_name, match_tier = _resolve_model(model_id, config)
    except ValueError as e:
        return JSONResponse(
            content=format_error(inbound_format, 400, "invalid_request_error", str(e)),
            status_code=400,
        )

    bind_request_context(provider=provider_key, model=model_id, format=provider.format)
    log.debug(
        "model_resolved",
        inbound_model=model_id,
        provider=provider_key,
        upstream_model=upstream_name,
        provider_format=provider.format,
        match_tier=match_tier,
    )

    # Copy and rewrite the model field without mutating the caller's dict.
    upstream_body = {**payload, "model": upstream_name}
    is_stream = bool(payload.get("stream", False))
    upstream_path = _FORMAT_PATH[provider.format]

    fwd_headers = _forwarded_headers(request)

    if inbound_format == provider.format:
        return await _passthrough(
            provider,
            provider_key,
            upstream_path,
            upstream_body,
            is_stream,
            inbound_format,
            request,
            upstream,
            log,
            fwd_headers,
            status_dict,
        )

    return await _translate(
        inbound_format,
        provider,
        provider_key,
        upstream_body,
        upstream_path,
        is_stream,
        request,
        upstream,
        log,
        fwd_headers,
        status_dict,
    )


def _safe_json(text: str) -> Any:
    try:
        import json

        return json.loads(text)
    except Exception:  # noqa: BLE001
        return {}


def _log_upstream_status(
    log: Any,
    provider_key: str,
    provider: ProviderConfig,
    status_code: int,
    body_text: str,
) -> None:
    """Emit a structured non-success record. INFO summary, DEBUG full body."""
    summary = _extract_error_message(_safe_json(body_text)) if body_text else ""
    fields: dict[str, Any] = {
        "provider": provider_key,
        "status": status_code,
        "summary": summary,
    }
    event = "upstream_status"
    if status_code == 401:
        event = "upstream_auth_failed"
        fields.update(provider.auth_source)
    severity = log.warning if status_code < 500 else log.error
    severity(event, **fields)
    if body_text:
        log.debug("upstream_body", provider=provider_key, body=body_text[:2048])


def _log_stream_status_error(
    log: Any,
    provider_key: str,
    provider: ProviderConfig,
    e: httpx.HTTPError,
) -> None:
    """For streaming exceptions: log the underlying HTTPStatusError specially
    when it carries a 401, otherwise fall through to the generic line."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status == 401:
        _log_upstream_status(log, provider_key, provider, status, str(e))
    else:
        log.error("stream_upstream_error", error=str(e))


async def _passthrough(
    provider: ProviderConfig,
    provider_key: str,
    upstream_path: str,
    body: dict[str, Any],
    is_stream: bool,
    inbound_format: str,
    request: Request,
    upstream: UpstreamClient,
    log: Any,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
) -> Response:
    log.info("passthrough", stream=is_stream)
    if is_stream:
        gen = upstream.stream(provider_key, upstream_path, body, extra_headers=fwd_headers)
        return StreamingResponse(
            _passthrough_stream_with_recovery(
                gen, request, provider, provider_key, inbound_format, status_dict
            ),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )
    try:
        resp = await upstream.request_json(
            provider_key, upstream_path, body, extra_headers=fwd_headers
        )
        record_provider_status(status_dict, provider_key, ok=resp.is_success)
    except httpx.HTTPError as e:
        log.error("upstream_error", error=str(e))
        record_provider_status(status_dict, provider_key, ok=False)
        return JSONResponse(
            content=format_error(inbound_format, 502, "upstream_error", str(e)),
            status_code=502,
        )
    if not resp.is_success:
        _log_upstream_status(log, provider_key, provider, resp.status_code, resp.text)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


async def _passthrough_stream_with_recovery(
    gen: AsyncIterator[bytes],
    request: Request,
    provider: ProviderConfig,
    provider_key: str,
    inbound_format: str,
    status_dict: dict[str, Any],
) -> AsyncIterator[bytes]:
    log = get_logger()
    try:
        async for chunk in gen:
            if await request.is_disconnected():
                break
            yield chunk
        record_provider_status(status_dict, provider_key, ok=True)
    except httpx.HTTPError as e:
        _log_stream_status_error(log, provider_key, provider, e)
        record_provider_status(status_dict, provider_key, ok=False)
        yield format_stream_error(inbound_format, "upstream_error", str(e))


async def _translate(
    inbound_format: str,
    provider: ProviderConfig,
    provider_key: str,
    payload: dict[str, Any],
    upstream_path: str,
    is_stream: bool,
    request: Request,
    upstream: UpstreamClient,
    log: Any,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
) -> Response:
    provider_format = provider.format
    log.info(
        "translate", inbound=inbound_format, provider_format=provider_format, stream=is_stream
    )

    try:
        ir = _PARSE_REQ[inbound_format](payload)
        upstream_body = _RENDER_REQ[provider_format](ir)
    except Exception as e:  # noqa: BLE001
        log.error("translate_request_error", error=str(e))
        return JSONResponse(
            content=format_error(inbound_format, 400, "translation_error", str(e)),
            status_code=400,
        )

    if is_stream:
        return StreamingResponse(
            _translate_stream(
                inbound_format,
                provider_format,
                upstream_body,
                provider,
                provider_key,
                upstream_path,
                request,
                upstream,
                fwd_headers,
                status_dict,
            ),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    try:
        resp = await upstream.request_json(
            provider_key, upstream_path, upstream_body, extra_headers=fwd_headers
        )
    except httpx.HTTPError as e:
        log.error("upstream_error", error=str(e))
        record_provider_status(status_dict, provider_key, ok=False)
        return JSONResponse(
            content=format_error(inbound_format, 502, "upstream_error", str(e)),
            status_code=502,
        )

    if not resp.is_success:
        _log_upstream_status(log, provider_key, provider, resp.status_code, resp.text)
        record_provider_status(status_dict, provider_key, ok=False)
        try:
            upstream_err = resp.json()
        except Exception:  # noqa: BLE001
            upstream_err = {"raw": resp.text}
        return JSONResponse(
            content=format_error(
                inbound_format,
                resp.status_code,
                "upstream_error",
                _extract_error_message(upstream_err),
            ),
            status_code=resp.status_code,
        )

    record_provider_status(status_dict, provider_key, ok=True)
    try:
        upstream_data = resp.json()
        ir_resp = _PARSE_RESP[provider_format](upstream_data)
        out = _RENDER_RESP[inbound_format](ir_resp)
    except Exception as e:  # noqa: BLE001
        log.error("translate_response_error", error=str(e))
        return JSONResponse(
            content=format_error(inbound_format, 502, "translation_error", str(e)),
            status_code=502,
        )
    return JSONResponse(content=out)


async def _translate_stream(
    inbound_format: str,
    provider_format: str,
    payload: dict[str, Any],
    provider: ProviderConfig,
    provider_key: str,
    upstream_path: str,
    request: Request,
    upstream: UpstreamClient,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
) -> AsyncIterator[bytes]:
    log = get_logger()
    try:
        upstream_bytes = upstream.stream(
            provider_key, upstream_path, payload, extra_headers=fwd_headers
        )
        ir_events = _STREAM_PARSE[provider_format](upstream_bytes)
        if inbound_format == "anthropic":
            ir_events = ac_stream.wrap_with_ping(ir_events)
        out_bytes = _STREAM_RENDER[inbound_format](ir_events)

        async for chunk in out_bytes:
            if await request.is_disconnected():
                break
            yield chunk
        record_provider_status(status_dict, provider_key, ok=True)
    except httpx.HTTPError as e:
        _log_stream_status_error(log, provider_key, provider, e)
        record_provider_status(status_dict, provider_key, ok=False)
        yield format_stream_error(inbound_format, "upstream_error", str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("stream_translate_error")
        yield format_stream_error(inbound_format, "translation_error", str(e))


def _extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and "message" in err:
            return str(err["message"])
        if isinstance(err, str):
            return err
        if "message" in payload:
            return str(payload["message"])
    return str(payload)
