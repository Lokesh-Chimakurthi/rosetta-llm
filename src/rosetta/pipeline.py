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
    which exits the `UpstreamClient.stream` context and closes the
    connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import orjson
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from rosetta.codecs import anthropic as ac_codec
from rosetta.codecs import openai_chat as oc_codec
from rosetta.codecs import openai_responses as or_codec
from rosetta.config import Config, ProviderConfig
from rosetta.errors import format_error, format_stream_error
from rosetta.ir.events import MessageDeltaEvent
from rosetta.ir.request import CanonicalRequest
from rosetta.ir.response import CanonicalResponse, StopInfo, Usage
from rosetta.observability import RequestLog, get_logger, request_log
from rosetta.routes.health import record_provider_status
from rosetta.stream_codecs import anthropic as ac_stream
from rosetta.stream_codecs import openai_chat as oc_stream
from rosetta.stream_codecs import openai_responses as or_stream
from rosetta.upstream import UpstreamClient, UpstreamError, UpstreamStatusError

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


def _resolve_model(model_id: str, config: Config) -> tuple[ProviderConfig, str, str]:
    # Claude Code gateway prefix: strip and re-resolve the underlying provider/model.
    if model_id.startswith(_CLAUDE_CODE_GW_PREFIX):
        inner = model_id[len(_CLAUDE_CODE_GW_PREFIX) :]
        if "/" in inner:
            return _resolve_model(inner, config)
        raise ValueError(f"Model id '{model_id}' has gateway prefix but no provider/model")
    if "/" not in model_id:
        raise ValueError(f"Model id '{model_id}' must be in format '<provider>/<model>'")
    provider_key, model_name = model_id.split("/", 1)
    provider = config.providers.get(provider_key)
    if provider is None:
        raise ValueError(f"Unknown provider '{provider_key}'")
    upstream_name = model_name
    for m in provider.static_models:
        if m.id == model_name:
            upstream_name = m.effective_upstream_name
            break
    return provider, provider_key, upstream_name


async def handle(
    inbound_format: str,
    payload: dict[str, Any],
    request: Request,
) -> Response:
    config: Config = request.app.state.config
    upstream: UpstreamClient = request.app.state.upstream
    status_dict: dict[str, Any] = request.app.state.provider_status
    log = get_logger()
    rec = request_log()
    rec.inbound_format = inbound_format

    model_id = payload.get("model", "")
    try:
        provider, provider_key, upstream_name = _resolve_model(model_id, config)
    except ValueError as e:
        rec.error_type = "invalid_request"
        return JSONResponse(
            content=format_error(inbound_format, 400, "invalid_request_error", str(e)),
            status_code=400,
        )

    # Copy and rewrite the model field without mutating the caller's dict.
    upstream_body = {**payload, "model": upstream_name}
    is_stream = bool(payload.get("stream", False))
    upstream_path = _FORMAT_PATH[provider.format]

    rec.provider = provider_key
    rec.model = model_id
    rec.provider_format = provider.format
    rec.upstream_model = upstream_name
    rec.stream = is_stream
    rec.route = "passthrough" if inbound_format == provider.format else "translate"

    fwd_headers = _forwarded_headers(request)

    if inbound_format == provider.format:
        return await _passthrough(
            provider_key,
            upstream_path,
            upstream_body,
            is_stream,
            inbound_format,
            request,
            upstream,
            fwd_headers,
            status_dict,
        )

    return await _translate(
        inbound_format,
        provider.format,
        upstream_body,
        provider_key,
        upstream_path,
        is_stream,
        request,
        upstream,
        log,
        fwd_headers,
        status_dict,
    )


async def _passthrough(
    provider_key: str,
    upstream_path: str,
    body: dict[str, Any],
    is_stream: bool,
    inbound_format: str,
    request: Request,
    upstream: UpstreamClient,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
) -> Response:
    body_bytes = orjson.dumps(body)
    if is_stream:
        return StreamingResponse(
            _passthrough_stream_with_recovery(
                upstream,
                provider_key,
                upstream_path,
                body_bytes,
                fwd_headers,
                request,
                inbound_format,
                status_dict,
            ),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )
    try:
        resp = await upstream.post(provider_key, upstream_path, body_bytes, fwd_headers)
        record_provider_status(status_dict, provider_key, ok=resp.ok)
    except UpstreamError as e:
        request_log().error_type = f"upstream_{e.kind}"
        record_provider_status(status_dict, provider_key, ok=False)
        return JSONResponse(
            content=format_error(inbound_format, 502, "upstream_error", str(e)),
            status_code=502,
        )
    if not resp.ok:
        request_log().error_type = "upstream_status"
    return Response(
        content=resp.body,
        status_code=resp.status,
        media_type=resp.headers.get("content-type", "application/json"),
    )


def _stream_error_type(err: UpstreamError | UpstreamStatusError) -> str:
    return "upstream_status" if isinstance(err, UpstreamStatusError) else f"upstream_{err.kind}"


async def _count_in(chunks: AsyncIterator[bytes], rec: RequestLog) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        rec.chunks_in += 1
        yield chunk


def _record_usage(rec: RequestLog, usage: Usage, stop: StopInfo | None) -> None:
    rec.input_tokens = usage.input_tokens
    rec.output_tokens = usage.output_tokens
    if rec.debug:
        rec.cache_read_input_tokens = usage.cache_read_input_tokens
        rec.cache_creation_input_tokens = usage.cache_creation_input_tokens
        rec.reasoning_tokens = usage.reasoning_tokens
        if stop is not None and stop.provider_raw:
            rec.raw_stop_reason = stop.provider_raw


async def _usage_tap(events: AsyncIterator[Any], rec: RequestLog) -> AsyncIterator[Any]:
    async for event in events:
        if isinstance(event, MessageDeltaEvent) and event.usage is not None:
            _record_usage(rec, event.usage, event.stop)
        yield event


def _stream_error_message(err: UpstreamError | UpstreamStatusError) -> str:
    if isinstance(err, UpstreamStatusError):
        body = err.response.body.decode(errors="replace")[:512]
        return f"Upstream returned {err.response.status}: {body}"
    return str(err)


async def _passthrough_stream_with_recovery(
    upstream: UpstreamClient,
    provider_key: str,
    upstream_path: str,
    body: bytes,
    fwd_headers: dict[str, str],
    request: Request,
    inbound_format: str,
    status_dict: dict[str, Any],
) -> AsyncIterator[bytes]:
    rec = request_log()
    try:
        async with upstream.stream(provider_key, upstream_path, body, fwd_headers) as stream:
            chunks = stream.iter_bytes()
            if rec.debug:
                chunks = _count_in(chunks, rec)
            async for chunk in chunks:
                if await request.is_disconnected():
                    rec.client_disconnected = True
                    break
                if rec.debug:
                    rec.chunks_out += 1
                yield chunk
        record_provider_status(status_dict, provider_key, ok=True)
    except (UpstreamError, UpstreamStatusError) as e:
        rec.error_type = _stream_error_type(e)
        record_provider_status(status_dict, provider_key, ok=False)
        yield format_stream_error(inbound_format, "upstream_error", _stream_error_message(e))


async def _translate(
    inbound_format: str,
    provider_format: str,
    payload: dict[str, Any],
    provider_key: str,
    upstream_path: str,
    is_stream: bool,
    request: Request,
    upstream: UpstreamClient,
    log: Any,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
) -> Response:
    try:
        ir = _PARSE_REQ[inbound_format](payload)
        upstream_body = _RENDER_REQ[provider_format](ir)
    except Exception as e:
        request_log().error_type = "translation_error"
        log.exception("translate_request_error")
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
        resp = await upstream.post(
            provider_key, upstream_path, orjson.dumps(upstream_body), fwd_headers
        )
    except UpstreamError as e:
        request_log().error_type = f"upstream_{e.kind}"
        record_provider_status(status_dict, provider_key, ok=False)
        return JSONResponse(
            content=format_error(inbound_format, 502, "upstream_error", str(e)),
            status_code=502,
        )

    if not resp.ok:
        request_log().error_type = "upstream_status"
        record_provider_status(status_dict, provider_key, ok=False)
        try:
            upstream_err = resp.json()
        except ValueError:
            upstream_err = {"raw": resp.body.decode(errors="replace")}
        return JSONResponse(
            content=format_error(
                inbound_format,
                resp.status,
                "upstream_error",
                _extract_error_message(upstream_err),
            ),
            status_code=resp.status,
        )

    record_provider_status(status_dict, provider_key, ok=True)
    try:
        upstream_data = resp.json()
        ir_resp = _PARSE_RESP[provider_format](upstream_data)
        out = _RENDER_RESP[inbound_format](ir_resp)
    except Exception as e:
        request_log().error_type = "translation_error"
        log.exception("translate_response_error")
        return JSONResponse(
            content=format_error(inbound_format, 502, "translation_error", str(e)),
            status_code=502,
        )
    _record_usage(request_log(), ir_resp.usage, ir_resp.stop)
    return JSONResponse(content=out)


async def _translate_stream(
    inbound_format: str,
    provider_format: str,
    payload: dict[str, Any],
    provider_key: str,
    upstream_path: str,
    request: Request,
    upstream: UpstreamClient,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
) -> AsyncIterator[bytes]:
    log = get_logger()
    rec = request_log()
    try:
        async with upstream.stream(
            provider_key, upstream_path, orjson.dumps(payload), fwd_headers
        ) as stream:
            chunks = stream.iter_bytes()
            if rec.debug:
                chunks = _count_in(chunks, rec)
            ir_events = _usage_tap(_STREAM_PARSE[provider_format](chunks), rec)
            if inbound_format == "anthropic":
                ir_events = ac_stream.wrap_with_ping(ir_events)
            out_bytes = _STREAM_RENDER[inbound_format](ir_events)

            async for chunk in out_bytes:
                if await request.is_disconnected():
                    rec.client_disconnected = True
                    break
                if rec.debug:
                    rec.chunks_out += 1
                yield chunk
        record_provider_status(status_dict, provider_key, ok=True)
    except (UpstreamError, UpstreamStatusError) as e:
        rec.error_type = _stream_error_type(e)
        record_provider_status(status_dict, provider_key, ok=False)
        yield format_stream_error(inbound_format, "upstream_error", _stream_error_message(e))
    except Exception as e:
        rec.error_type = "translation_error"
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
