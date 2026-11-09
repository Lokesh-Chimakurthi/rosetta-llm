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

from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from rosetta.auto_mode.classifier import Classifier
from rosetta.auto_mode.review import Review, requested_session, without_review_request
from rosetta.codecs.extras import extras_for
from rosetta.codecs.registry import (
    PARSE_REQUEST,
    PARSE_RESPONSE,
    RENDER_REQUEST,
    RENDER_RESPONSE,
    STREAM_PARSE,
    STREAM_RENDER,
    UPSTREAM_PATH,
)
from rosetta.config import Config, Format, ProviderConfig
from rosetta.errors import format_error, format_stream_error
from rosetta.ir.events import MessageDeltaEvent
from rosetta.ir.response import StopInfo, Usage
from rosetta.observability import RequestLog, log_translation_failed, request_log
from rosetta.routes.health import record_provider_status
from rosetta.routing import resolve_model
from rosetta.stream_codecs import anthropic as ac_stream
from rosetta.upstream import UpstreamClient, UpstreamError, UpstreamStatusError

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


async def handle(
    inbound_format: Format,
    payload: dict[str, Any],
    request: Request,
) -> Response:
    config: Config = request.app.state.config
    upstream: UpstreamClient = request.app.state.upstream
    status_dict: dict[str, Any] = request.app.state.provider_status
    rec = request_log()

    model_id = payload.get("model", "")
    rec.model = model_id or None
    try:
        provider, provider_key, upstream_name = resolve_model(model_id, config)
    except ValueError as e:
        rec.error = "invalid_request"
        return JSONResponse(
            content=format_error(inbound_format, 400, "invalid_request_error", str(e)),
            status_code=400,
        )

    # Copy and rewrite the model field without mutating the caller's dict.
    upstream_body = {**payload, "model": upstream_name}
    is_stream = bool(payload.get("stream", False))
    upstream_path = UPSTREAM_PATH[provider.format]

    rec.upstream_model = upstream_name
    rec.stream = is_stream
    rec.route = "passthrough" if inbound_format == provider.format else "translate"
    rec.log_started()

    fwd_headers = _forwarded_headers(request)

    review: Review | None = None
    if inbound_format == "anthropic" and _rosetta_reviews(provider, inbound_format):
        session = requested_session(payload)
        if session is not None:
            classifier: Classifier | None = request.app.state.classifier
            if classifier is None:
                rec.error = "auto_mode_unreviewed"
                return JSONResponse(
                    content=format_error(
                        inbound_format, 400, "invalid_request_error", _unreviewed_message(model_id)
                    ),
                    status_code=400,
                )
            review = Review(classifier, session)
            upstream_body, fwd_headers = without_review_request(upstream_body, fwd_headers)

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
            review,
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
        fwd_headers,
        status_dict,
        review,
    )


def _rosetta_reviews(provider: ProviderConfig, inbound_format: Format) -> bool:
    """Whether Rosetta, not the provider, answers Claude Code's auto mode review."""
    return provider.format != inbound_format or provider.auto_mode_review == "rosetta"


def _unreviewed_message(model_id: str) -> str:
    # Claude Code retries without `safeguards` on a 400 that names the field:
    # https://code.claude.com/docs/en/llm-gateway-protocol#feature-pass-through
    return (
        f"safeguards: Rosetta cannot review auto mode tool calls for '{model_id}' because "
        "auto_mode.classifier_model is not set in its config; set it, or start Claude Code "
        "with CLAUDE_CODE_AUTO_MODE_SERVER=0 to use its own classifier"
    )


async def _passthrough(
    provider_key: str,
    upstream_path: str,
    body: dict[str, Any],
    is_stream: bool,
    inbound_format: Format,
    request: Request,
    upstream: UpstreamClient,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
    review: Review | None,
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
                review,
            ),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )
    try:
        resp = await upstream.post(provider_key, upstream_path, body_bytes, fwd_headers)
        record_provider_status(status_dict, provider_key, ok=resp.ok)
    except UpstreamError as e:
        request_log().error = f"upstream_{e.kind}"
        record_provider_status(status_dict, provider_key, ok=False)
        return JSONResponse(
            content=format_error(inbound_format, 502, "upstream_error", str(e)),
            status_code=502,
        )
    if not resp.ok:
        request_log().error = "upstream_status"
    elif review is not None:
        try:
            message = resp.json()
        except ValueError:
            message = None
        if isinstance(message, dict):
            return JSONResponse(
                content=await review.stamp_message(message), status_code=resp.status
            )
    return Response(
        content=resp.body,
        status_code=resp.status,
        media_type=resp.headers.get("content-type", "application/json"),
    )


def _stream_error_type(err: UpstreamError | UpstreamStatusError) -> str:
    return "upstream_status" if isinstance(err, UpstreamStatusError) else f"upstream_{err.kind}"


def _record_usage(rec: RequestLog, usage: Usage, stop: StopInfo | None) -> None:
    rec.input_tokens = usage.input_tokens
    rec.output_tokens = usage.output_tokens
    if stop is not None and stop.normalized:
        rec.stop_reason = stop.normalized
    if rec.debug:
        rec.cache_read_tokens = usage.cache_read_input_tokens
        rec.cache_creation_tokens = usage.cache_creation_input_tokens
        rec.reasoning_tokens = usage.reasoning_tokens
        if stop is not None and stop.provider_raw:
            rec.upstream_stop_reason = stop.provider_raw


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
    inbound_format: Format,
    status_dict: dict[str, Any],
    review: Review | None,
) -> AsyncIterator[bytes]:
    rec = request_log()
    try:
        async with upstream.stream(provider_key, upstream_path, body, fwd_headers) as stream:
            chunks = stream.iter_bytes()
            if review is not None:
                chunks = review.stamp_stream(chunks)
            async for chunk in chunks:
                if await request.is_disconnected():
                    rec.client_disconnected = True
                    break
                yield chunk
        record_provider_status(status_dict, provider_key, ok=True)
    except (UpstreamError, UpstreamStatusError) as e:
        rec.error = _stream_error_type(e)
        record_provider_status(status_dict, provider_key, ok=False)
        yield format_stream_error(inbound_format, "upstream_error", _stream_error_message(e))


async def _translate(
    inbound_format: Format,
    provider_format: Format,
    payload: dict[str, Any],
    provider_key: str,
    upstream_path: str,
    is_stream: bool,
    request: Request,
    upstream: UpstreamClient,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
    review: Review | None,
) -> Response:
    try:
        ir = PARSE_REQUEST[inbound_format](payload)
        upstream_body = RENDER_REQUEST[provider_format](ir)
    except Exception as e:  # noqa: BLE001 - any codec failure on client input becomes a 400; logged
        log_translation_failed("request", e)
        return JSONResponse(
            content=format_error(inbound_format, 400, "translation_error", str(e)),
            status_code=400,
        )
    dropped = sorted(ir.raw_extras.keys() - extras_for(ir, provider_format).keys())
    if dropped:
        request_log().dropped_fields = dropped

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
                review,
            ),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    try:
        resp = await upstream.post(
            provider_key, upstream_path, orjson.dumps(upstream_body), fwd_headers
        )
    except UpstreamError as e:
        request_log().error = f"upstream_{e.kind}"
        record_provider_status(status_dict, provider_key, ok=False)
        return JSONResponse(
            content=format_error(inbound_format, 502, "upstream_error", str(e)),
            status_code=502,
        )

    if not resp.ok:
        request_log().error = "upstream_status"
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
        ir_resp = PARSE_RESPONSE[provider_format](upstream_data)
        out = RENDER_RESPONSE[inbound_format](ir_resp)
    except Exception as e:  # noqa: BLE001 - any codec failure on upstream output becomes a 502; logged
        log_translation_failed("response", e)
        return JSONResponse(
            content=format_error(inbound_format, 502, "translation_error", str(e)),
            status_code=502,
        )
    _record_usage(request_log(), ir_resp.usage, ir_resp.stop)
    if review is not None:
        out = await review.stamp_message(out)
    return JSONResponse(content=out)


async def _translate_stream(
    inbound_format: Format,
    provider_format: Format,
    payload: dict[str, Any],
    provider_key: str,
    upstream_path: str,
    request: Request,
    upstream: UpstreamClient,
    fwd_headers: dict[str, str],
    status_dict: dict[str, Any],
    review: Review | None,
) -> AsyncIterator[bytes]:
    rec = request_log()
    try:
        async with upstream.stream(
            provider_key, upstream_path, orjson.dumps(payload), fwd_headers
        ) as stream:
            ir_events = _usage_tap(STREAM_PARSE[provider_format](stream.iter_bytes()), rec)
            if inbound_format == "anthropic":
                ir_events = ac_stream.wrap_with_ping(ir_events)
            out_bytes = STREAM_RENDER[inbound_format](ir_events)
            if review is not None:
                out_bytes = review.stamp_stream(out_bytes)

            async for chunk in out_bytes:
                if await request.is_disconnected():
                    rec.client_disconnected = True
                    break
                yield chunk
        record_provider_status(status_dict, provider_key, ok=True)
    except (UpstreamError, UpstreamStatusError) as e:
        rec.error = _stream_error_type(e)
        record_provider_status(status_dict, provider_key, ok=False)
        yield format_stream_error(inbound_format, "upstream_error", _stream_error_message(e))
    except Exception as e:  # noqa: BLE001 - becomes an in-stream error event; logged
        log_translation_failed("stream", e)
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
