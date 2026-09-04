---
type: architecture-concept
title: Pipeline Dispatcher
description: Central request dispatcher that chooses passthrough versus IR translation, resolves provider models, forwards gateway headers, and maps errors per wire format.
tags: [pipeline, routing, translation, passthrough, error-handling]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-05dd3552f31fbb4e671f95a6
    resource: repo://src/rosetta/pipeline.py
  - id: openwiki-source-fd9a1cf2b28012f820c00121
    resource: repo://src/rosetta/routes/chat_completions.py
  - id: openwiki-source-297d1698fdfe3cf4e5029238
    resource: repo://src/rosetta/routes/health.py
  - id: openwiki-source-23f514a4ac7bbd471f5f2894
    resource: repo://src/rosetta/routes/messages.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# Pipeline Dispatcher

`src/rosetta/pipeline.py` is the single choke point behind all three inference endpoints, deciding per request whether to forward bytes verbatim or to translate through the canonical IR in either non-streaming or streaming mode.

## Responsibility and ownership

- The pipeline owns inbound-format versus provider-format dispatch, model-id resolution to provider plus upstream model name, Anthropic gateway header forwarding, upstream path selection, provider-status recording, and inbound-format-aware error envelopes.
- The three thin route handlers (`routes/chat_completions.py`, `routes/messages.py`, `routes/responses.py`) own only endpoint binding and pass their fixed `inbound_format` string plus the raw body dict into `pipeline.handle`, so all routing intelligence lives in one module.
- Codec dispatch tables (`_PARSE_REQ`, `_RENDER_REQ`, `_PARSE_RESP`, `_RENDER_RESP`, `_STREAM_PARSE`, `_STREAM_RENDER`) own the mapping from format name to converter function, which keeps `handle`, `_passthrough`, and `_translate` free of per-format branches.

## Entry and dispatch flow

- `handle(inbound_format, payload, request)` reads `model` from the payload, resolves it via `_resolve_model`, binds `provider`/`model`/`format` into the structlog request context, copies the payload with only the `model` field rewritten to the upstream name, and then branches on `inbound_format == provider.format` to `_passthrough` versus `_translate`.
- `is_stream` is derived solely from the boolean `stream` payload field, and `upstream_path` is derived solely from the provider format (`/chat/completions`, `/responses`, `/messages`), which means a client can request any inbound endpoint with `stream: true` against any provider format.
- Forwarded headers are extracted once per request by `_forwarded_headers` and threaded through both paths into every `UpstreamClient` call, so gateway metadata survives regardless of translation direction.

## Model resolution

- Model ids must have the shape `<provider_key>/<model_name>` split on the first `/`, where `provider_key` must match a configured provider and `model_name` is remapped through that provider's static model table to `effective_upstream_name` when present.
- The `claude-code/` gateway prefix is stripped and the remainder is re-resolved recursively, so `claude-code/openai/gpt-5.4` routes exactly like `openai/gpt-5.4`, while a bare `claude-code/<name>` without an inner slash is rejected as invalid.
- Unknown providers and malformed ids return an immediate `400` in the inbound format's error envelope without ever contacting upstream, and `_resolve_model` never mutates configuration state.

## Passthrough fast path

- When inbound and provider formats match, `_passthrough` forwards the copied body verbatim with zero IR allocation, which preserves unknown fields byte-for-byte and avoids any translation loss or cost.
- Non-streaming passthrough returns the upstream `content`, `status_code`, and `content-type` unchanged (recording ok/error provider status), while transport-level `httpx.HTTPError` maps to a `502 upstream_error` envelope in the inbound format.
- Streaming passthrough yields upstream bytes directly behind `_passthrough_stream_with_recovery`, which polls `request.is_disconnected()` per chunk to abort on client disconnect, records provider status on clean exhaustion, and converts mid-stream `httpx.HTTPError` into a single inbound-format stream error event instead of truncating silently.

## Translate slow path

- `_translate` runs inbound `parse_request` → provider `render_request` for the outbound leg, where any exception maps to a `400 translation_error` in the inbound envelope because the fault lies in the client payload or its IR conversion, not in the upstream.
- Non-streaming translation posts the rendered provider body, maps transport errors to `502 upstream_error`, maps non-2xx upstream responses by extracting the nested error message (`error.message`, string `error`, top-level `message`, else raw payload) into an inbound-format envelope that preserves the upstream status code, and only on success runs provider `parse_response` → inbound `render_response` with response-translation failures mapped to `502 translation_error`.
- Streaming translation chains provider `_STREAM_PARSE` → optional Anthropic `wrap_with_ping` → inbound `_STREAM_RENDER` as lazy async generators, which keeps memory constant, propagates backpressure, polls disconnect per emitted chunk, records provider status once at exhaustion, and converts transport versus translation exceptions into distinct `upstream_error` versus `translation_error` stream events.
- `wrap_with_ping` is applied only when the inbound (outbound-render) format is Anthropic, so Chat and Responses consumers never see keepalive events they cannot parse.

## Headers, status, and cancellation

- Only `anthropic-beta`, `anthropic-version`, and `x-claude-code-session-id` are forwarded upstream per the Claude Code LLM Gateway spec, which preserves prompt caching and feature detection while preventing arbitrary client headers from reaching providers.
- Every upstream-contacting success, transport failure, and non-2xx upstream response records provider ok/error status into `app.state.provider_status`, which is what `GET /providers` later reports, while pre-upstream 400s (unknown model, request-translation failure) and response/stream translation-error paths return without recording status.
- Cancellation is cooperative: streaming generators check `request.is_disconnected()` before yielding each chunk and simply break, which exits the `httpx.stream` context manager in `UpstreamClient.stream` and closes the upstream connection without extra signalling.
- All streaming responses carry `Cache-Control: no-cache`, `Connection: keep-alive`, and `X-Accel-Buffering: no`, which disables proxy buffering that would otherwise defeat chunk-by-chunk SSE delivery.

## Representative tests

- `tests/test_e2e.py` covers Chat passthrough fidelity, Anthropic→Chat bidirectional translation with `stop_reason` normalization, unknown-provider 400s, upstream 429 remapping into the inbound (Anthropic) envelope, and Chat stream passthrough preserving `[DONE]` framing.
