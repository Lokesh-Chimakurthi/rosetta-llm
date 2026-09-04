---
type: endpoint-reference
title: HTTP API Endpoints
description: Route surface for inference, token counting, health, and provider status with shared request schemas and format-aware error envelopes.
tags: [http-api, endpoints, routes, errors, tokens]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-d2a960bf12051a1a201115c4
    resource: repo://src/rosetta/errors.py
  - id: openwiki-source-05dd3552f31fbb4e671f95a6
    resource: repo://src/rosetta/pipeline.py
  - id: openwiki-source-fd9a1cf2b28012f820c00121
    resource: repo://src/rosetta/routes/chat_completions.py
  - id: openwiki-source-297d1698fdfe3cf4e5029238
    resource: repo://src/rosetta/routes/health.py
  - id: openwiki-source-23f514a4ac7bbd471f5f2894
    resource: repo://src/rosetta/routes/messages.py
  - id: openwiki-source-1389d58c07780299ab4fb628
    resource: repo://src/rosetta/routes/responses.py
  - id: openwiki-source-e0700983bd29681ed4359c3b
    resource: repo://src/rosetta/routes/schemas.py
  - id: openwiki-source-3c53fda42ed9ad77de3ae365
    resource: repo://src/rosetta/tokens.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# HTTP API Endpoints

The public HTTP surface exposes three inference families plus local token counting, liveness, and provider status, with every inference route delegating to the same pipeline dispatcher so behavior stays uniform.

## Responsibility and ownership

- `routes/chat_completions.py` owns `POST /v1/chat/completions`, `routes/messages.py` owns `POST /v1/messages` plus `POST /v1/messages/count_tokens`, `routes/responses.py` owns `POST /v1/responses`, and `routes/health.py` owns `GET /`, `GET /health`, and `GET /providers`.
- `routes/schemas.py` owns the shared `ProxyRequest` shape plus `HealthResponse`, `ProvidersResponse`, `CountTokensResponse`, and both model-list response shapes, while `routes/deps.py` owns the `get_config`/`get_upstream` dependencies and their `ConfigDep`/`UpstreamDep` aliases.
- `errors.py` owns inbound-format-aware error envelopes for both JSON and streaming failures, and `tokens.py` owns the local tiktoken approximation backing `count_tokens`.

## Inference endpoints

- All three inference handlers accept the permissive `ProxyRequest` (`model`, `stream`, plus `extra="allow"` for every format-specific field), dump it to a plain dict, and call `pipeline.handle` with their fixed inbound format string, which is why unknown fields survive passthrough verbatim instead of being rejected by validation.
- `POST /v1/chat/completions` serves OpenAI Chat Completions with `stream: true` selecting SSE, `POST /v1/messages` serves Anthropic Messages with the same streaming flag, and `POST /v1/responses` serves OpenAI Responses, while model routing in every case uses the `<provider_key>/<model_name>` scheme with optional `claude-code/` prefix stripping resolved inside the pipeline.
- The chat, messages, and responses routers each depend on `get_config`, which raises `RuntimeError` when `app.state.config` is uninitialized rather than serving requests against missing configuration.

## Token counting, health, and status

- `POST /v1/messages/count_tokens` runs synchronously and locally via `count_tokens_anthropic`, so it never contacts upstream, never requires a valid model route, and returns only `{"input_tokens": N}`.
- The counter walks `system` (string or text-block list), every message `content` (strings, text/tool_use/tool_result/image/document/thinking blocks with tool envelopes charged a 16-token overhead and images/PDFs charged fixed 85/256-token baselines), and tool definitions (`name`, `description`, JSON-serialized `input_schema`) using the `o200k_base` encoding, which is an OpenAI-family approximation of Anthropic's tokenizer accurate only to roughly 5–15%.
- `GET /health` returns `{"status": "ok", ...}` unconditionally, `GET /` redirects to `/docs`, and `GET /providers` merges configured providers with the in-memory status dict (defaulting untouched providers to `unknown` with null timestamp) into `ProviderStatusItem` entries carrying `key`, `format`, `last_status`, and `last_check_ts`.
- Provider status entries are written by `record_provider_status` on upstream-contacting pipeline outcomes, so `/providers` reflects the latest upstream success or failure per provider key rather than a live probe.

## Error envelopes

- `format_error` emits the Anthropic shape (`{"type": "error", "error": {"type", "message"}}`) when the inbound format is Anthropic and the OpenAI shape (`{"error": {"message", "type", "code": None, "param": None}}`) otherwise, which keeps client SDKs parsing errors in the dialect they sent.
- `format_stream_error` serializes the same envelope as a single SSE frame (`event: error` plus `data:` for Anthropic, bare `data:` for OpenAI formats), which is how mid-stream transport and translation failures surface inline without breaking the event stream.
- Auth middleware errors follow the same per-path convention keyed off `/v1/messages` prefix matching, while pipeline 400s cover unknown models and request-translation failures and 502s cover upstream transport and response-translation failures.

## Representative tests

- `tests/test_e2e.py` covers health, local token counting lower bounds, Chat passthrough fidelity, Anthropic→Chat translation with `stop_reason` normalization, unknown-provider 400s, upstream 429 remapping into the inbound envelope, and Chat stream passthrough preserving `[DONE]` framing.
