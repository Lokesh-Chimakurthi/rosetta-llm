---
type: operations-guide
title: Deployment and Observability
description: App factory and lifespan, upstream connection pool, ASGI auth and request-id middleware, structured logging, Docker images, and provider status.
tags: [deployment, observability, asgi, auth, logging, docker]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-bb1ebe868e35e9e500714501
    resource: repo://Dockerfile
  - id: openwiki-source-8c778e0ba817410e6f956e5c
    resource: repo://src/rosetta/app.py
  - id: openwiki-source-3c7e6c06db912ea69a88490a
    resource: repo://src/rosetta/auth.py
  - id: openwiki-source-32b50e124ec435bb03509662
    resource: repo://src/rosetta/observability.py
  - id: openwiki-source-5dfdd5645d86935ac47030b1
    resource: repo://src/rosetta/upstream.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# Deployment and Observability

The service is a FastAPI app with a managed lifespan, one `httpx` client per provider, and two pure-ASGI middleware layers, packaged as a multi-stage Docker image with a Hugging Face variant.

## Responsibility and ownership

- `app.py` owns the `create_app` factory (state init, middleware order, router mounting) and the `lifespan` (upstream start/close plus initial and periodic model refresh), while `asgi.py` owns the import-time `rosetta.asgi:app` singleton built from `ROSETTA_CONFIG` for uvicorn workers.
- `upstream.py` owns the per-provider `httpx.AsyncClient` pool with auth header injection and the `request_json`/`stream` call paths, `auth.py` owns bearer-token enforcement, and `observability.py` owns structlog setup plus the request-id middleware with debug body buffering.
- `Dockerfile` owns the reproducible multi-stage production image and `Dockerfile.hf` owns the thin Hugging Face Spaces wrapper that reuses the published GHCR image.

## App factory and lifespan

- `create_app` initializes `app.state.config`, an empty `provider_status` dict, an empty `model_snapshot` dict, and a `model_refresh_lock`, then adds `AuthMiddleware` followed by `RequestContextMiddleware` (outermost executes last-added first, so request-id context is established before auth runs) and mounts the health, messages, chat, responses, and models routers.
- Lifespan constructs `UpstreamClient` from configured providers and starts it, fires an immediate `refresh_models` task plus a 300-second `_models_refresh_loop` task that logs but never propagates refresh exceptions, and on shutdown cancels both tasks (suppressing `CancelledError`) before closing all upstream clients.
- `ROSETTA_CONFIG` defaults to `~/.rosetta-llm/config.json` in both entry points, `setup_logging` runs once before serving, and the CLI path (`__main__.py`) applies `--host`/`--port` overrides then calls `uvicorn.run` with `access_log=False`, `log_config=None`, and configurable `proxy_headers`/`forwarded_allow_ips`.

## Upstream pool

- `UpstreamClient.start` builds one `httpx.AsyncClient` per provider keyed by provider key with `base_url` stripped of trailing slashes, Anthropic providers receiving `x-api-key` plus defaulted `anthropic-version: 2023-06-01` while OpenAI-family providers receive `Authorization: Bearer`, every client getting `user-agent: rosetta/0.1`, merged `extra_headers` (including env-resolved secrets), per-provider connect/read timeouts with write pinned to read and a 10-second pool timeout, and HTTP/2 disabled.
- `request_json` serializes bodies with orjson and posts with `content-type: application/json` plus per-call extra headers (the forwarded gateway headers), while `stream` posts with an additional `accept: text/event-stream`, raises `httpx.HTTPStatusError` with the first 512 characters of the error body on 4xx/5xx so stream codecs can convert it downstream, and otherwise yields raw byte chunks that the pipeline polls against client disconnect.
- Unknown provider keys raise `ValueError` from `get_client` rather than opening ad-hoc connections, and `close` closes every client and clears the pool.

## Auth and request context

- `AuthMiddleware` is pure ASGI (not `BaseHTTPMiddleware`) precisely so SSE streams flow chunk-by-chunk without buffering, passing through non-HTTP scopes, open-access configurations with no keys, and the public paths `/health` and `/providers` untouched.
- Protected paths accept either `Authorization: Bearer <token>` (case-insensitive scheme) or `x-api-key`, compare against the configured key set, and on failure return `401` with a per-dialect body (Anthropic shape for `/v1/messages*`, OpenAI shape otherwise) without ever reaching the router.
- `RequestContextMiddleware` is also pure ASGI, clears contextvars per request, binds `request_id` (from `x-request-id` or random hex) plus method and path, echoes `x-request-id` on every response via a send wrapper, redacts `authorization`/`x-api-key`/`api-key`/`cookie` in debug logs, caps logged bodies at 8192 bytes, and replays buffered bodies downstream so debug logging never consumes the request stream.

## Logging and Docker

- `setup_logging` merges contextvars, log level, and ISO UTC timestamps, rendering human-readable console output at `debug` versus JSON otherwise with filtering at the configured level, while `bind_request_context` adds `provider`/`model`/`format` per inference request and the middleware clears contextvars after each response to prevent cross-request leakage.
- The production `Dockerfile` builds a venv with `uv sync --frozen --no-dev` on `python:3.13-slim`, copies it into a minimal runtime stage with `curl`/`ca-certificates`, runs as a non-root `app` user, sets `ROSETTA_CONFIG=/app/config.json`, exposes 7860, defines a 30s-interval `/health` curl `HEALTHCHECK`, and defaults to `rosetta-llm --host 0.0.0.0 --port 7860` with the config intended to be mounted as a volume.
- `Dockerfile.hf` reuses the published GHCR image without rebuilding and only copies a baked `config.json`, with provider secrets supplied via Space variables referenced by `api_key_env` rather than committed keys.

## Representative tests

- `tests/test_e2e.py` covers unauthenticated rejection with `401` plus bearer-authenticated success when `proxy.api_keys` is configured, and header propagation asserting env-resolved plus literal custom headers both arrive upstream.
