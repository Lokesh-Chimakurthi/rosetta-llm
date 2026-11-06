---
type: getting-started
title: Quickstart
description: Install, configure, and run the Rosetta proxy and make a first cross-format translated request.
tags: [quickstart, setup, configuration, docker, translation]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-b4d29a449a925a20917bce6e
    resource: repo://config.example.jsonc
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-0c8cc5debd4af7702e061205
    resource: repo://src/rosetta/__main__.py
  - id: openwiki-source-e3ab085c973475f4f7e838f9
    resource: repo://src/rosetta/asgi.py
  - id: openwiki-source-c28da7447e07ed3182c8f133
    resource: repo://src/rosetta/config.py
  - id: openwiki-source-05dd3552f31fbb4e671f95a6
    resource: repo://src/rosetta/pipeline.py
  - id: openwiki-source-297d1698fdfe3cf4e5029238
    resource: repo://src/rosetta/routes/health.py
generated: { by: "opencode", at: "2026-09-04T02:22:07.783Z" }
---

# Quickstart

Rosetta is a multi-format bidirectional translation proxy for LLM APIs that lets any client SDK talk to any provider by translating between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages behind three endpoint families with full streaming support.

## Responsibility and task routing

- Start here to install, configure, and run the proxy and to make a first translated request; follow the cross-links when the task moves past first boot into internals, endpoint details, or production deployment.
- Translation behavior (passthrough versus IR paths, model resolution, error mapping) is documented in the pipeline dispatcher, the canonical IR model it translates through, and the per-format codec pages for Anthropic, OpenAI Chat, and OpenAI Responses.
- Endpoint details (all routes, token counting, provider status) live under HTTP API endpoints, model listing and Claude Code picker integration live under model discovery, streaming internals live under streaming translation, and production concerns live under configuration plus deployment and observability.

## Install and run

- The fastest path is `uvx` with no install: create `~/.rosetta-llm`, copy `config.example.jsonc` to `~/.rosetta-llm/config.json`, edit in providers plus API keys, then run `uvx rosetta-llm` (or `uvx rosetta-llm --config /path/to/config.json`, equivalently `ROSETTA_CONFIG=/path/to/config.json uvx rosetta-llm`).
- For a persistent install run `uv tool install rosetta-llm` then `rosetta-llm --help` or `rosetta-llm --config ~/my-config.json --port 9999`, for Docker run `docker run -p 7860:7860 -v ~/.rosetta-llm/config.json:/app/config.json -e ANTHROPIC_API_KEY=... -e OPENAI_API_KEY=... ghcr.io/lokesh-chimakurthi/rosetta-llm:latest`, and for source checkouts run `uv sync` then `python -m rosetta`.
- Default resolution order for configuration is `--config` flag first, then `ROSETTA_CONFIG`, then `~/.rosetta-llm/config.json`, with the service binding `0.0.0.0:7860` unless overridden by config or `--host`/`--port`.

## Minimal configuration

- Providers are keyed by the prefix used in model ids (`<provider_key>/<model_name>`, e.g. `abc/kimi-k2.5` or `anthropic/claude-opus-4-7`), each declaring `format` (`openai_chat`/`openai_responses`/`anthropic`), `base_url`, credentials (`api_key` or `api_key_env`), and either an explicit `models` list or `["*"]` for upstream auto-discovery.
- Copy the `abc` (OpenAI Chat), `anthropic`, and `openai` (Responses) stanzas from `config.example.jsonc` as starting points, set secrets via `api_key_env` environment variables rather than committing keys, and leave `proxy.api_keys` empty for local development versus populating it to require `Authorization: Bearer` on inference routes.

## First translated requests

- Anthropic client → OpenAI-backed model: `POST http://localhost:7860/v1/messages` with `{"model": "openai/gpt-5.4", "max_tokens": 256, "messages": [{"role": "user", "content": "Hello!"}]}` translates via the IR to the provider's Responses format and back, succeeding only when the `openai` provider key exists in config.
- OpenAI client → Anthropic-backed model: `POST http://localhost:7860/v1/chat/completions` with `{"model": "anthropic/claude-opus-4-7", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}` translates the other direction over SSE, while same-format calls (e.g. Chat → `openai_chat` provider) take the verbatim passthrough fast path.
- Verify with `GET /health` (liveness), `GET /providers` (per-provider ok/error/unknown status), and `GET /v1/models` (merged `<provider>/<model>` list), noting that Claude Code discovery additionally benefits from `ANTHROPIC_BASE_URL=http://localhost:7860` plus `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`.

## Development checks

- Run `uv sync --group dev` once, then `uv run pytest -q`, `uv run mypy src/`, and `uv run ruff check src/ tests/` to validate behavior, types, and lint before changing translation logic.
