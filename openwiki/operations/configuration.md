---
type: operations-guide
title: Configuration
description: config.json shape, provider formats, environment secret resolution, validation rules, and CLI overrides.
tags: [configuration, providers, secrets, validation, cli]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-b4d29a449a925a20917bce6e
    resource: repo://config.example.jsonc
  - id: openwiki-source-0c8cc5debd4af7702e061205
    resource: repo://src/rosetta/__main__.py
  - id: openwiki-source-e3ab085c973475f4f7e838f9
    resource: repo://src/rosetta/asgi.py
  - id: openwiki-source-c28da7447e07ed3182c8f133
    resource: repo://src/rosetta/config.py
  - id: openwiki-source-5dfdd5645d86935ac47030b1
    resource: repo://src/rosetta/upstream.py
  - id: openwiki-source-81af13fa7982f0b3becf1286
    resource: repo://tests/test_config.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# Configuration

All proxy behavior derives from one Pydantic-validated JSON document (JSONC comments allowed at load time) plus a small set of CLI and environment overrides resolved before the app starts.

## Responsibility and ownership

- `config.py` owns the schema (`Config`, `ProviderConfig`, `ModelConfig`, `Capabilities`, `TimeoutConfig`, `ProxyConfig`), the `api_key_env`/`extra_headers_env` resolution validators, the `is_auto_discover`/`static_models`/`effective_upstream_name` derived properties, and the comment-stripping `load_config` entry point.
- `config.example.jsonc` owns the documented reference shape operators copy to `~/.rosetta-llm/config.json`, and `__main__.py` owns CLI parsing plus `--host`/`--port` overrides applied after validation.
- `tests/test_config.py` owns the contract for header/env resolution success, missing-variable failure, literal-versus-env collision, and secret non-leakage in error messages.

## Document shape and defaults

- Top level carries `host` (default `0.0.0.0`), `port` (default `7860`), `proxy.api_keys` (default empty meaning open access), `log_level` (`debug`/`info`/`warning`/`error`, default `info`), and `providers` keyed by prefix used in `<provider_key>/<model_name>` routing.
- Each provider declares `format` (`openai_chat`/`openai_responses`/`anthropic`, selecting both the upstream path suffix and the auth header scheme), `base_url` (provider root; the proxy appends `/chat/completions`, `/responses`, or `/messages`), `api_key` or `api_key_env` (at least one source required; a literal `api_key` wins when both are set), `extra_headers`/`extra_headers_env`, `timeout` (default 30s connect / 600s read, with write and pool fixed in the upstream client), `models` (explicit list or `["*"]` for auto-discovery), and `models_ttl_seconds` (default 300).
- Each static model declares `id` (the suffix clients address), optional `upstream_name` remap (defaulting to `id` via `effective_upstream_name`), permissive-default `supports` capabilities (`tools`/`vision`/`thinking`/`structured_output` all true unless overridden), and `thinking_budget_default` (default 12288, also used to synthesize discovery `max_tokens` as four times the budget).

## Secrets and validation

- `api_key_env` resolves at validation time from the named environment variable and raises when unset or empty, and a provider with neither `api_key` nor resolvable `api_key_env` fails validation, so the process refuses to start without credentials rather than serving unauthenticated upstream calls.
- `extra_headers_env` maps header name to env var name with the same must-be-set-and-nonempty rule, merges resolved values into `extra_headers`, rejects any header declared in both literal and env maps as a config error, and never interpolates the secret value into the raised error message (only the variable name appears).
- Provider keys containing `/` are rejected because `/` is the model-id separator, `load_config` strips only full-line `//` comments (enabling the JSONC example to load as strict JSON afterward), and `extra_headers_env` empty or absent is a no-op preserving empty `extra_headers`.

## Runtime overrides

- The config path resolves as `--config`/`-c` flag first, then `ROSETTA_CONFIG` environment variable, then `~/.rosetta-llm/config.json`, and the same precedence file is used by both the `rosetta-llm` CLI and the `rosetta.asgi:app` import path (the latter reading only the environment variable).
- `--host` and `--port` override the loaded document after validation, `--proxy-headers` (default on) and `--forwarded-allow-ips` (default `*`) pass through to uvicorn's proxy handling, and `log_level` from the document drives both structlog setup and uvicorn's log level with `access_log` disabled.

## Representative tests

- `tests/test_config.py` covers env resolution into `extra_headers`, unset-variable `ValidationError` naming the variable, literal/env collision naming the header, literal-only and empty-env no-op behavior, and absence of the secret value from collision error text.
- `tests/test_e2e.py` covers end-to-end consumption of `api_key_env` providers plus `extra_headers_env` propagation asserting the resolved secret and literal headers both arrive upstream.
