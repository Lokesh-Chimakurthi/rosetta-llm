---
type: endpoint-reference
title: Model Discovery
description: Merged GET /v1/models across providers with background TTL snapshots, upstream auto-discovery, and Claude Code picker prefixing.
tags: [models, discovery, claude-code, cache, pagination]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-8c778e0ba817410e6f956e5c
    resource: repo://src/rosetta/app.py
  - id: openwiki-source-de520cf98fbc326777c3a55a
    resource: repo://src/rosetta/routes/models.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# Model Discovery

`GET /v1/models` serves a pre-built in-memory snapshot merging every configured provider, refreshed in the background so gateway discovery stays instant even when upstreams are slow.

## Responsibility and ownership

- `routes/models.py` owns snapshot construction (`_build_snapshot`), upstream fan-out (`_fetch_all_providers`), paginated serving (`get_models`, `_anthropic_page`), and manual refresh (`POST /v1/models/refresh`), while `app.py` owns the lifespan that creates the snapshot dict, triggers the initial refresh, and runs the 300-second background refresh loop.
- `UpstreamClient` owns the per-provider HTTP clients used for `["*"]` auto-discovery, and the pipeline owns the inverse mapping that strips the `claude-code/` display prefix back to a routable provider model at inference time.
- Capability synthesis (`_synthesize_caps`, `_anthropic_capabilities`) owns the translation from upstream `architecture`/`supported_parameters` metadata into the Anthropic capability document Claude Code consumes.

## Snapshot and refresh lifecycle

- The snapshot dict holds three pre-rendered lists (`openai`, `anthropic`, `claude_code`) plus `last_refresh`, and the endpoint serves one of them directly with no upstream I/O, which is what keeps responses under a few milliseconds for Claude Code's tight discovery timeout.
- Lifespan creates `UpstreamClient`, starts it, fires an immediate `refresh_models` task plus a 300-second `_models_refresh_loop` task, and on shutdown cancels both tasks (suppressing `CancelledError`) before closing upstream clients, so refresh failures can never prevent startup or leak connections.
- `refresh_models` fans out to all providers in parallel via `asyncio.gather`, rebuilds all three shapes from the fetched entries, stamps `last_refresh`, and logs total plus elapsed milliseconds, while any top-level exception is caught and logged as `models_refresh_failed` leaving the previous snapshot intact.
- `POST /v1/models/refresh` serializes concurrent refreshes behind `app.state.model_refresh_lock` and returns `{"refreshed": true, "models": N}` sized from the Claude Code list.

## Auto-discovery and static fallback

- Providers with `models == ["*"]` are auto-discovery providers queried at `GET /models` first with fallback to `GET /v1/models`, following `has_more`/`last_id` pagination until exhausted, while static providers contribute no fetch and are rendered from configuration.
- A failed or empty fetch for an auto-discovery provider yields no entries (the provider simply disappears from the list), whereas a static provider with a failed fetch falls back to its configured models with synthesized `display_name`, `max_tokens` derived as `thinking_budget_default * 4`, and `max_input_tokens` of 200000.
- Fetched entries preserve upstream `display_name`/`name`, `max_tokens` (including `top_provider.max_completion_tokens` fallback), `max_input_tokens`/`context_length` (including `top_provider.context_length` fallback), `created`/`created_at` with epoch-to-ISO conversion, and native `capabilities` when present, otherwise synthesizing image/thinking/structured-output capabilities from `architecture.input_modalities` and `supported_parameters`.
- Per-provider fetches are isolated (one provider's exception cannot fail the gather), and the OpenAI shape additionally derives `architecture.modality` plus a sorted `supported_parameters` list from the entry's capability document.

## Claude Code picker behavior

- Format selection is header-driven: presence of `x-claude-code-session-id` or a `claude-code/`-prefixed `user-agent` selects the Claude Code list, else presence of `anthropic-version` selects the Anthropic page shape, else the OpenAI list shape is returned, with Claude Code consumers always receiving the Anthropic page shape even without `anthropic-version`.
- The Claude Code list copies the Anthropic list with every id not already starting with `claude` or `anthropic` prefixed as `claude-code/<provider>/<model>`, which bypasses Claude Code's built-in picker filter, while already-Anthropic ids pass through unchanged.
- Anthropic and Claude Code responses are paginated through `after_id`/`before_id`/`limit` (1–1000, default 20) with `has_more`/`first_id`/`last_id` computed by `_anthropic_page`, OpenAI responses return the full list unpaginated, and all responses carry `Cache-Control: public, max-age=60`.

## Representative tests

- `tests/test_e2e.py` covers the static merged OpenAI list asserting sorted ids `["abc/m1", "anth/claude"]`, which exercises the static-fallback snapshot path without upstream fetching.
