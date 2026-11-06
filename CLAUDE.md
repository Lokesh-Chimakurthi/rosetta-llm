# rosetta-llm

Multi-format bidirectional LLM proxy. Translates between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages — letting any client SDK talk to any provider regardless of the provider's native API format.

## Quick commands

```bash
uv sync                        # install deps + dev tools
uv run pytest tests/ -q        # run tests (80)
uv run ruff check src/ tests/  # lint
uv run ruff format src/ tests/ # format
uv run mypy src/rosetta        # type-check (strict mode)
uv run python -m rosetta       # start the proxy (reads ~/.rosetta-llm/config.json)
```

## Running

```bash
# uvx (no install)
uvx rosetta-llm

# uv tool (persistent)
uv tool install rosetta-llm
rosetta-llm --config /path/to/config.json --port 7860

# Docker
docker run -p 7860:7860 -v ~/.rosetta-llm/config.json:/app/config.json \
  ghcr.io/lokesh-chimakurthi/rosetta-llm:latest
```

## Architecture

**Passthrough fast path**: when `inbound_format == provider.format`, forward verbatim — only the `model` field is rewritten (strip prefix / apply `upstream_name`). No IR allocation.

**Translation slow path**: `inbound bytes → IR → provider format → upstream → provider bytes → IR → inbound format`. Streaming uses per-format state machines that emit/consume `CanonicalStreamEvent`.

**Canonical IR** (`ir/`): every block carries an optional `_raw` dict sidecar (PrivateAttr, defaults to `None`) so format-specific fields survive lossy rendering and round-trip correctly. This is critical for reasoning fidelity and tool-call identity preservation.

## Key design decisions

- **uv project** — always use `uv run <cmd>`.
- **No pydantic-ai** — pure httpx async. pydantic-ai's agent abstraction obscures wire-level fidelity.
- **No BaseHTTPMiddleware** — blocks SSE streaming. Auth and request-id are pure ASGI middleware.
- **Dockerfile** — multi-stage build using uv. Project is installed into the venv; CMD uses the `rosetta-llm` CLI. Config is mounted at `/app/config.json` via the `ROSETTA_CONFIG` env var.
- **Dockerfile.hf** — Hugging Face Spaces variant. Pulls the GHCR image directly (`FROM ghcr.io/...`) — no clone, no build. Just COPY your config.json in.

## Claude Code gateway

When `ANTHROPIC_BASE_URL` points at Rosetta, Claude Code discovers models via `GET /v1/models` at startup. Rosetta detects Claude Code by the `X-Claude-Code-Session-Id` header and returns a model list tailored for the `/model` picker:

- Models named `claude-*` or `anthropic/*` pass through unchanged.
- All other models get a `claude-code/` prefix — this bypasses Claude Code's built-in filter (which only shows models starting with `claude` or `anthropic`).

On inference, the `claude-code/` prefix is stripped and the model resolves normally. Session headers (`anthropic-beta`, `anthropic-version`, `x-claude-code-session-id`) are forwarded to every upstream call.

## Critical invariants

### Anthropic codec
- **Tool-result ordering**: within a user message, all `tool_result` blocks MUST precede any `text` block. `_enforce_tool_result_ordering()` auto-repairs on render.
- **Tool-use input**: Anthropic's `tool_use.input` is a JSON object. Serialize to `arguments_json_text` (string) in IR via orjson; deserialize back on render.
- **Reasoning signature encoding**: `f"{encrypted_content}@{reasoning_id}"` — split on the **last** `@` to decode. Compaction items prefix with `cm1#`. This is lossless across Anthropic ↔ Responses round-trips.
- **thinking.type**: parsed as `reasoning.thinking_type` (enabled/adaptive/disabled). Rendered back as `thinking: {type, budget_tokens}`.
- **cache_control**: preserved in `_raw` on every content block. Rendered back when present.
- **Tool extras**: `defer_loading`, `type`, `cache_control` from tool `_raw` are merged into rendered tool definitions.
- **Tool search**: `server_tool_use` + `tool_search_tool_result` blocks parse into `ServerToolUsePart` / `ToolSearchResultPart` and render back. A bare `tool_reference` name resolves against the request's `tools[]` catalog at parse time. Search tool entries render verbatim from `_raw`, never with an injected `input_schema`.

### OpenAI Chat codec
- **System/developer roles**: extracted from message list, concatenated with `\n\n`, stored as `system` on IR.
- **`tool`/`function` role**: each tool message emits one `role=tool` message in the rendered Chat body. Tool results are split from user messages.
- **`file` content part**: degraded to `"[File attached: <name>]"` text placeholder.
- **`input_audio`/`input_video`**: degraded to text placeholder.
- **Stop**: `stop` field accepts string or array. Arrays map to `stop_sequences`; single strings also map.

### OpenAI Responses codec
- **Input items**: message, function_call, function_call_output, reasoning, compaction, item_reference.
- **Output items**: message, function_call, reasoning, compaction.
- **Reasoning**: `encrypted_content` + `id` round-trip via Anthropic signature. `summary` accepts concise/detailed/auto.
- **Tool search**: `tool_search_call` / `tool_search_output` items map to/from the anthropic search blocks, paired through a synthesized `srvtoolu_` id when `call_id` is null (server execution). Client/BYOT execution has no anthropic equivalent and is refused with a translation error (search items at parse, `tools[]` entries at anthropic render), never silently converted. The search variant (regex/bm25) lives in IR only, on the server-tool block name; Responses has no variant field of its own, so the wire stays spec-clean and bm25 degrades to regex across any responses hop (the parse side still honors `search_variant` from peer proxies). `defer_loading` on functions round-trips via the IR `deferred` flag; a `{"type":"tool_search","execution":"server"}` entry is synthesized when any tool is deferred and none is present (regex variant by default). tool-search `arguments` is a JSON object on this wire; a JSON string is rejected by validation (loud 400).

### Pipeline
- **Header forwarding**: `anthropic-beta`, `anthropic-version`, `x-claude-code-session-id` extracted from inbound request and forwarded to every upstream call.
- **Model resolution**: `<provider_key>/<model_name>` split on first `/`. Optional `upstream_name` per model remaps what we forward. `claude-code/` prefix stripped and re-resolved. A bare model id (no `/`) routes via `default_provider` when set; an empty id is never routed.
- **Provider status**: recorded after each upstream call; exposed via `GET /providers`.
- **Cancellation**: starlette cancels the streaming generator on client disconnect (ASGI < 2.4 task-group branch), closing the upstream `httpx.stream` context.

### Streaming
- **Ping injection**: when outbound format is anthropic, `wrap_with_ping()` emits a `ping` event if 15s pass without an upstream event. Uses a producer-task + queue pattern so the source generator is never cancelled mid-await.
- **Chat stream**: text blocks emit `PartStartEvent(index=0, part_type="text")` before deltas. `reasoning_content` opens a reasoning block with PartStart. Tool calls get unique indices offset from the text block.
- **Responses stream**: whitespace runaway guard — if >20 consecutive whitespace chars in tool-call argument deltas, emits an error event.
- **`[DONE]`**: the Chat stream renderer is defensive — it emits a terminal `[DONE]` even if `MessageStopEvent` was never received.
- **Usage**: Chat stream requests set `stream_options.include_usage`; the Anthropic stream renderer emits usage on a single terminal `message_delta`, cache fields included.

### Models endpoint
- Static models: listed in config, emitted immediately.
- `["*"]` auto-discover: queries upstream `/models` first, falls back to `/v1/models`. Results cached with per-provider `models_ttl_seconds`.
- `display_name` from upstream is preserved when present.
- A single `asyncio.Lock` (`app.state.model_refresh_lock`) serializes auto-discovery refreshes across providers, preventing concurrent fan-out on cache miss.
- **Claude Code**: detected via `x-claude-code-session-id` header. Returns models with `claude-code/` prefix for non-Anthropic models so they appear in the picker.

## Config.json shape

```json
{
  "host": "0.0.0.0",
  "port": 7860,
  "proxy": {"api_keys": []},
  "log_level": "info",
  "default_provider": null,
  "providers": {
    "<key>": {
      "format": "openai_chat | openai_responses | anthropic",
      "base_url": "https://...",
      "api_key": "sk-..." or "api_key_env": "ENV_VAR",
      "extra_headers": {},
      "extra_headers_env": {},
      "timeout": {"connect": 30, "read": 600},
      "models": [{"id": "model-name", "upstream_name": "...", "supports": {...}, "thinking_budget_default": 12288}],
      "models_ttl_seconds": 300
    }
  }
}
```

Model id format: `<provider_key>/<model_name>` (e.g., `anthropic/claude-opus-4-7`). Config lives at `~/.rosetta-llm/config.json` by default. Override with `--config` flag or `ROSETTA_CONFIG` env var.

## Tests

- `tests/codecs/test_roundtrip.py` — codec property tests: Anthropic round-trip, tool-use input as object, tool-result ordering, Chat→Anthropic tool-call ID preservation, reasoning lossless round-trip (encrypted_content+id via signature), response round-trip, unknown-param passthrough, max_tokens synthesis, streaming partial-JSON buffering.
- `tests/test_e2e.py` — FastAPI TestClient + respx: health, models, count_tokens, Chat passthrough, Anthropic→Chat translation, unknown provider 400, upstream-error format matching, auth, stream passthrough.

Run with: `uv run pytest tests/ -q` (80 tests, ~2s).

<!-- OPENWIKI:START -->

## OpenWiki

See [AGENTS.md](AGENTS.md) for OpenWiki agent instructions.

<!-- OPENWIKI:END -->
