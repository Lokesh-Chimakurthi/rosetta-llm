# rosetta-llm

Multi-format bidirectional LLM proxy. Translates between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages — letting any client SDK talk to any provider regardless of the provider's native API format.

## Quick commands

```bash
uv sync                        # install deps + dev tools
uv run pytest tests/ -q        # run tests
uv run ruff check src/ tests/  # lint
uv run ruff format src/ tests/ # format
uv run pyrefly check           # type-check (strict preset, src + tests)
uv run python -m rosetta       # start the proxy (reads ~/.rosetta-llm/config.json)
```

## Running

```bash
# uvx (no install)
uvx rosetta-llm

# uv tool (persistent)
uv tool install rosetta-llm
rosetta-llm serve --config /path/to/config.json --port 7860   # `serve` is the default command
rosetta-llm check --config /path/to/config.json               # validate + secret-free summary; exit 2 if serve would refuse
rosetta-llm version

# Docker (config must set proxy.api_keys: the container binds 0.0.0.0)
docker run -p 127.0.0.1:7860:7860 -v ~/.rosetta-llm/config.json:/app/config.json \
  ghcr.io/lokesh-chimakurthi/rosetta-llm:latest
```

CLI (`__main__.py`, argparse only): `serve` and `check` share `--config/-c`, `--host`, `--port/-p`, `--log-level`, `--allow-unauthenticated`, `--proxy-headers/--no-proxy-headers`, `--forwarded-allow-ips`; `serve` adds `--root-path`, `--timeout-keep-alive`, `--limit-concurrency`. Precedence: CLI flag > config > default. A first argument that is not a command means `serve`. Config and exposure failures print the reason and exit 2.

## Architecture

**Passthrough fast path**: when `inbound_format == provider.format`, forward verbatim — only the `model` field is rewritten (strip prefix / apply `upstream_name`). No IR allocation.

**Translation slow path**: `inbound bytes → IR → provider format → upstream → provider bytes → IR → inbound format`. Streaming uses per-format state machines that emit/consume `CanonicalStreamEvent`.

**Canonical IR** (`ir/`): every block carries a `_raw` dict sidecar (PrivateAttr) so format-specific fields survive lossy rendering and round-trip correctly. This is critical for reasoning fidelity and tool-call identity preservation.

## Key design decisions

- **uv project** — always use `uv run <cmd>`, never `.venv/bin/python` directly.
- **No pydantic-ai** — pure httpx2 async. pydantic-ai's agent abstraction obscures wire-level fidelity.
- **Upstream seam** — `upstream.py` is the only module that imports httpx2. Everything else uses `UpstreamClient.post/get/stream`, `UpstreamResponse`, `UpstreamStream`, `UpstreamError` (connect/timeout/protocol, message never names the host) and `UpstreamStatusError` (stream status >= 400, raised on enter). Stream requests send `accept-encoding: identity`; all providers share one TLS context (system trust store via truststore).
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
- **Phase markers** (commentary/final_answer): preserved in `_raw`, emitted only when the source item had them.

### Pipeline
- **Header forwarding**: `anthropic-beta`, `anthropic-version`, `x-claude-code-session-id` extracted from inbound request and forwarded to every upstream call.
- **Untranslated fields**: top-level request fields a codec doesn't translate land in `raw_extras`, tagged with `raw_extras_format`. `codecs/extras.py::extras_for` is the one rule for what a renderer forwards: everything to the same format, only `_SHARED_FIELDS` (same name and meaning) between Chat and Responses, nothing between Anthropic and either OpenAI format. Strict upstreams reject unknown fields. The pipeline logs dropped names (never values) as `dropped_fields` at debug.
- **Auto mode review**: passthrough relays Claude Code's `safeguards` request field and the upstream's `safeguard_results` (top-level, or in the final `message_delta.delta`) verbatim, and must never rewrite tool-use ids, because verdicts are keyed by them. On translation `safeguards` is dropped, like every Anthropic-only field (see **Untranslated fields**). Never synthesize `safeguard_results`: a completed response without them is what makes Claude Code fall back to its own classifier requests (model chosen client-side via `CLAUDE_CODE_AUTO_MODE_MODEL`).
- **Model resolution**: `<provider_key>/<model_name>` split on first `/`. Optional `upstream_name` per model remaps what we forward. `claude-code/` prefix stripped and re-resolved.
- **Provider status**: recorded after each upstream call; exposed via `GET /providers`.
- **Cancellation**: `request.is_disconnected()` polled in streaming loops; on disconnect, the generator exits, closing the `UpstreamClient.stream` context and its connection.

### Auth and exposure
- `ensure_safe_bind` (auth.py) is the one home of the exposure rule: non-loopback host + no `api_keys` + not `allow_unauthenticated` → `InsecureBindError` from `create_app`; the CLI exits 2.
- Keys are fixed at app creation and compared with `hmac.compare_digest`; Bearer or `x-api-key` may match. Public paths: `/health`, `/docs`, `/docs/oauth2-redirect`, `/openapi.json`.
- `errors.inbound_format_for(path, headers)` is the single home for which error envelope a request gets.

### API docs
- FastAPI's default `/docs` (Swagger UI assets from its CDN) and `/openapi.json`; ReDoc is off. Every route declares `summary`, `description` and its error statuses via `routes.schemas.error_responses(fmt, ...)`, which picks the envelope model for the route's format.

### Logging
- `RequestContextMiddleware` owns the per-request `RequestLog` (contextvar, `request_log()`); code sets fields on it instead of logging. `request_started` is emitted once (pipeline after model resolution, else at response start); `request_completed` only with `log_completed`, level by status. `/health` is never logged.
- Debug adds `upstream_request`/`upstream_response` (from `upstream.py`). Headers and bodies appear only with debug plus `log_headers`/`log_bodies`; bodies come from a pass-through receive tap capped at 64 KiB — never buffer the body. No per-chunk counters or logs.
- Credential header values are redacted; startup and error messages never contain provider URLs or keys. README "Logging" documents every event and field.

### Streaming
- **Ping injection**: when outbound format is anthropic, `wrap_with_ping()` emits a `ping` event if 15s pass without an upstream event. Uses a producer-task + queue pattern so the source generator is never cancelled mid-await.
- **Chat stream**: text blocks emit `PartStartEvent(index=0, part_type="text")` before deltas. `reasoning_content` opens a reasoning block with PartStart. Tool calls get unique indices offset from the text block.
- **Responses stream**: whitespace runaway guard — if >20 consecutive whitespace chars in tool-call argument deltas, emits an error event.
- **`[DONE]`**: Chat and Responses stream renderers are defensive — they emit a terminal `[DONE]` even if `MessageStopEvent` was never received.

### Models endpoint
- Static models: listed in config, emitted immediately.
- `["*"]` auto-discover: queries upstream `/models` first, falls back to `/v1/models`. Results cached with per-provider `models_ttl_seconds`.
- `display_name` from upstream is preserved when present.
- Per-provider `asyncio.Lock` prevents concurrent fan-out on cache miss.
- **Claude Code**: detected via `x-claude-code-session-id` header. Returns models with `claude-code/` prefix for non-Anthropic models so they appear in the picker.

## Config.json shape

```json
{
  "host": "127.0.0.1",
  "port": 7860,
  "proxy": {"api_keys": [], "allow_unauthenticated": false, "trust_proxy_headers": false, "forwarded_allow_ips": "127.0.0.1"},
  "log_level": "info",
  "log_completed": false,
  "log_headers": false,
  "log_bodies": false,
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

- `tests/codecs/test_roundtrip.py` — codec property tests: Anthropic round-trip, tool-use input as object, tool-result ordering, Chat→Anthropic tool-call ID preservation, reasoning lossless round-trip (encrypted_content+id via signature), response round-trip, unknown-param passthrough within a format and its absence across formats (shared Chat↔Responses fields excepted), max_tokens synthesis, streaming partial-JSON buffering.
- `tests/support/upstream.py` — `FakeUpstream` (an httpx2 transport routing by provider key and path, recording every request; `add_json`, `add_sse` with fixture name or bytes plus `chunk_size`/`delay_s`, `add_error`, `streams_opened`/`streams_closed`), `build_client(config, fake)` and `load_fixture(fmt, name)`. Tests never patch the network.
- `tests/support/contracts.py` — SDK-schema plus lifecycle validators for every wire format (`validate_anthropic_message/stream`, `validate_chat_completion/stream`, `validate_response/responses_stream`); each failure starts with `[<rule>]`.
- `tests/fixtures/sse/<format>/` — upstream SSE fixtures (text, tool_call, reasoning, error_midstream; anthropic also tool_call_safeguards), each opening with a comment naming the documentation it follows.
- `tests/test_harness.py` — the harness itself: fixtures pass, targeted mutations fail their named rule, passthrough stays byte-identical under any upstream chunking (hypothesis).
- `tests/test_upstream.py` — `UpstreamClient` contract: error-status streams, capped error body, stream headers, transport-error kinds without host leakage.
- `tests/test_e2e.py` — FastAPI TestClient + FakeUpstream: health, models, count_tokens, Chat passthrough, Anthropic→Chat translation, unknown provider 400, upstream-error format matching, auth, stream passthrough, auto mode review (passthrough keeps `safeguards`/`safeguard_results`, translation drops `safeguards`).

Run with: `uv run pytest tests/ -q`.
