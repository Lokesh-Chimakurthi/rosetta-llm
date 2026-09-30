# Rosetta

Multi-format bidirectional translation proxy for LLM APIs. Translates between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages formats — letting any client SDK talk to any provider regardless of which native API the provider speaks.

## Quick Start

### uvx (no install required)

```bash
# Create your config
mkdir -p ~/.rosetta-llm
cp config.example.jsonc ~/.rosetta-llm/config.json
# Edit ~/.rosetta-llm/config.json with your providers and API keys

# Run instantly
uvx rosetta-llm
```

Or point to a custom config:

```bash
uvx rosetta-llm --config /path/to/config.json

# Equivalent via env var:
ROSETTA_CONFIG=/path/to/config.json uvx rosetta-llm
```

### uv tool install (persistent)

```bash
uv tool install rosetta-llm
rosetta-llm --help
rosetta-llm --config ~/my-config.json --port 9999
```

### Docker

```bash
docker run -p 7860:7860 \
  -v ~/.rosetta-llm/config.json:/app/config.json \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e OPENAI_API_KEY=sk-... \
  ghcr.io/lokesh-chimakurthi/rosetta-llm:latest
```

### From source

```bash
git clone https://github.com/Lokesh-Chimakurthi/rosetta-llm.git
cd rosetta-llm
uv sync
python -m rosetta
```

## Features

- **Three endpoint families**: `/v1/chat/completions`, `/v1/responses`, `/v1/messages` — all with full streaming support
- **Passthrough fast path**: zero-overhead when inbound format matches provider's native format
- **Canonical IR translation**: lossless cross-format translation including thinking blocks, tool calls, and reasoning
- **Claude Code gateway**: full model picker integration — non-Anthropic models appear in `/model` via `claude-code/` prefixing
- **Provider routing**: `<provider>/<model>` prefix scheme (e.g., `anthropic/claude-opus-4-7`)
- **Bearer-token auth**: optional proxy-level API key authentication
- **Structured JSON logging**: request-scoped with configurable log levels
- **Docker support**: multi-stage build with `python:3.13-slim`

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/messages` | Anthropic Messages |
| POST | `/v1/messages/count_tokens` | Local tiktoken token count |
| POST | `/v1/chat/completions` | OpenAI Chat Completions |
| POST | `/v1/responses` | OpenAI Responses |
| GET | `/v1/models` | Merged model list |
| GET | `/health` | Liveness check |
| GET | `/providers` | Provider status |

## Model ID Format

Models are addressed as `<provider_key>/<model_name>` where `provider_key` matches a key in your config's `providers` section.

Examples:
- `abc/kimi-k2.5` — routes to the "abc" provider with model "kimi-k2.5"
- `anthropic/claude-opus-4-7` — routes to the "anthropic" provider
- `openai/gpt-5.4` — routes to the "openai" provider

## Claude Code Integration

Rosetta is a fully compatible [Claude Code LLM gateway](https://code.claude.com/docs/en/llm-gateway). Point Claude Code at Rosetta and all configured providers appear in the `/model` picker — including non-Anthropic models.

### Setup

```bash
export ANTHROPIC_BASE_URL=http://localhost:7860
export ANTHROPIC_AUTH_TOKEN=sk-proxy-XXXX   # if proxy auth is enabled
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
```

Or in Claude Code settings (`~/.claude/settings.json`):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:7860",
    "ANTHROPIC_AUTH_TOKEN": "sk-proxy-XXXX",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1"
  }
}
```

### Model picker

On startup, Claude Code queries `GET /v1/models` with its session headers. Rosetta detects Claude Code (via the `X-Claude-Code-Session-Id` header) and returns a model list tailored for the picker:

- Models already named `claude-*` or `anthropic/*` pass through unchanged
- All other models get a `claude-code/` prefix — this ensures they pass Claude Code's built-in model filter (which only shows models starting with `claude` or `anthropic`)

For example, if your config has an OpenAI provider with `gpt-5.4`, it appears in the picker as `claude-code/openai/gpt-5.4`. When selected, Rosetta strips the `claude-code/` prefix internally and routes to the correct provider.

### Headers forwarded upstream

Rosetta forwards Claude Code's session headers (`anthropic-beta`, `anthropic-version`, `X-Claude-Code-Session-Id`) to every upstream call, preserving prompt caching and feature detection.

## Translation Matrix

The proxy automatically translates between formats:

| Client Endpoint | Provider Format | Path |
|----------------|-----------------|------|
| `/v1/messages` | anthropic | passthrough |
| `/v1/messages` | openai_chat | translate via IR |
| `/v1/messages` | openai_responses | translate via IR |
| `/v1/chat/completions` | openai_chat | passthrough |
| `/v1/chat/completions` | anthropic | translate via IR |
| `/v1/chat/completions` | openai_responses | translate via IR |
| `/v1/responses` | openai_responses | passthrough |
| `/v1/responses` | anthropic | translate via IR |
| `/v1/responses` | openai_chat | translate via IR |

## Usage Examples

### Anthropic client -> OpenAI-backed model

```bash
curl http://localhost:7860/v1/messages \
  -H "Authorization: Bearer sk-proxy-XXXX" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-5.4",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### OpenAI client -> Anthropic-backed model

```bash
curl http://localhost:7860/v1/chat/completions \
  -H "Authorization: Bearer sk-proxy-XXXX" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-opus-4-7",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Logging

Every line is one JSON object on stdout with `event`, `level` and `timestamp`. Lines emitted while serving a request also carry `request_id`, which is echoed in the `x-request-id` response header. An inbound `x-request-id` is reused only when it matches `^[A-Za-z0-9._-]{1,128}$`; otherwise a new id is generated.

`log_level` is `debug`, `info` (default), `warning` or `error`. Precedence: `--log-level` flag > `log_level` in config > `info`.

### Info

- `startup` — once at boot: `version`, `host`, `port`, `log_level`, and `providers` (each `key`, `format`, `discover`). No URLs or secrets.
- `request_completed` — exactly once per HTTP request except `/health`, when the response body finishes, when the client disconnects, or when the app raises. Fields are present only when set:

| Field | Meaning |
|-------|---------|
| `method`, `path`, `status` | The inbound request and the status sent |
| `duration_ms` | Time from request start to completion |
| `route` | `passthrough` (same format) or `translate` |
| `provider`, `model` | Provider key and the model id the client sent |
| `stream` | Whether the client asked for a stream |
| `ttft_ms` | Streams only: time to the first body bytes sent |
| `input_tokens`, `output_tokens` | Translated responses only, from the upstream usage |
| `error_type` | Set on failure: `invalid_request`, `upstream_connect`, `upstream_timeout`, `upstream_protocol`, `upstream_status`, `translation_error`, or the exception class name |
| `client_disconnected` | `true` when the client went away before the response finished |

- `models_refreshed` — after each model-list refresh: `total`, `elapsed_ms`.

Unexpected translation failures additionally log one `error`-level line with a traceback (`translate_request_error`, `translate_response_error`, `stream_translate_error`).

### Debug

`request_completed` also carries `inbound_format`, `provider_format`, `upstream_model`, `upstream_status`, `upstream_request_id`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `reasoning_tokens`, `raw_stop_reason`, `dropped_params`, `session_id` (from `x-claude-code-session-id`) and `client`, when set. Four stage events are added per request:

- `request_received` — `headers` (values of `authorization`, `proxy-authorization`, `x-api-key`, `api-key`, `cookie` and `set-cookie` replaced by `***`), `query`, `body_bytes`, `body_preview` (first 2048 characters).
- `upstream_request` — `provider`, `path`, `upstream_model`, `stream`, `body_bytes`, and the forwarded header names (names only).
- `upstream_response` — `status`, `ttfb_ms`, `request-id`, every `*ratelimit*` response header, and on status >= 400 `error_preview` (first 1024 characters of the body).
- `stream_summary` — streams only: `chunks_in` (upstream chunks), `chunks_out` (chunks sent), `dropped_events`, `keepalives`.

Nothing is logged per chunk or per token at any level, and debug detail is only computed when debug is enabled.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ROSETTA_CONFIG` | Path to config.json (default: `~/.rosetta-llm/config.json`) |
| Provider-specific | Set via `api_key_env` in config (e.g., `ANTHROPIC_API_KEY`) |

## Development

```bash
uv sync --group dev
uv run pytest -q
uv run pyrefly check
uv run ruff check src/ tests/
```
