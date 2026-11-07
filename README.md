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

### Command line

| Command | Does |
|---------|------|
| `rosetta-llm serve [options]` | Run the proxy. The default: `rosetta-llm [options]` is the same. |
| `rosetta-llm check [options]` | Load and validate the config with the same overrides, run the exposure check, print a summary without secrets or URLs. Exit status 0 when `serve` would start, 2 otherwise. |
| `rosetta-llm version` | Print the version (also `rosetta-llm --version`). |

Options for `serve` and `check` (each overrides the config; precedence: flag > config > default):

| Option | Default |
|--------|---------|
| `--config`, `-c PATH` | `$ROSETTA_CONFIG`, else `~/.rosetta-llm/config.json` |
| `--host HOST` | `127.0.0.1` |
| `--port`, `-p PORT` | `7860` |
| `--log-level {debug,info,warning,error}` | `info` |
| `--log-completed` / `--no-log-completed` | off (see [Logging](#logging)) |
| `--log-headers` / `--no-log-headers` | off; debug only |
| `--log-bodies` / `--no-log-bodies` | off; debug only |
| `--allow-unauthenticated` | off: a non-loopback host needs `proxy.api_keys` |
| `--proxy-headers` / `--no-proxy-headers` | off |
| `--forwarded-allow-ips IPS` | `127.0.0.1` |

`serve` only (server settings, not in the config):

| Option | Default |
|--------|---------|
| `--root-path PATH` | none — set when a reverse proxy mounts Rosetta under a sub-path |
| `--timeout-keep-alive SECONDS` | `5` |
| `--limit-concurrency N` | unlimited; beyond `N` connections clients get 503 |

A missing, unreadable or invalid config, or an unsafe bind, prints the reason and exits with status 2.

### Docker

The image binds `0.0.0.0` inside the container, so the mounted config must set `proxy.api_keys` (or `proxy.allow_unauthenticated: true` when another layer authenticates) — otherwise the proxy refuses to start. Publish the port on loopback unless other machines need it:

```json
{
  "proxy": {"api_keys": ["sk-proxy-change-me"]},
  "providers": {"...": "..."}
}
```

```bash
docker run -p 127.0.0.1:7860:7860 \
  -v ~/.rosetta-llm/config.json:/app/config.json \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e OPENAI_API_KEY=sk-... \
  ghcr.io/lokesh-chimakurthi/rosetta-llm:latest
```

Clients then send `Authorization: Bearer sk-proxy-change-me` or `x-api-key: sk-proxy-change-me`.

### Network exposure and authentication

- `host` defaults to `127.0.0.1`. Binding any non-loopback address with no `proxy.api_keys` is refused at startup (exit status 2) unless `proxy.allow_unauthenticated` is true.
- With keys configured, every request needs `Authorization: Bearer <key>` or `x-api-key: <key>` matching any key, except `/health`, `/docs`, `/docs/oauth2-redirect` and `/openapi.json`. A 401 uses the error envelope of the API the path belongs to (Anthropic for `/v1/messages*`, and for `/v1/models*` when `anthropic-version` is sent).
- `X-Forwarded-*` headers are ignored unless `proxy.trust_proxy_headers` is true, and then only from `proxy.forwarded_allow_ips` (default `127.0.0.1`).
- Precedence for each of these: CLI flag (`--host`, `--allow-unauthenticated`, `--proxy-headers/--no-proxy-headers`, `--forwarded-allow-ips`) > config > default.

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
- **Safe by default**: binds loopback; refuses unauthenticated non-loopback binds; `Bearer` or `x-api-key` proxy keys
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
| POST | `/v1/models/refresh` | Re-query discovering providers now |
| GET | `/health` | Liveness check |
| GET | `/providers` | Provider status |
| GET | `/docs` | Interactive API docs (Swagger UI; the browser loads its assets from FastAPI's default CDN) |
| GET | `/openapi.json` | OpenAPI schema: summaries, descriptions and error envelopes for every route |

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

One JSON object per line on stdout, each with `ts`, `level` and `event`. Lines logged while serving a request carry `request_id`, which is echoed in the `x-request-id` response header. An inbound `x-request-id` is reused only when it matches `^[A-Za-z0-9._-]{1,128}$`; otherwise a 16-hex-character id is generated. `/health` is never logged, and nothing is logged per chunk or per token.

| Setting | Config key | Flag | Default |
|---------|-----------|------|---------|
| Level: `debug`, `info`, `warning`, `error` | `log_level` | `--log-level` | `info` |
| Log `request_completed` too | `log_completed` | `--log-completed` / `--no-log-completed` | off |
| Add headers to debug events (credentials redacted) | `log_headers` | `--log-headers` / `--no-log-headers` | off |
| Add bodies to debug events (capped at 64 KiB) | `log_bodies` | `--log-bodies` / `--no-log-bodies` | off |

Precedence for each: flag > config > default. `log_headers` and `log_bodies` only take effect at `debug`.

### Info

```json
{"ts":"2026-09-30T13:31:07.402Z","level":"info","event":"startup","version":"0.2.2","host":"127.0.0.1","port":7860,"log_level":"info","providers":[{"key":"octo","format":"openai_chat","discover":true}]}
{"ts":"2026-09-30T13:31:17.661Z","level":"info","event":"request_started","request_id":"b4f81a35294d4639","path":"/v1/messages","model":"octo/deepseek-v4.1-flash","route":"translate","stream":true}
```

- `startup` — once at boot; providers carry `key`, `format`, `discover` (no URLs or secrets).
- `request_started` — once per request, when the model is resolved (or when the response starts for requests that are not model calls): `path`, and when known `model`, `route` (`passthrough` or `translate`), `stream`.
- `models_refreshed` — after each model-list refresh: `total`, `elapsed_ms`.
- `translation_failed` — `error` level, when a request or response cannot be translated: `direction`, `error`. The traceback is added at `debug`.

With `log_completed`, each request also logs `request_completed` — at `info` for 2xx/3xx, `warning` for 4xx, `error` for 5xx — when the response finishes, the client disconnects, or the app raises:

```json
{"ts":"2026-09-30T13:31:38.385Z","level":"info","event":"request_completed","request_id":"b4f81a35294d4639","path":"/v1/messages","status":200,"duration_ms":20725,"model":"octo/deepseek-v4.1-flash","route":"translate","stream":true,"ttft_ms":2568,"input_tokens":115104,"output_tokens":1860,"stop_reason":"end_turn"}
{"ts":"2026-09-30T13:32:01.118Z","level":"warning","event":"request_completed","request_id":"9c2e71d0a4f84b21","path":"/v1/messages","status":429,"duration_ms":312,"model":"octo/deepseek-v4-pro","route":"translate","stream":false,"error":"upstream_status","upstream_status":429}
```

| Field | Meaning |
|-------|---------|
| `status`, `duration_ms` | Status sent and total time |
| `ttft_ms` | Streams only: time to the first bytes sent |
| `input_tokens`, `output_tokens`, `stop_reason` | Translated responses only, from the upstream usage and stop reason |
| `error` | `invalid_request`, `upstream_connect`, `upstream_timeout`, `upstream_protocol`, `upstream_status`, `translation_error`, or the exception class name |
| `upstream_status` | With `error`, the status the provider returned |
| `client_disconnected` | `true` when the client left before the response finished |

### Debug

Adds `upstream_request` and `upstream_response`; with `log_completed`, `request_completed` also carries `cache_read_tokens`, `cache_creation_tokens`, `reasoning_tokens`, `upstream_stop_reason`, `upstream_status`, `session_id` (from `x-claude-code-session-id`) and `dropped_fields`. `dropped_fields` lists, by name only, the request fields translation dropped because the target format doesn't define them. Between Chat and Responses, Rosetta forwards only the fields both define the same way: `parallel_tool_calls`, `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, `safety_identifier`, `service_tier` and `store`. It forwards none between Anthropic and either OpenAI format.

```json
{"ts":"2026-09-30T13:31:17.664Z","level":"debug","event":"upstream_request","request_id":"b4f81a35294d4639","provider":"octo","path":"/v1/chat/completions","upstream_model":"deepseek-v4.1-flash","stream":true,"body_bytes":48211}
{"ts":"2026-09-30T13:31:20.074Z","level":"debug","event":"upstream_response","request_id":"b4f81a35294d4639","status":200,"ttfb_ms":2410}
```

`upstream_response` carries `upstream_request_id` when the provider sends a `request-id` or `x-request-id` header. With `log_headers`, `request_started`, `upstream_request` and `upstream_response` gain `headers`; values of `authorization`, `proxy-authorization`, `x-api-key`, `api-key`, `cookie` and `set-cookie` are replaced by `***`. With `log_bodies` they gain `body` (the first 64 KiB), `body_bytes`, and `body_truncated: true` when cut; a successful streaming response body is never logged.

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
