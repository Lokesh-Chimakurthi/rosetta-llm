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
  ghcr.io/lokesh-chimakurthi/rosetta-llm:main
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

| Method | Path                        | Purpose                    |
| ------ | --------------------------- | -------------------------- |
| POST   | `/v1/messages`              | Anthropic Messages         |
| POST   | `/v1/messages/count_tokens` | Local tiktoken token count |
| POST   | `/v1/chat/completions`      | OpenAI Chat Completions    |
| POST   | `/v1/responses`             | OpenAI Responses           |
| GET    | `/v1/models`                | Merged model list          |
| GET    | `/health`                   | Liveness check             |
| GET    | `/providers`                | Provider status            |

## Model ID Format

Models are addressed as `<provider_key>/<model_name>` where `provider_key` matches a key in your config's `providers` section.

Examples:
- `abc/kimi-k2.5` — routes to the "abc" provider with model "kimi-k2.5"
- `anthropic/claude-opus-4-7` — routes to the "anthropic" provider
- `openai/gpt-5.4` — routes to the "openai" provider

### Shorthand and aliases

The `<provider>/<model>` form is the canonical lookup and always wins. When a request arrives without the provider prefix, Rosetta falls back to a tiered shorthand search across every configured model:

1. **Exact match** on the model `id`, any string in `aliases`, or a full-string match against `alias_pattern` (a Python regex).
2. **Case-insensitive substring match** against the model `id` and each alias — only used if no exact match was found.

If exactly one model matches, it resolves. If more than one matches in the same tier, the request fails with a 400 listing every candidate `provider/model` so the caller knows what to type to disambiguate.

This makes Rosetta a true drop-in for Claude Code: typing `haiku`, `claude-haiku`, or `claude-haiku-4-5` into the `/model` picker (or leaving Claude Code's built-in defaults like `ANTHROPIC_DEFAULT_OPUS_MODEL` untouched) all resolve to the same configured target as long as nothing else collides.

Example config that maps Claude Code's hard-coded default model slots at OpenRouter:

```jsonc
{
  "providers": {
    "openrouter": {
      "format": "openai_chat",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY",
      "models": [
        {
          "id": "anthropic/claude-opus-4.1",
          "aliases": ["claude-opus-4-7", "claude-opus-4-5"]
        },
        {
          "id": "anthropic/claude-sonnet-4.5",
          "alias_pattern": "^claude-sonnet-.*"
        },
        {
          "id": "openai/gpt-4o-mini",
          "aliases": ["claude-haiku-4-5"]
        }
      ]
    }
  }
}
```

## Claude Code Integration

Rosetta is a fully compatible [Claude Code LLM gateway](https://code.claude.com/docs/en/llm-gateway). Point Claude Code at Rosetta and all configured providers appear in the `/model` picker — including non-Anthropic models.

### Setup
```bash
export ANTHROPIC_BASE_URL=http://localhost:7860
export ANTHROPIC_AUTH_TOKEN=sk-proxy-XXXX   # if proxy auth is enabled
```

Or in Claude Code settings (`~/.claude/settings.json`):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:7860",
    "ANTHROPIC_AUTH_TOKEN": "sk-proxy-XXXX"
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

| Client Endpoint        | Provider Format  | Path             |
| ---------------------- | ---------------- | ---------------- |
| `/v1/messages`         | anthropic        | passthrough      |
| `/v1/messages`         | openai_chat      | translate via IR |
| `/v1/messages`         | openai_responses | translate via IR |
| `/v1/chat/completions` | openai_chat      | passthrough      |
| `/v1/chat/completions` | anthropic        | translate via IR |
| `/v1/chat/completions` | openai_responses | translate via IR |
| `/v1/responses`        | openai_responses | passthrough      |
| `/v1/responses`        | anthropic        | translate via IR |
| `/v1/responses`        | openai_chat      | translate via IR |

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

## Environment Variables

| Variable          | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `ROSETTA_CONFIG`  | Path to config.json (default: `~/.rosetta-llm/config.json`) |
| Provider-specific | Set via `api_key_env` in config (e.g., `ANTHROPIC_API_KEY`) |

## Development
```bash
uv sync --group dev
uv run pytest -q
uv run mypy src/
uv run ruff check src/ tests/
```
