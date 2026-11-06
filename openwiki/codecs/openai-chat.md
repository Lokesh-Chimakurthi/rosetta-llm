---
type: codec-reference
title: OpenAI Chat Codec
description: Chat Completions parse/render rules for system extraction, tool roles, degraded media, reasoning fields, and stop-reason mapping.
tags: [openai-chat, codec, translation, tool-use, stop-reasons]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-9491929169892770b24f5748
    resource: repo://src/rosetta/codecs/openai_chat.py
  - id: openwiki-source-8e7f5ab8aa7a0b79c6aa940c
    resource: repo://src/rosetta/stop_reasons.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# OpenAI Chat Codec

`src/rosetta/codecs/openai_chat.py` converts OpenAI Chat Completions bodies to and from the canonical IR, bridging Chat's flat message list and `tool`-role convention to the IR's structured parts and Anthropic-style tool results.

## Responsibility and ownership

- The codec owns all four non-streaming conversions for the `openai_chat` format, including Chat-specific concerns such as `system`/`developer` extraction, `tool`/`function` message handling, `image_url`/`input_audio`/`file` content parts, `reasoning_content`/`refusal` passthrough, and `stop` versus `stop_sequences` duality.
- It shares stop-reason vocabulary with the other codecs through `src/rosetta/stop_reasons.py` (`OPENAI_CHAT_STOP_IN` for `finish_reason` → normalized, `OPENAI_CHAT_STOP_OUT` for normalized → `finish_reason`).
- Streaming Chat SSE lives separately in `stream_codecs/openai_chat.py` and is documented under streaming translation rather than here.

## Request parsing

- `system` and `developer` roles are removed from the message list, their string or text-block content is concatenated with `\n\n` into IR `system`, and all other roles flow into normalized messages, which is why multi-message system prompts collapse to one IR field.
- Each `tool`/`function`-role message emits one IR message with role `tool` carrying a single `ToolResultPart` whose `call_id` comes from `tool_call_id`/`function_call_id` and whose content is the stringified message body, preserving the one-message-per-call shape Chat requires.
- Assistant messages collect string or list content into parts, append one `ToolCallPart` per `tool_calls` entry with `arguments` defaulting to `"{}"`, and additionally capture `reasoning_content`/`reasoning` as visible `ReasoningPart` text plus `refusal` as `RefusalPart`, so provider reasoning extensions survive translation.
- `image_url` parts distinguish `data:` URLs (decoded to base64 `ImagePart` with media type from the data-URL header) from remote URLs (kept as `source_type="url"`), while `input_audio` degrades to a text placeholder and `file` parts degrade to a `"[File attached: <name>]"` text placeholder because the IR has no native audio content type.
- Tool definitions unwrap `tools[].function` into IR `Tool` (`name`, `description`, `parameters` as `input_schema` with missing `properties` repaired to `{}`), `reasoning_effort` becomes `reasoning.effort`, `stop` accepts either a string or string array into `stop_sequences`, `max_tokens`/`max_completion_tokens` coalesce into `max_output_tokens`, and unknown top-level keys land in `raw_extras`.

## Request rendering

- IR `system` (string or `TextPart` list) renders as a single leading `{"role": "system", "content": ...}` message, and IR messages with role `system` are skipped thereafter so system text is never duplicated downstream.
- Every `ToolResultPart` found in any message is emitted first as its own `{"role": "tool", "tool_call_id", "content"}` message with multi-part results flattened to `\n\n`-joined text, which splits Anthropic-style bundled tool results into the one-message-per-call shape Chat upstreams expect.
- Assistant rendering collapses a single text part to a plain string `content`, keeps multi-part content as typed blocks, emits `""` when there is neither text nor tool calls (satisfying Chat's non-null content expectation), and re-attaches accumulated `reasoning_content` and `refusal` fields alongside `tool_calls` with `arguments` defaulting to `"{}"`.
- Images render back to `image_url` blocks (remote URLs verbatim, base64 as `data:<media>;base64,...`), tools render as `{"type": "function", "function": {...}}` with empty schemas repaired and `strict` only emitted when true, and `stop_sequences` render as a bare string for length one versus an array otherwise to match Chat's dual-typed `stop` field.
- `max_output_tokens` renders as `max_tokens`, sampling renders only when set (`temperature`, `top_p`, `seed`), `reasoning.effort` renders as `reasoning_effort`, and `raw_extras` merge with `setdefault` so normalized fields always win.

## Response conversion

- `parse_response` reads the first `choices` entry, converts string or list `message.content` to parts, captures `reasoning_content`/`reasoning` and `refusal`, converts each `tool_calls` entry to `ToolCallPart` with raw retained, maps `finish_reason` through `OPENAI_CHAT_STOP_IN` into normalized stop while keeping the raw value, and maps `prompt_tokens`/`completion_tokens` plus nested `cached_tokens`/`cache_write_tokens`/`reasoning_tokens` details into the five IR usage counters.
- `render_response` joins all text parts into one Chat string `content` (`None` when empty alongside tool calls, per Chat convention), re-emits tool calls with `type: function`, folds reasoning/refusal text back into their Chat fields, maps normalized stop through `OPENAI_CHAT_STOP_OUT` (defaulting to `stop`), synthesizes a `chatcmpl-<hex>` id when the IR has none, and carries through `system_fingerprint`/`service_tier`/`store` from `_raw` when present.
- Usage rendering always emits `prompt_tokens`/`completion_tokens`/`total_tokens` and only adds `prompt_tokens_details`/`completion_tokens_details` when cache or reasoning counters are nonzero, which keeps responses minimal for providers that do not report those fields.

## Representative tests

- `tests/codecs/test_roundtrip.py` covers Chat→Anthropic tool-call id preservation with input-object decoding, Chat-response→Anthropic conversion with usage and `stop`→`end_turn` mapping, and `max_tokens` synthesis behavior for the reverse direction, while `tests/test_e2e.py` covers Chat passthrough fidelity and Anthropic→Chat end-to-end translation.
