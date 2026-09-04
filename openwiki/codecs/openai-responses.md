---
type: codec-reference
title: OpenAI Responses Codec
description: Responses API parse/render rules for input and output items plus reasoning and compaction round-trips.
tags: [openai-responses, codec, translation, reasoning, tool-use]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-996593ade93e4e9f94eb4d14
    resource: repo://src/rosetta/codecs/openai_responses.py
generated: { by: "opencode", at: "2026-09-04T02:22:07.783Z" }
---

# OpenAI Responses Codec

`src/rosetta/codecs/openai_responses.py` converts OpenAI Responses `input`/`output` item lists to and from the canonical IR, handling the richest item vocabulary of the three formats including reasoning, compaction, and function-call items.

## Responsibility and ownership

- The codec owns all four non-streaming conversions for the `openai_responses` format, with `_parse_input_item` and `_parse_reasoning_item` covering the six input item types (`message`, `function_call`, `function_call_output`, `reasoning`, `compaction`, `item_reference`) and the parallel output-item branch in `parse_response`.
- It owns the Responses-side half of the lossless reasoning round-trip (the Anthropic side lives in the Anthropic codec), the `cm1#`-prefixed compaction carrier, and the `instructions`-versus-`system` bridge.
- Streaming Responses SSE lives separately in `stream_codecs/openai_responses.py` and is documented under streaming translation rather than here.

## Request parsing

- `input` items of type `message` become IR messages with `input_text`/`output_text` → `TextPart`, `input_image` data-URLs → base64 `ImagePart` versus remote URLs → `url` `ImagePart`, `input_file` → `DocumentPart` from `file_data`/`file_id`, `input_video` → a `"[Video input not supported by target]"` text placeholder, and `refusal` → `RefusalPart`, with unknown roles coerced to `user`.
- `function_call` items become assistant messages carrying one `ToolCallPart` (`call_id`, `name`, `arguments` defaulting to `"{}"`), while `function_call_output` items become `tool`-role messages carrying one `ToolResultPart` whose string output becomes one `TextPart` and whose list output keeps only text entries.
- `reasoning` items concatenate `summary[].text` entries with `type == summary_text` into `ReasoningPart.text`, keep `encrypted_content` and `id` verbatim, and mark visibility `redacted` when encrypted content exists without summary text versus `visible` otherwise, which preserves the distinction between hidden chain-of-thought and summarized reasoning.
- `compaction` items become redacted `ReasoningPart` objects whose `signature` carrier is `cm1#<encrypted>@<id>`, `item_reference` items become empty-text user messages that preserve only `_raw` because the proxy cannot resolve cross-item references, and unrecognized item types return `None` and are skipped entirely.
- Top-level `instructions` becomes IR `system` when it is a nonempty string, tool entries map `type` to IR `kind` (`function` versus `hosted` with hosted names defaulting to the wire type), `parameters` becomes `input_schema`, `reasoning.effort`/`summary` and `include: ["reasoning.encrypted_content"]` become `ReasoningConfig`, and unknown top-level keys land in `raw_extras`.

## Request rendering

- IR `system` renders back to top-level `instructions`, and every `ToolCallPart`, `ToolResultPart`, and `ReasoningPart` flushes any accumulated message content first so each of those parts becomes its own top-level `input` item in wire order rather than being nested inside a message.
- Tool calls render as `function_call` items with verbatim `call_id`/`name`/`arguments`, tool results render as `function_call_output` with text parts joined to one `output` string, and reasoning parts with a `cm1#` signature prefix render as `compaction` items while all other reasoning parts render as `reasoning` items with `id`, `encrypted_content`, and `summary` text entries.
- Plain text accumulates into `message` items via `_message_item` with `input_text` for user roles versus `output_text` otherwise (non-user/non-assistant/non-system roles coerced to `user`), images render as `input_image` with data-URL or remote forms plus `detail: auto`, documents render as `input_file` with `data:<media>;base64,...` payloads and a fixed `document.pdf` filename, and refusals render as `refusal` content entries.
- Tools render with hosted kinds recovering their wire `type` from `_raw`, `tool_choice` dicts with `name` normalize to `{"type": "function", ...}` while `any`/`required` collapse to `required`, `max_output_tokens`/sampling render only when set, and `raw_extras` merge with `setdefault` so normalized fields always win.

## Response conversion

- `parse_response` walks `output` items into assistant messages (`message` → text/refusal parts, `function_call` → `ToolCallPart`, `reasoning` via the shared helper, `compaction` → redacted carrier part), then derives normalized stop as `tool_use` when any tool call exists else `end_turn` for `completed`, `max_tokens` for `incomplete` with `max_output_tokens` reason, and `refusal` for `failed`.
- Usage maps `input_tokens`/`output_tokens` plus nested `cached_tokens`/`cache_write_tokens`/`reasoning_tokens` details into the five IR counters, and the full payload is retained in `_raw` for response-field passthrough.
- `render_response` emits text as `output_text` with empty `annotations`, tool calls as `function_call` with synthesized `fc_<hex>` ids plus completed status, reasoning with `id`/`encrypted_content`/`summary`, refusals inline, grouped text into `msg_<hex>` message items, normalized stop mapped back (`end_turn`/`tool_use` → `completed`, `max_tokens` → `incomplete`, `refusal` → `failed`), and a synthesized `resp_<hex>` id when the IR has none.

## Representative tests

- `tests/codecs/test_roundtrip.py` covers the Responses→Anthropic→Responses reasoning round-trip asserting `signature == "ENCRYPTED_BLOB@rs_abc"` mid-flight and lossless `id` plus `encrypted_content` recovery, which is the primary fidelity guarantee for this codec.
