---
type: codec-reference
title: Anthropic Codec
description: Anthropic Messages parse/render rules including tool-result ordering, reasoning signature encoding, cache_control preservation, and max_tokens synthesis.
tags: [anthropic, codec, translation, tool-use, reasoning]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-776012b5f97460c2a8640c38
    resource: repo://src/rosetta/codecs/anthropic.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# Anthropic Codec

`src/rosetta/codecs/anthropic.py` converts Anthropic Messages request/response bodies to and from the canonical IR while preserving the wire-level invariants Claude and third-party Anthropic-compatible providers enforce.

## Responsibility and ownership

- The codec owns all four non-streaming conversions for the `anthropic` format (`parse_request`, `render_request`, `parse_response`, `render_response`) plus the shared block-level helpers `_parse_content_block` and `_render_content_part` used by both directions.
- It owns the reasoning signature encoding (`encrypted_content@reasoning_id` split on the last `@`), the `tool_result`-before-`text` ordering repair, the `tool_use.input` object/string bridge, and the `cache_control` sidecar round-trip.
- Streaming SSE for the same format lives separately in `stream_codecs/anthropic.py` and is documented under streaming translation rather than here.

## Request parsing

- Only `user` and `assistant` roles are admitted and string content becomes a single `TextPart` while list content dispatches block-by-block through `_parse_content_block`, so stray `system` entries in the message list are ignored in favor of the top-level `system` field.
- `system` accepts either a plain string or a list of text blocks normalized to `TextPart`, `tools` become IR `Tool` objects with the full source dict retained in `_raw`, and `thinking.type` plus `budget_tokens` become `reasoning.thinking_type`/`budget_tokens` while `output_config.effort` becomes `reasoning.effort`.
- `tool_use.input` (a JSON object on the wire) is serialized to `arguments_json_text` via orjson with `None` mapping to `"{}"` and pre-serialized strings passing through, which unifies the IR representation across all three formats.
- `tool_result` string content becomes one `TextPart` while list content recurses through `_parse_content_block`, preserving `tool_use_id` as `call_id` and `is_error` verbatim, and `thinking` blocks decode `signature` into the `(encrypted_content, reasoning_id)` pair while `redacted_thinking` becomes a `ReasoningPart` with `visibility="redacted"`.
- `image` sources distinguish `url` versus `base64` into `source_type`/`media_type`/`data`, `document` defaults to `application/pdf`, every parsed part retains its source dict in `_raw`, and any top-level key outside `_REQUEST_TOP_LEVEL_KEYS` is captured into `raw_extras` for verbatim re-emission.

## Request rendering

- Rendering always emits `max_tokens` and synthesizes `4096` when the IR has no limit because Anthropic rejects requests without it, which is what allows Chat-originated requests that never set a limit to succeed.
- `system` strings pass through while `TextPart` lists render back to Anthropic text blocks, `tool`-role IR messages are folded into `pending_tool_results` and flushed as user messages, and orphaned tool results with no following user message are appended as a trailing user message so no tool output is silently dropped.
- `_enforce_tool_result_ordering` stable-partitions every emitted user message so all `tool_result` blocks precede any `text` block, which auto-repairs Chat-style orderings that Anthropic would otherwise reject.
- `ToolCallPart` deserializes `arguments_json_text` back to a JSON object (falling back to `{"raw_arguments": text}` on invalid JSON so partial streaming remnants never crash rendering), single-text tool results collapse back to Anthropic's string shorthand while multi-part results render only text/image/document children, and reasoning parts re-encode `signature` from `encrypted_content`/`reasoning_id` when no explicit signature is stored.
- `cache_control` saved in `_raw` on text, image, document, tool_call, tool_result, and reasoning blocks is re-attached on render (refusal blocks carry no `cache_control`), tool definitions start from their full `_raw` dict before `name`/`description`/`input_schema` are overwritten (defaulting empty schemas to `{"type": "object", "properties": {}}`), and `raw_extras` are merged with `setdefault` so normalized fields always win over replayed unknowns.
- `tool_choice` normalizes Chat-style `{"type": "function", ...}` to Anthropic `{"type": "tool", "name": ...}` and maps `auto`/`any`+`required`/`none` to the Anthropic vocabulary while preserving `disable_parallel_tool_use` and passing through already-native dicts.

## Response conversion

- `parse_response` maps every `content` block through `_parse_content_block` into one assistant message, normalizes `stop_reason` through the shared `ANTHROPIC_STOP` table while retaining the raw value and `stop_sequence`, and extracts all four Anthropic usage counters including both cache fields.
- `render_response` re-renders every output part through `_render_content_part` and emits the standard `type: message` envelope with `stop_reason` mapped back through the same table, which is why Chat `stop` and Responses completions both surface as Anthropic `end_turn` and friends.

## Representative tests

- `tests/codecs/test_roundtrip.py` covers simple request round-trips, tool-use input remaining a JSON object after render, tool-result reordering enforcement, Chat→Anthropic tool-call id preservation, Responses↔Anthropic reasoning signature round-trips, unknown-param passthrough via `raw_extras`, and `max_tokens` synthesis when the source omits the field.
