---
type: architecture-concept
title: Canonical IR
description: Canonical request, response, and streaming-event intermediate representation with _raw sidecars that preserve format-specific fidelity across translation.
tags: [ir, canonical-model, translation, pydantic, streaming]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-6818c07f74329edf240e39bc
    resource: repo://src/rosetta/ir/events.py
  - id: openwiki-source-1980a5cac045950c45f16dcf
    resource: repo://src/rosetta/ir/helpers.py
  - id: openwiki-source-75b9c5aed846cad86656d530
    resource: repo://src/rosetta/ir/request.py
  - id: openwiki-source-a367ab9f570354db1230121a
    resource: repo://src/rosetta/ir/response.py
  - id: openwiki-source-05dd3552f31fbb4e671f95a6
    resource: repo://src/rosetta/pipeline.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# Canonical IR

The canonical intermediate representation (IR) is the single model-agnostic language all translation flows through, so any of the three wire formats (Anthropic Messages, OpenAI Chat Completions, OpenAI Responses) can convert to and from any other without pairwise converters.

## Responsibility and ownership

- The IR owns the normalized shape of requests (`CanonicalRequest`), responses (`CanonicalResponse`), and streaming (`CanonicalStreamEvent`), and every codec and stream-codec pair converts only between its own wire format and the IR, never directly between wire formats.
- `src/rosetta/ir/request.py` owns message, content-part, tool, reasoning, and sampling types, while `src/rosetta/ir/response.py` owns normalized output messages plus usage and stop info, and `src/rosetta/ir/events.py` owns the streaming event union consumed by the pipeline's `_translate_stream` fan-in/fan-out.
- `src/rosetta/ir/helpers.py` owns the single `_raw` attachment helper (`with_raw`), which is the only sanctioned way to carry lossy format-specific fields alongside normalized data.

## Request model

- `CanonicalRequest` carries `model`, `messages`, `system` (string or structured `TextPart` list), `tools`, `tool_choice`, `max_output_tokens`, `sampling`, `stop_sequences`, `reasoning`, `stream`, `metadata`, and `raw_extras` for unknown top-level wire fields that must round-trip untouched.
- `Message` has a four-valued role (`system`, `user`, `assistant`, `tool`) where the `tool` role exists only for the OpenAI Chat-style flow and Anthropic instead represents tool results as `ToolResultPart` blocks inside a user message.
- `ContentPart` is a discriminated union on `type` covering `text`, `image`, `document`, `tool_call`, `tool_result`, `reasoning`, and `refusal`, so pattern-matching on one field is sufficient to dispatch rendering in any codec.
- `ToolCallPart` always stores arguments as the JSON-serialized string `arguments_json_text` (default `"{}"`), which means codecs convert to and from wire objects at the boundary and the IR itself never has to handle dual string/object shapes.
- `ToolResultPart` nests its own `content_parts` list plus `call_id` and `is_error`, which lets one tool result carry mixed text/image/document payloads without flattening them into the parent message.
- `ReasoningPart` separates `text`, `signature`, `encrypted_content`, `reasoning_id`, and `summary` with a `visible`/`redacted` flag, which is what makes the Anthropic signature encoding (`encrypted_content@reasoning_id`) and the Responses `reasoning` item id round-trip losslessly.
- `Tool` normalizes definitions to `name`, `description`, `input_schema`, `kind` (`function` vs `hosted`), and `strict`, while `ReasoningConfig` normalizes `effort`, `budget_tokens`, `thinking_type` (`enabled`/`adaptive`/`disabled`), `summary`, and `include_encrypted` across providers that use budgets versus effort levels.
- `SamplingConfig` groups `temperature`, `top_p`, `top_k`, and `seed` independently of transport, and `raw_extras` plus per-object `_raw` sidecars together guarantee that unknown or provider-specific parameters survive a parse/render cycle.

## Response model

- `CanonicalResponse` carries `id`, `model`, `output_messages` (reusing the request-side `Message` type), `stop` (`StopInfo`), and `usage` (`Usage`), plus a private `_raw` sidecar for provider envelopes that have no normalized equivalent.
- `Usage` normalizes five counters (`input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `reasoning_tokens`) so Anthropic cache accounting and OpenAI token accounting map onto one shape.
- `StopInfo` keeps both the normalized reason (`normalized`, Anthropic vocabulary) and the wire value (`provider_raw`) plus the matched `stop_sequence`, which preserves debugging fidelity when Chat `finish_reason` values differ from Anthropic `stop_reason` values.

## Streaming events

- `CanonicalStreamEvent` is the union of `MessageStartEvent`, `PartStartEvent`, `PartDeltaEvent`, `PartStopEvent`, `MessageDeltaEvent`, `MessageStopEvent`, `PingEvent`, and `ErrorEvent`, giving streaming translation a fixed eight-event vocabulary regardless of which SSE dialect is on either side.
- `PartStartEvent` opens an indexed content block with `part_type` plus optional `call_id`/`name` for tool calls, `PartDeltaEvent` carries incremental `text`/`json`/`reasoning`/`signature` payloads by index, and `PartStopEvent` closes the block, which mirrors how Chat deltas, Responses output-item deltas, and Anthropic content-block deltas all delimit work differently on the wire.
- `MessageStartEvent` carries the model name, `MessageDeltaEvent` carries incremental `stop`/`usage` updates, `MessageStopEvent` terminates the message, `PingEvent` is a keepalive marker consumed only by the Anthropic renderer, and `ErrorEvent` carries `error_type` plus message for inline stream failures.

## `_raw` sidecar mechanism

- Every `_IRBase` subtype declares `_raw: dict` as a Pydantic `PrivateAttr` (never a serialized field), and every stream event type except `PingEvent` does the same, so sidecar data travels with the object through translation but never leaks into normalized validation or output unless a renderer explicitly reads it back.
- `with_raw(obj, raw)` attaches the sidecar via `object.__setattr__`, which is necessary because Pydantic v2 blocks normal attribute assignment for `PrivateAttr` defaults and the helper keeps all codecs using one attachment path.
- `_IRBase` sets `extra="ignore"` on input, which means unknown wire fields are dropped from normalized attributes by default and only survive when a codec explicitly copies them into `raw_extras` or `_raw`.
- Representative uses include Anthropic `cache_control` per content block, tool-definition extras (`defer_loading`, `type`, `cache_control`), and reasoning payloads that one format cannot express natively.

## Lifecycle and invariants

- Non-streaming translation follows inbound-bytes → `parse_request` → `CanonicalRequest` → provider `render_request` → upstream → provider `parse_response` → `CanonicalResponse` → inbound `render_response`, and streaming translation follows the same shape with `CanonicalStreamEvent` sequences instead of whole-body objects.
- The passthrough fast path deliberately bypasses IR allocation entirely and only rewrites the `model` field on a copied dict, so IR construction cost is paid only when `inbound_format != provider.format`.
- Recursive types (`ToolResultPart.content_parts`, `Message`, `CanonicalRequest`) require explicit `model_rebuild()` calls at module bottom, without which forward-reference resolution fails at import time.
- `ToolResultPart.content_parts` uses a mutable-default-safe `Field(default_factory=list)` throughout the IR, and `arguments_json_text` defaulting to `"{}"` keeps tool-call rendering total even when the wire omitted arguments.

## Representative tests

- `tests/codecs/test_roundtrip.py` exercises Anthropic request round-trips, tool-use input object preservation, tool-result reordering, Chat-to-Anthropic tool-call id preservation, Responses↔Anthropic reasoning signature round-trips, Chat-response→Anthropic conversion with usage mapping, unknown-param passthrough via `raw_extras`, and `max_tokens` synthesis when the source omits it.
