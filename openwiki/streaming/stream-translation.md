---
type: architecture-concept
title: Streaming Translation
description: SSE state machines that translate streaming responses through CanonicalStreamEvent with ping keepalives, DONE guarantees, and disconnect handling.
tags: [streaming, sse, translation, keepalive, cancellation]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-04T02:30:17.829Z
sources:
  - id: openwiki-source-05dd3552f31fbb4e671f95a6
    resource: repo://src/rosetta/pipeline.py
  - id: openwiki-source-2edcb99449bca3f4b80824d3
    resource: repo://src/rosetta/stream_codecs/anthropic.py
  - id: openwiki-source-a49989c20b885883c8cdca5e
    resource: repo://src/rosetta/stream_codecs/openai_chat.py
  - id: openwiki-source-883b15902209141cb2ca4110
    resource: repo://src/rosetta/stream_codecs/openai_responses.py
  - id: openwiki-source-5dfdd5645d86935ac47030b1
    resource: repo://src/rosetta/upstream.py
generated: { by: "opencode", at: "2026-09-04T02:30:17.829Z" }
---

# Streaming Translation

Streaming translation chains a provider-side SSE parser, an optional Anthropic keepalive stage, and an inbound-side SSE renderer as lazy async generators, so bytes flow chunk-by-chunk with constant memory and either side can use any of the three SSE dialects.

## Responsibility and ownership

- `stream_codecs/anthropic.py`, `stream_codecs/openai_chat.py`, and `stream_codecs/openai_responses.py` each own a `parse` (wire SSE bytes → `CanonicalStreamEvent`) plus a `render` (`CanonicalStreamEvent` → wire SSE bytes) pair, while the pipeline owns chaining them together with `wrap_with_ping` selection, disconnect polling, provider-status recording, and stream error mapping.
- `ir/events.py` owns the eight-event vocabulary the three codecs share, and `UpstreamClient.stream` owns the raw byte source including surfacing 4xx/5xx error bodies as `httpx.HTTPStatusError` for the parsers to convert downstream.
- Non-streaming whole-body conversion lives separately in `codecs/` and shares only the IR event/response/helper types plus the stop-reason tables, never whole-body parse/render logic, with these state machines.

## Chaining and lifecycle

- `_translate_stream` builds `upstream.stream(...)` → provider `_STREAM_PARSE` → optional `wrap_with_ping` (Anthropic inbound only) → inbound `_STREAM_RENDER` and yields each rendered chunk after polling `request.is_disconnected()`, which keeps backpressure end-to-end and aborts cleanly by breaking out of the generator and exiting the `httpx.stream` context.
- Transport `httpx.HTTPError` maps to an inbound-format `upstream_error` stream event while any other exception maps to `translation_error`, and loop exit without transport error records provider ok status once after the final chunk.
- Passthrough streaming bypasses all three stages and yields upstream bytes directly behind disconnect polling plus the same transport-error recovery, so same-format streams pay no parsing cost.
- All streaming responses carry `Cache-Control: no-cache`, `Connection: keep-alive`, and `X-Accel-Buffering: no` to defeat proxy buffering.

## Parser state machines

- The Anthropic parser buffers on `\n\n` frames, dispatches on `event:`/`data:` lines with `data.type` fallback, maps `message_start`/`content_block_start`/`content_block_delta`/`content_block_stop`/`message_delta`/`message_stop`/`ping`/`error` to the matching IR events (text/thinking/tool_use starts distinguished with call `id`/`name` preserved, text/input_json/thinking/signature deltas distinguished by delta type, usage plus normalized stop carried on message deltas), and skips undecodable JSON lines without failing the stream.
- The Chat parser tracks `started`, `text_block_open`, `reasoning_block_open`, and seen tool indices, emits `MessageStartEvent` from the first data chunk, converts `reasoning_content`/`reasoning` deltas to reasoning blocks that close before text opens, offsets tool-call indices by one to avoid colliding with text block zero, accumulates partial `arguments` strings as `json` deltas, converts usage-only chunks and `finish_reason` arrivals to `MessageDeltaEvent` with normalized stop, and closes open blocks plus all seen tools before emitting the delta.
- The Responses parser tracks the current output index plus per-index whitespace counters, maps `response.created` to start, `response.output_item.added` for message/function_call/reasoning to typed `PartStartEvent`, text/reasoning/argument deltas to matching `PartDeltaEvent`, item-done events to `PartStopEvent`, `response.completed`/`incomplete` to `MessageDeltaEvent` with `end_turn` versus `max_tokens` derived from `incomplete_details.reason`, and `response.failed`/`error` to `ErrorEvent`, including a whitespace-runaway guard that emits `invalid_request_error` when a single tool-call argument stream exceeds 20 consecutive whitespace characters.
- All three parsers treat `[DONE]` as `MessageStopEvent` (Chat and Responses) or ignore it (Anthropic has no such sentinel), and malformed JSON lines are skipped rather than aborting the stream.

## Renderer guarantees

- The Anthropic renderer synthesizes `msg_<hex>` ids (reusing `_raw.id` when present), maps IR part types to `text`/`thinking`/`tool_use` content blocks with `input: {}` initialization, maps delta types back to the four Anthropic delta shapes, emits usage deltas with `output_tokens` always and `input_tokens` only when nonzero, and renders pings and errors in Anthropic framing.
- The Chat renderer emits a role-only chunk on first text/tool activity for clients that expect it, compacts IR tool indices to dense OpenAI indices, carries `reasoning` deltas as `reasoning_content`, buffers the final usage plus mapped `finish_reason` until `MessageStopEvent`, and defensively emits a terminal chunk plus `[DONE]` even when `MessageStopEvent` never arrives.
- The Responses renderer emits `response.created` with `resp_stream` (or `_raw.id` when present), typed `response.output_item.added` frames with synthesized `msg_`/`fc_`/`rs_` ids, matching text/argument/reasoning deltas, generic `response.output_item.done` closers, `response.completed` versus `response.incomplete` derived from normalized stop with usage totals, and a bare `[DONE]` terminator.

## Ping injection and cancellation

- `wrap_with_ping` runs the source generator in a producer task feeding a queue and parks its consumer with a 15-second `asyncio.wait_for`, yielding a synthetic `PingEvent` on each timeout so Anthropic SDK clients do not disconnect during slow upstream generation, while never cancelling the source mid-await and propagating source exceptions downstream after draining.
- Cancellation is cooperative at the pipeline layer: every emitted chunk checks `request.is_disconnected()` first and breaks, which unwinds through the renderer and parser generators into `UpstreamClient.stream`'s context exit and closes the upstream connection.

## Representative tests

- `tests/test_e2e.py` covers Chat stream passthrough asserting `[DONE]` framing plus `hi` payload survival through the byte path, while `tests/codecs/test_roundtrip.py` covers the partial-JSON accumulation premise that streaming tool-argument deltas rely on.
