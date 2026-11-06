# Anthropic ↔ OpenAI translation audit (vs shunt, 2026-09-09)

Scope: how faithfully this proxy translates Anthropic Messages ↔ OpenAI shapes, focused on deferred tools (`defer_loading`, `tool_search`, `tool_reference`, `server_tool_use`) and tool rewriting. All `file:line` cites are against `c021937`. Test counts below ran green on that revision (`uv run pytest`, no network). Evidence labels: `measured` = probed locally, `derived` = read off source.

## Directions

All 3×3 format pairs route through codec dicts at `src/rosetta/pipeline.py:51-80` (`_PARSE_REQ`/`_RENDER_REQ`/`_PARSE_RESP`/`_RENDER_RESP`/`_STREAM_PARSE`/`_STREAM_RENDER`), including anthropic→chat, anthropic→responses, chat→anthropic, responses→anthropic. The passthrough fast path (`:155`, `inbound_format == provider.format`) skips all codec work: no IR, no parse, no validation — it also skips every refusal the codec would raise (an anthropic→anthropic hop forwards `server_tool_use` blocks verbatim). Translation errors surface as 400 `translation_error` on requests, 502 on responses, inline SSE error events on streams.

## Deferred tools

- `defer_loading` on tool defs parses into the IR flag `Tool.deferred` (`src/rosetta/codecs/anthropic.py:328`) and renders back on anthropic (`:477-478`) and responses wires (`src/rosetta/codecs/openai_responses.py:124-125`). The chat wire has no deferred concept at all — the flag is lost on any chat render (measured).
- `tool_reference` names inside a search result resolve against the request's own `tools[]` catalog at parse time (`:335`); resolution exists only there. A bare `tool_reference` block type falls into the unknown-block fallback and becomes a text part holding the Python dict repr (measured).
- Search history round-trips through `ServerToolUsePart` / `ToolSearchResultPart`; a responses `tool_search_call` with no `call_id` gets a synthesized `srvtoolu_<hex>` id that pairs its output (`src/rosetta/codecs/openai_responses.py:147`).
- When any tool is deferred and no search tool is named, a search entry is synthesized on both wires — anthropic `{"type": "tool_search_tool_regex_20251119"}` (`src/rosetta/codecs/anthropic.py:36-39,482-483`), responses `{"type":"tool_search","execution":"server"}` (`src/rosetta/codecs/openai_responses.py:506-507`); regex is the default because a bm25 preference cannot be known.
- bm25 degrades to regex across any responses hop: the variant lives only on the IR block name, and the responses wire has no variant field (`src/rosetta/codecs/openai_responses.py:541-552`, pinned by `test_bm25_variant_stays_in_ir_and_strips_from_responses_wire`). The parse side still honors a peer proxy's `search_variant`.
- BYOT / client-executed search (`execution: "client"`) is refused loudly with a 400 on anthropic-bound renders (`src/rosetta/codecs/openai_responses.py:140-142`, `src/rosetta/codecs/anthropic.py:461-464`) — but the chat render converts the same definition into a bogus plain function tool silently (measured): the refusal class is not closed over the chat hop.
- Search blocks (`server_tool_use` / `tool_search_tool_result`) vanish silently on any chat-bound render, stream and non-stream (measured) — the chat codec has no branch for them.

## Tool rewriting

- `input_schema` ↔ `parameters` passes by reference; the chat side only injects `properties: {}` when missing.
- `strict` is never parsed from the anthropic wire into IR (`src/rosetta/codecs/anthropic.py:321-334`): anthropic→openai renders lose it (responses gets `strict: false`, chat omits it). Conversely the `_raw` sidecar merge (`anthropic.py:472-473`) pollutes anthropic-bound tool defs with chat-shaped keys — a chat→anthropic hop emits `{"type":"function","function":{...},"strict":true,...}` on the anthropic wire (measured).
- `cache_control` on tools survives only anthropic→anthropic (via `_raw`); both openai renders drop it. Content-block `cache_control` does survive anthropic round-trips.
- `tool_choice` maps fully across all three wires, including `disable_parallel_tool_use` on the anthropic side — but that flag is dropped by both openai renders, and a responses `parallel_tool_calls` never reaches IR (silently dropped at parse).
- Call ids survive every hop verbatim. Responses item-level `id`s are regenerated (`fc_<uuid>`, `msg_<uuid>`, `resp_<uuid>`).
- Tool arguments are a JSON string in IR (`arguments_json_text`); non-JSON arguments degrade to `{"raw_arguments": <text>}` silently (`anthropic.py:85-91`). On the responses wire `function_call.arguments` is a string (an object there is the 400) and `tool_search_call.arguments` is an object (a string there is the 400).
- Tool-result ordering repair (`_enforce_tool_result_ordering`, `anthropic.py:401-405`) fixes chat `role=tool` history into anthropic's tool_result-before-text shape (pinned `tests/codecs/test_roundtrip.py:56-74`).
- Non-text content inside `tool_result` (e.g. images) is dropped on both openai renders (kept anthropic→anthropic).
- Hosted tool defs (e.g. `web_search_20250305`) are silently converted to ordinary client function tools on out-hops (measured) — only the content-block and search-def shapes are refused.

## Streaming gaps

- The responses stream renderer maps `stop_reason: tool_use` to `response.incomplete` (`src/rosetta/stream_codecs/openai_responses.py:410-412`) while the non-stream path maps it to `completed` (`src/rosetta/codecs/openai_responses.py:748-753`): streaming and non-streaming disagree on every non-`end_turn` stop.
- Refusal text is dropped in stream renders in both directions (non-stream chat↔anthropic carries it).
- The chat stream renderer emits a defensive terminal `[DONE]` (`src/rosetta/stream_codecs/openai_chat.py:283-294`); the responses stream renderer has no defensive emit — without a `MessageStopEvent` the stream ends with no `[DONE]` (measured, contradicts the CLAUDE.md claim).

## Test pins

`uv run pytest tests/codecs/test_deferred_tools.py -q` → 30 passed; full suite → 55 passed. The deferred-tools file pins the refusal matrix (BYOT, missing `execution`, lone `tool_search_output`, non-search `server_tool_use`, `web_search_tool_result`), variant degradation, catalog resolution, stream one-shot repetition, and `srvtoolu_` id pairing. No e2e test covers the responses-inbound or anthropic→responses paths; those are covered by codec/stream unit tests only.

## Comparison verdict

The server-side search model here is the half shunt does not implement (shunt runs the client-executed native `tool_search` protocol that this proxy refuses). The 3×3 matrix including chat-completions is coverage shunt lacks entirely. The gaps that matter for fidelity: bm25 degradation, the chat-bound BYOT hole, `strict`/`cache_control` loss outbound, the stream stop-reason divergence, and the missing defensive `[DONE]` on responses streams.

## Deletion trigger

Delete this file when the gap list above is empty or the codec design is replaced; nothing else cites it.
