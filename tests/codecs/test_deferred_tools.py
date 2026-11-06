"""Deferred-tools / tool-search parity tests for the Anthropic and Responses codecs."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import pytest

from rosetta.codecs import anthropic as ac
from rosetta.codecs import openai_responses as orc
from rosetta.ir.request import ToolSearchResultPart
from rosetta.stream_codecs import anthropic as ac_stream
from rosetta.stream_codecs import openai_responses as or_stream

_SEARCH_TOOL = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
_WEATHER_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}}


def _weather_tool(**extras: object) -> dict[str, object]:
    tool: dict[str, object] = {
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": _WEATHER_SCHEMA,
    }
    tool.update(extras)
    return tool


def test_anthropic_deferred_tools_render_as_responses_tool_search() -> None:
    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [_weather_tool(defer_loading=True), _SEARCH_TOOL],
    }
    rendered = orc.render_request(ac.parse_request(payload))

    search_entries = [t for t in rendered["tools"] if t.get("type") == "tool_search"]
    assert len(search_entries) == 1
    assert search_entries[0]["execution"] == "server"

    weather = next(t for t in rendered["tools"] if t.get("name") == "get_weather")
    assert weather["type"] == "function"
    assert weather.get("defer_loading") is True
    assert weather["parameters"] == _WEATHER_SCHEMA


def test_deferred_tools_synthesize_search_entry_both_directions() -> None:
    anthropic_payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_weather_tool(defer_loading=True)],
    }
    rendered_responses = orc.render_request(ac.parse_request(anthropic_payload))
    assert any(t.get("type") == "tool_search" for t in rendered_responses["tools"])

    responses_payload = {
        "model": "gpt-5.5",
        "input": [],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": _WEATHER_SCHEMA,
                "defer_loading": True,
            }
        ],
    }
    rendered_anthropic = ac.render_request(orc.parse_request(responses_payload))
    search_entries = [
        t
        for t in rendered_anthropic["tools"]
        if t.get("type") == "tool_search_tool_regex_20251119"
    ]
    assert len(search_entries) == 1
    assert search_entries[0]["name"] == "tool_search_tool_regex"
    assert "input_schema" not in search_entries[0]
    weather = next(t for t in rendered_anthropic["tools"] if t.get("name") == "get_weather")
    assert weather.get("defer_loading") is True


def test_responses_tool_search_entry_renders_as_anthropic_search_tool() -> None:
    payload = {
        "model": "gpt-5.5",
        "input": [],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": _WEATHER_SCHEMA,
                "defer_loading": True,
            },
            {"type": "tool_search", "execution": "server"},
        ],
    }
    rendered = ac.render_request(orc.parse_request(payload))
    search_entries = [
        t for t in rendered["tools"] if t.get("type") == "tool_search_tool_regex_20251119"
    ]
    assert len(search_entries) == 1
    assert search_entries[0]["name"] == "tool_search_tool_regex"


def test_responses_tool_search_output_to_anthropic_blocks() -> None:
    payload = {
        "id": "resp_x",
        "object": "response",
        "model": "gpt-5.5",
        "status": "completed",
        "output": [
            {
                "type": "tool_search_call",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "arguments": {"pattern": "weather"},
            },
            {
                "type": "tool_search_output",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": _WEATHER_SCHEMA,
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    rendered = ac.render_response(orc.parse_response(payload))
    blocks = rendered["content"]
    assert [b["type"] for b in blocks] == [
        "server_tool_use",
        "tool_search_tool_result",
        "tool_use",
    ]

    server_use = blocks[0]
    assert re.fullmatch(r"srvtoolu_[a-zA-Z0-9_]+", server_use["id"])
    assert server_use["name"] == "tool_search_tool_regex"
    assert server_use["input"] == {"pattern": "weather"}

    search_result = blocks[1]
    assert search_result["tool_use_id"] == server_use["id"]
    assert search_result["content"] == {
        "type": "tool_search_tool_search_result",
        "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
    }

    assert blocks[2]["id"] == "call_1"
    assert blocks[2]["input"] == {"city": "Paris"}


def test_anthropic_tool_search_blocks_to_responses_output() -> None:
    payload = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_abc123",
                "name": "tool_search_tool_regex",
                "input": {"pattern": "weather"},
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_abc123",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
                },
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    rendered = orc.render_response(ac.parse_response(payload))
    items = rendered["output"]
    assert [i["type"] for i in items] == [
        "tool_search_call",
        "tool_search_output",
        "function_call",
    ]

    call = items[0]
    assert call["execution"] == "server"
    assert call["call_id"] is None
    assert call["arguments"] == {"pattern": "weather"}

    output = items[1]
    assert output["execution"] == "server"
    assert output["call_id"] is None
    assert [t["name"] for t in output["tools"]] == ["get_weather"]

    assert items[2]["call_id"] == "toolu_1"


def test_anthropic_tool_search_blocks_to_responses_input() -> None:
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_abc123",
                        "name": "tool_search_tool_regex",
                        "input": {"pattern": "weather"},
                    },
                    {
                        "type": "tool_search_tool_result",
                        "tool_use_id": "srvtoolu_abc123",
                        "content": {
                            "type": "tool_search_tool_search_result",
                            "tool_references": [
                                {"type": "tool_reference", "tool_name": "get_weather"}
                            ],
                        },
                    },
                ],
            }
        ],
        "tools": [_weather_tool(defer_loading=True), _SEARCH_TOOL],
    }
    rendered = orc.render_request(ac.parse_request(payload))
    items = rendered["input"]
    assert [i["type"] for i in items] == ["tool_search_call", "tool_search_output"]
    assert items[0]["call_id"] is None
    assert items[1]["call_id"] is None
    tools = items[1]["tools"]
    assert [t["name"] for t in tools] == ["get_weather"]
    assert tools[0]["parameters"] == _WEATHER_SCHEMA


def test_responses_tool_search_input_to_anthropic_blocks() -> None:
    payload = {
        "model": "gpt-5.5",
        "input": [
            {
                "type": "tool_search_call",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "arguments": {"pattern": "weather"},
            },
            {
                "type": "tool_search_output",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": _WEATHER_SCHEMA,
                    }
                ],
            },
        ],
    }
    rendered = ac.render_request(orc.parse_request(payload))
    assert rendered["messages"]
    message = rendered["messages"][0]
    assert message["role"] == "assistant"
    blocks = message["content"]
    assert [b["type"] for b in blocks] == ["server_tool_use", "tool_search_tool_result"]
    assert re.fullmatch(r"srvtoolu_[a-zA-Z0-9_]+", blocks[0]["id"])
    assert blocks[1]["tool_use_id"] == blocks[0]["id"]
    assert blocks[1]["content"]["tool_references"] == [
        {"type": "tool_reference", "tool_name": "get_weather"}
    ]


async def test_stream_tool_search_items_render_as_anthropic_blocks() -> None:
    sse = (
        b'data: {"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress","output":[]}}\n\n'
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"tool_search_call","execution":"server","call_id":null,"status":"in_progress","arguments":{}}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"tool_search_call","execution":"server","call_id":null,"status":"completed","arguments":{"pattern":"w"}}}\n\n'
        b'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"tool_search_output","execution":"server","call_id":null,"status":"in_progress","tools":[]}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":1,"item":{"type":"tool_search_output","execution":"server","call_id":null,"status":"completed","tools":[{"type":"function","name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]}}\n\n'
        b'data: {"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"","status":"in_progress"}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\\"ci"}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","output_index":2,"delta":"ty\\":\\"Paris\\"}"}\n\n'
        b'data: {"type":"response.function_call_arguments.done","output_index":2,"arguments":"{\\"city\\":\\"Paris\\"}"}\n\n'
        b'data: {"type":"response.completed","response":{"id":"r1","model":"m1","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    out = b"".join([event async for event in ac_stream.render(or_stream.parse(chunks()))])

    starts: list[dict[str, Any]] = []
    deltas: list[str] = []
    for block in out.split(b"\n\n"):
        for line in block.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            data = json.loads(line[6:])
            if data.get("type") == "content_block_start":
                starts.append(data["content_block"])
            elif (
                data.get("type") == "content_block_delta"
                and data["delta"].get("type") == "input_json_delta"
            ):
                deltas.append(data["delta"]["partial_json"])

    assert [s["type"] for s in starts] == [
        "server_tool_use",
        "tool_search_tool_result",
        "tool_use",
    ]
    assert starts[0]["input"] == {"pattern": "w"}
    assert starts[1]["tool_use_id"] == starts[0]["id"]
    assert starts[1]["content"]["tool_references"] == [
        {"type": "tool_reference", "tool_name": "get_weather"}
    ]
    assert "".join(deltas) == '{"city":"Paris"}'


async def test_stream_anthropic_search_blocks_render_as_responses_items() -> None:
    sse = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_abc123","name":"tool_search_tool_regex","input":{"pattern":"w"}}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_abc123","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"get_weather"}]}}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"Paris\\"}"}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":2}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    out = b"".join([event async for event in or_stream.render(ac_stream.parse(chunks()))])

    added: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    deltas: list[str] = []
    for block in out.split(b"\n\n"):
        for line in block.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if data.get("type") == "response.output_item.added":
                added.append(data["item"])
            elif data.get("type") == "response.output_item.done":
                done.append(data)
            elif data.get("type") == "response.function_call_arguments.delta":
                deltas.append(data["delta"])

    assert [i["type"] for i in added] == [
        "tool_search_call",
        "tool_search_output",
        "function_call",
    ]
    assert added[0]["call_id"] is None
    assert added[0]["arguments"] == {"pattern": "w"}
    assert [t["name"] for t in added[1]["tools"]] == ["get_weather"]
    assert "".join(deltas) == '{"city":"Paris"}'

    # Search blocks are one-shot: their `done` events repeat the item so a
    # downstream parser reading only `done` keeps the payload.
    search_done = [
        d
        for d in done
        if d.get("item", {}).get("type") in ("tool_search_call", "tool_search_output")
    ]
    assert len(search_done) == 2
    assert search_done[0]["item"]["arguments"] == {"pattern": "w"}
    assert [t["name"] for t in search_done[1]["item"]["tools"]] == ["get_weather"]


def test_byot_tool_search_items_refuse_parse() -> None:
    payload = {
        "id": "resp_x",
        "object": "response",
        "model": "gpt-5.5",
        "status": "completed",
        "output": [
            {
                "type": "tool_search_call",
                "execution": "client",
                "call_id": "ts_9",
                "status": "completed",
                "arguments": {"pattern": "w"},
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    with pytest.raises(ValueError, match="client-executed tool search"):
        orc.parse_response(payload)


def test_tool_search_item_without_execution_refuses_parse() -> None:
    payload = {
        "id": "resp_x",
        "object": "response",
        "model": "gpt-5.5",
        "status": "completed",
        "output": [
            {
                "type": "tool_search_call",
                "call_id": "ts_9",
                "status": "completed",
                "arguments": {"pattern": "w"},
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    with pytest.raises(ValueError, match="without execution"):
        orc.parse_response(payload)


def test_client_execution_search_entry_refuses_anthropic_render() -> None:
    payload = {
        "model": "gpt-5.5",
        "input": [],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": _WEATHER_SCHEMA,
                "defer_loading": True,
            },
            {
                "type": "tool_search",
                "execution": "client",
                "description": "client-side search",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
    }
    with pytest.raises(ValueError, match="client-executed tool search"):
        ac.render_request(orc.parse_request(payload))


def test_lone_tool_search_output_refuses_parse() -> None:
    payload = {
        "model": "gpt-5.5",
        "input": [
            {
                "type": "tool_search_output",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "tools": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="without a preceding tool_search_call"):
        orc.parse_request(payload)


def test_search_blocks_preserve_cache_control() -> None:
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_abc123",
                        "name": "tool_search_tool_regex",
                        "input": {},
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "tool_search_tool_result",
                        "tool_use_id": "srvtoolu_abc123",
                        "content": {
                            "type": "tool_search_tool_search_result",
                            "tool_references": [],
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ],
    }
    rendered = ac.render_request(ac.parse_request(payload))
    blocks = rendered["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}


async def test_stream_tool_search_without_execution_refuses_parse() -> None:
    sse = (
        b'data: {"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress","output":[]}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"tool_search_call","call_id":"ts_9","status":"completed","arguments":{"pattern":"w"}}}\n\n'
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    with pytest.raises(ValueError, match="without execution"):
        _ = [event async for event in or_stream.parse(chunks())]


async def test_stream_byot_tool_search_refuses_parse() -> None:
    sse = (
        b'data: {"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress","output":[]}}\n\n'
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"tool_search_call","execution":"client","call_id":"ts_9","status":"in_progress","arguments":{}}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"tool_search_call","execution":"client","call_id":"ts_9","status":"completed","arguments":{"pattern":"w"}}}\n\n'
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    with pytest.raises(ValueError, match="client-executed tool search"):
        _ = [event async for event in or_stream.parse(chunks())]


def test_bm25_variant_stays_in_ir_and_strips_from_responses_wire() -> None:
    payload = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_abc123",
                "name": "tool_search_tool_bm25",
                "input": {"query": "weather"},
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    ir = ac.parse_response(payload)
    assert ir.output_messages[0].parts[0].name == "tool_search_tool_bm25"

    rendered_responses = orc.render_response(ir)
    call = rendered_responses["output"][0]
    assert "search_variant" not in call

    # Responses has no variant channel: bm25 degrades to regex across the wire.
    rendered_back = ac.render_response(orc.parse_response(rendered_responses))
    block = rendered_back["content"][0]
    assert block["name"] == "tool_search_tool_regex"

    # Anthropic-to-anthropic keeps the variant via the block name.
    anthropic_roundtrip = ac.render_response(ir)
    assert anthropic_roundtrip["content"][0]["name"] == "tool_search_tool_bm25"


def test_regex_search_variant_is_the_default() -> None:
    payload = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_abc123",
                "name": "tool_search_tool_regex",
                "input": {"pattern": "weather"},
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    rendered_responses = orc.render_response(ac.parse_response(payload))
    assert "search_variant" not in rendered_responses["output"][0]

    rendered_back = ac.render_response(orc.parse_response(rendered_responses))
    assert rendered_back["content"][0]["name"] == "tool_search_tool_regex"


def test_catalog_resolved_tools_preserve_raw_extras() -> None:
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_search_tool_result",
                        "tool_use_id": "srvtoolu_abc123",
                        "content": {
                            "type": "tool_search_tool_search_result",
                            "tool_references": [
                                {"type": "tool_reference", "tool_name": "get_weather"}
                            ],
                        },
                    }
                ],
            }
        ],
        "tools": [
            _weather_tool(
                defer_loading=True,
                cache_control={"type": "ephemeral"},
                vendor_extra="keep-me",
            )
        ],
    }
    ir = ac.parse_request(payload)
    result_part = ir.messages[0].parts[0]
    assert isinstance(result_part, ToolSearchResultPart)
    resolved = result_part.tools[0]
    assert resolved._raw.get("cache_control") == {"type": "ephemeral"}
    assert resolved._raw.get("vendor_extra") == "keep-me"

    rendered = ac.render_request(ir)
    tool = next(t for t in rendered["tools"] if t.get("name") == "get_weather")
    assert tool.get("cache_control") == {"type": "ephemeral"}
    assert tool.get("vendor_extra") == "keep-me"
    assert tool.get("defer_loading") is True


async def test_stream_tool_search_output_preserves_full_tools() -> None:
    sse = (
        b'data: {"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress","output":[]}}\n\n'
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"tool_search_call","execution":"server","call_id":null,"status":"in_progress","arguments":{}}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"tool_search_call","execution":"server","call_id":null,"status":"completed","arguments":{"pattern":"w"}}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":1,"item":{"type":"tool_search_output","execution":"server","call_id":null,"status":"completed","tools":[{"type":"function","name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]}}\n\n'
        b'data: {"type":"response.completed","response":{"id":"r1","model":"m1","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    out = b"".join([event async for event in or_stream.render(or_stream.parse(chunks()))])
    added: list[dict[str, Any]] = []
    for block in out.split(b"\n\n"):
        for line in block.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if data.get("type") == "response.output_item.added":
                added.append(data["item"])

    output_item = next(i for i in added if i["type"] == "tool_search_output")
    assert output_item["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]


async def test_stream_unknown_server_tool_name_refuses_parse() -> None:
    sse = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_abc123","name":"web_search_tool","input":{}}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    with pytest.raises(ValueError, match="only tool search server tools"):
        _ = [event async for event in ac_stream.parse(chunks())]


def test_non_search_server_tool_use_refuses_parse() -> None:
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_abc123",
                        "name": "web_search_tool",
                        "input": {"query": "weather"},
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="only tool search server tools"):
        ac.parse_request(payload)


def test_catalog_miss_tool_reference_keeps_defer_loading() -> None:
    reference = {
        "type": "tool_reference",
        "tool_name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": _WEATHER_SCHEMA,
        "defer_loading": True,
        "strict": True,
        "vendor_extra": "keep-me",
    }
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_search_tool_result",
                        "tool_use_id": "srvtoolu_abc123",
                        "content": {
                            "type": "tool_search_tool_search_result",
                            "tool_references": [reference],
                        },
                    }
                ],
            }
        ],
    }
    ir = ac.parse_request(payload)
    result_part = ir.messages[0].parts[0]
    assert isinstance(result_part, ToolSearchResultPart)
    resolved = result_part.tools[0]
    assert resolved.deferred is True
    assert resolved.strict is True
    assert resolved._raw == reference

    rendered = orc.render_request(ir)
    output_item = rendered["input"][0]
    assert output_item["type"] == "tool_search_output"
    tool = output_item["tools"][0]
    assert tool.get("defer_loading") is True
    assert tool.get("strict") is True


async def test_stream_lone_tool_search_output_refuses_parse() -> None:
    sse = (
        b'data: {"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress","output":[]}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"tool_search_output","execution":"server","call_id":null,"status":"completed","tools":[]}}\n\n'
        b'data: {"type":"response.completed","response":{"id":"r1","model":"m1","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    with pytest.raises(ValueError, match="without a preceding tool_search_call"):
        _ = [event async for event in or_stream.parse(chunks())]


async def test_stream_bm25_omits_search_variant_on_wire() -> None:
    sse = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_abc123","name":"tool_search_tool_bm25","input":{"query":"w"}}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    out = b"".join([event async for event in or_stream.render(ac_stream.parse(chunks()))])
    assert b'"search_variant"' not in out


async def test_stream_search_blocks_survive_proxy_to_proxy_loop() -> None:
    anthropic_sse = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_abc123","name":"tool_search_tool_regex","input":{"pattern":"w"}}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_abc123","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"get_weather"}]}}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    async def chunks(data: bytes) -> AsyncIterator[bytes]:
        yield data

    responses_wire = b"".join(
        [event async for event in or_stream.render(ac_stream.parse(chunks(anthropic_sse)))]
    )
    final = b"".join(
        [event async for event in ac_stream.render(or_stream.parse(chunks(responses_wire)))]
    )

    starts: list[dict[str, Any]] = []
    for block in final.split(b"\n\n"):
        for line in block.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            data = json.loads(line[6:])
            if data.get("type") == "content_block_start":
                starts.append(data["content_block"])

    assert [s["type"] for s in starts] == ["server_tool_use", "tool_search_tool_result"]
    assert starts[0]["input"] == {"pattern": "w"}
    assert starts[1]["tool_use_id"] == starts[0]["id"]
    assert starts[1]["content"]["tool_references"] == [
        {"type": "tool_reference", "tool_name": "get_weather"}
    ]


async def test_stream_search_done_without_item_uses_added_payload() -> None:
    sse = (
        b'data: {"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress","output":[]}}\n\n'
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"tool_search_call","execution":"server","call_id":null,"status":"in_progress","arguments":{"pattern":"w"}}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"tool_search_output","execution":"server","call_id":null,"status":"in_progress","tools":[{"type":"function","name":"get_weather"}]}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":1}\n\n'
        b'data: {"type":"response.completed","response":{"id":"r1","model":"m1","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    out = b"".join([event async for event in ac_stream.render(or_stream.parse(chunks()))])

    starts: list[dict[str, Any]] = []
    for block in out.split(b"\n\n"):
        for line in block.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            data = json.loads(line[6:])
            if data.get("type") == "content_block_start":
                starts.append(data["content_block"])

    assert [s["type"] for s in starts] == ["server_tool_use", "tool_search_tool_result"]
    assert starts[0]["input"] == {"pattern": "w"}
    assert starts[1]["content"]["tool_references"] == [
        {"type": "tool_reference", "tool_name": "get_weather"}
    ]


def test_web_search_tool_result_refuses_parse() -> None:
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_abc123",
                        "content": [{"type": "text", "text": "result"}],
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="cannot be translated"):
        ac.parse_request(payload)


async def test_stream_web_search_tool_result_refuses_parse() -> None:
    sse = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_abc123","content":[]}}\n\n'
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    with pytest.raises(ValueError, match="cannot be translated"):
        _ = [event async for event in ac_stream.parse(chunks())]


async def test_stream_duplicate_done_does_not_duplicate_search_block() -> None:
    sse = (
        b'data: {"type":"response.created","response":{"id":"r1","model":"m1","status":"in_progress","output":[]}}\n\n'
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"tool_search_call","execution":"server","call_id":null,"status":"in_progress","arguments":{"pattern":"w"}}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"tool_search_call","execution":"server","call_id":null,"status":"completed","arguments":{"pattern":"w"}}}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b'data: {"type":"response.completed","response":{"id":"r1","model":"m1","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield sse

    out = b"".join([event async for event in ac_stream.render(or_stream.parse(chunks()))])

    starts: list[dict[str, Any]] = []
    for block in out.split(b"\n\n"):
        for line in block.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            data = json.loads(line[6:])
            if data.get("type") == "content_block_start":
                starts.append(data["content_block"])

    assert [s["type"] for s in starts] == ["server_tool_use"]
    assert starts[0]["input"] == {"pattern": "w"}
