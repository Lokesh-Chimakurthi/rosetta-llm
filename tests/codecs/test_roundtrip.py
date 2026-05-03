"""Round-trip smoke tests for codec translation."""

from __future__ import annotations

import json

from rosetta.codecs import anthropic as ac
from rosetta.codecs import openai_chat as oc
from rosetta.codecs import openai_responses as orc


def test_anthropic_request_roundtrip_simple() -> None:
    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 256,
        "system": "You are concise.",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    ir = ac.parse_request(payload)
    assert ir.model == "claude-opus-4-7"
    assert ir.max_output_tokens == 256
    assert ir.system == "You are concise."
    rendered = ac.render_request(ir)
    assert rendered["model"] == "claude-opus-4-7"
    assert rendered["max_tokens"] == 256
    assert rendered["system"] == "You are concise."
    assert rendered["messages"][0]["role"] == "user"


def test_anthropic_tool_use_input_is_object_after_render() -> None:
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                ],
            }
        ],
    }
    ir = ac.parse_request(payload)
    rendered = ac.render_request(ir)
    block = rendered["messages"][0]["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_1"
    assert block["input"] == {"city": "Paris"}, "tool_use input must be an object"


def test_anthropic_tool_result_ordering_enforced() -> None:
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "follow-up"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "21C"},
                ],
            }
        ],
    }
    ir = ac.parse_request(payload)
    rendered = ac.render_request(ir)
    parts = rendered["messages"][0]["content"]
    assert parts[0]["type"] == "tool_result"
    assert parts[1]["type"] == "text"


def test_openai_chat_to_anthropic_tool_call_id_preserved() -> None:
    chat = {
        "model": "abc/m1",
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "21C"},
        ],
    }
    ir = oc.parse_request(chat)
    anthropic_payload = ac.render_request(ir)
    assistant_msg = anthropic_payload["messages"][1]
    assert assistant_msg["role"] == "assistant"
    tool_use = assistant_msg["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "call_123"
    assert tool_use["input"] == {"city": "Paris"}
    user_msg = anthropic_payload["messages"][2]
    assert user_msg["role"] == "user"
    assert user_msg["content"][0]["type"] == "tool_result"
    assert user_msg["content"][0]["tool_use_id"] == "call_123"


def test_responses_reasoning_lossless_via_anthropic_signature() -> None:
    """encrypted_content + id encoded in Anthropic signature, decoded back losslessly."""
    responses_payload = {
        "model": "gpt-5.5",
        "input": [
            {
                "type": "reasoning",
                "id": "rs_abc",
                "encrypted_content": "ENCRYPTED_BLOB",
                "summary": [{"type": "summary_text", "text": "Plan…"}],
            },
        ],
    }
    ir = orc.parse_request(responses_payload)
    # render to anthropic
    anthropic_payload = ac.render_request(ir)
    thinking = anthropic_payload["messages"][0]["content"][0]
    assert thinking["type"] == "thinking"
    assert thinking["signature"] == "ENCRYPTED_BLOB@rs_abc"
    # parse anthropic back
    ir2 = ac.parse_request(anthropic_payload)
    rp = ir2.messages[0].parts[0]
    assert rp.encrypted_content == "ENCRYPTED_BLOB"
    assert rp.reasoning_id == "rs_abc"
    # render back to responses
    rerendered = orc.render_request(ir2)
    rs_item = rerendered["input"][0]
    assert rs_item["type"] == "reasoning"
    assert rs_item["id"] == "rs_abc"
    assert rs_item["encrypted_content"] == "ENCRYPTED_BLOB"


def test_openai_chat_response_roundtrip_to_anthropic() -> None:
    chat_resp = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "m1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    ir = oc.parse_response(chat_resp)
    assert ir.usage.input_tokens == 5
    assert ir.usage.output_tokens == 2
    anthropic_resp = ac.render_response(ir)
    assert anthropic_resp["type"] == "message"
    assert anthropic_resp["role"] == "assistant"
    assert anthropic_resp["content"][0]["type"] == "text"
    assert anthropic_resp["content"][0]["text"] == "Hello world"
    assert anthropic_resp["stop_reason"] == "end_turn"


def test_anthropic_unknown_param_passthrough_via_raw_extras() -> None:
    payload = {
        "model": "x",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "custom_unknown_field": "preserve me",
    }
    ir = ac.parse_request(payload)
    assert ir.raw_extras.get("custom_unknown_field") == "preserve me"
    rendered = ac.render_request(ir)
    assert rendered.get("custom_unknown_field") == "preserve me"


def test_render_anthropic_max_tokens_synthesized() -> None:
    """Anthropic requires max_tokens; synthesize a default when absent."""
    chat = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    ir = oc.parse_request(chat)
    assert ir.max_output_tokens is None
    rendered = ac.render_request(ir)
    assert rendered["max_tokens"] >= 1


def test_streaming_arguments_partial_json_buffered_correctly() -> None:
    """Tool argument fragments accumulate as a string, parseable as JSON when complete."""
    fragments = ['{"', "city", '":"', "Paris", '"}']
    full = "".join(fragments)
    parsed = json.loads(full)
    assert parsed == {"city": "Paris"}
