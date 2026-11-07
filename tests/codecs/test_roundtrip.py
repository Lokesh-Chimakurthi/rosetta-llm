"""Round-trip smoke tests for codec translation."""

from __future__ import annotations

import json

import pytest

from rosetta.codecs import anthropic as ac
from rosetta.codecs import openai_chat as oc
from rosetta.codecs import openai_responses as orc
from rosetta.ir.request import ReasoningPart


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
    assert isinstance(rp, ReasoningPart)
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


def test_untranslated_anthropic_fields_never_reach_another_format() -> None:
    """Fields only Anthropic defines (auto mode review, context editing) stay Anthropic-only."""
    extras = {
        "safeguards": [{"type": "dangerous_tool_use", "classifier_context": {"v": 1}}],
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        "service_tier": "standard_only",
    }
    ir = ac.parse_request(
        {"model": "x", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], **extras}
    )
    for rendered in (oc.render_request(ir), orc.render_request(ir)):
        assert not extras.keys() & rendered.keys()
    assert ac.render_request(ir).items() >= extras.items()


def test_shared_openai_fields_cross_between_chat_and_responses_only() -> None:
    """Chat and Responses define these fields identically; Anthropic defines none of them."""
    ir = oc.parse_request(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "parallel_tool_calls": False,
            "store": False,
            "n": 2,
            "response_format": {"type": "json_object"},
        }
    )
    responses = orc.render_request(ir)
    assert (responses["parallel_tool_calls"], responses["store"]) == (False, False)
    assert "n" not in responses
    assert "response_format" not in responses
    assert not {"parallel_tool_calls", "store", "n", "response_format"} & set(
        ac.render_request(ir)
    )


def test_responses_only_fields_never_reach_chat() -> None:
    ir = orc.parse_request(
        {
            "model": "x",
            "input": "hi",
            "text": {"verbosity": "low"},
            "truncation": "auto",
            "safety_identifier": "user-1",
        }
    )
    chat = oc.render_request(ir)
    assert chat["safety_identifier"] == "user-1"
    assert "text" not in chat
    assert "truncation" not in chat


def test_shared_responses_fields_reach_chat_and_not_anthropic() -> None:
    shared = {
        "parallel_tool_calls": False,
        "store": False,
        "service_tier": "flex",
        "prompt_cache_key": "k",
    }
    ir = orc.parse_request({"model": "x", "input": "hi", **shared})
    assert oc.render_request(ir).items() >= shared.items()
    assert not shared.keys() & ac.render_request(ir).keys()


@pytest.mark.parametrize("key", ["previous_response_id", "conversation"])
def test_responses_server_state_cannot_be_translated(key: str) -> None:
    """Dropping it would answer without the conversation; refuse with the remedy instead."""
    with pytest.raises(ValueError, match=f"'{key}'.*full conversation in 'input'"):
        orc.parse_request({"model": "x", "input": "hi", key: "resp_123"})


def test_responses_null_server_state_is_ignored() -> None:
    ir = orc.parse_request({"model": "x", "input": "hi", "previous_response_id": None})
    assert [m.role for m in ir.messages] == ["user"]


def test_responses_string_input_is_one_user_message() -> None:
    chat = oc.render_request(orc.parse_request({"model": "x", "input": "hello"}))
    assert chat["messages"] == [{"role": "user", "content": "hello"}]


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


def test_stream_options_include_usage_injected_when_streaming() -> None:
    """A streaming Chat request must ask the upstream for the final usage chunk."""
    chat = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    ir = oc.parse_request(chat)
    rendered = oc.render_request(ir)
    assert rendered["stream_options"] == {"include_usage": True}


def test_stream_options_client_value_preserved_not_overwritten() -> None:
    """A client-sent stream_options (via raw_extras) wins over the injected default."""
    chat = {
        "model": "m",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "stream_options": {"include_usage": False, "custom": 1},
    }
    ir = oc.parse_request(chat)
    rendered = oc.render_request(ir)
    assert rendered["stream_options"] == {"include_usage": False, "custom": 1}


def test_stream_options_absent_when_not_streaming() -> None:
    """Non-streaming requests must not carry stream_options."""
    chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    ir = oc.parse_request(chat)
    rendered = oc.render_request(ir)
    assert "stream_options" not in rendered
