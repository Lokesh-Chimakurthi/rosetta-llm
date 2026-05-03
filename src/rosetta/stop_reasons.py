"""Shared stop-reason mappings used by both codec and stream_codec modules."""

# Anthropic stop_reason ↔ normalized reason (identical in both directions).
ANTHROPIC_STOP = {
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
    "pause_turn": "pause_turn",
}

# OpenAI Chat finish_reason → Anthropic-normalized.
OPENAI_CHAT_STOP_IN = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}

# Anthropic-normalized → OpenAI Chat finish_reason.
OPENAI_CHAT_STOP_OUT = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "refusal": "content_filter",
}
