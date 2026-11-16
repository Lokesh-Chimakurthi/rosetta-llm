"""Canonical response IR — model-agnostic representation of an LLM response."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from rosetta.ir.request import Message


class Usage(BaseModel):
    """Token counts with one meaning across formats.

    `input_tokens` counts every prompt token, cache reads and writes included,
    as OpenAI's `prompt_tokens` and `input_tokens` do. Anthropic reports only
    the uncached remainder; its codecs convert at the boundary.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0


class StopInfo(BaseModel):
    normalized: str = ""
    provider_raw: str = ""
    stop_sequence: str | None = None


class CanonicalResponse(BaseModel):
    id: str = ""
    model: str = ""
    output_messages: list[Message] = Field(default_factory=list)
    stop: StopInfo = Field(default_factory=StopInfo)
    usage: Usage = Field(default_factory=Usage)
    _raw: dict[str, Any] | None = PrivateAttr(default=None)
