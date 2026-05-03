"""Canonical response IR — model-agnostic representation of an LLM response."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from rosetta.ir.request import Message


class Usage(BaseModel):
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
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)
