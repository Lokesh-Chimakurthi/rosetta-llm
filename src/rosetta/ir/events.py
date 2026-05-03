"""Canonical stream events — discriminated union for streaming translation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, PrivateAttr

from rosetta.ir.response import StopInfo, Usage


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    model: str = ""
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)


class PartStartEvent(BaseModel):
    type: Literal["part_start"] = "part_start"
    index: int = 0
    part_type: str = ""
    call_id: str | None = None
    name: str | None = None
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)


class PartDeltaEvent(BaseModel):
    type: Literal["part_delta"] = "part_delta"
    index: int = 0
    delta_type: Literal["text", "json", "reasoning", "signature"] = "text"
    text: str = ""
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)


class PartStopEvent(BaseModel):
    type: Literal["part_stop"] = "part_stop"
    index: int = 0
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    stop: StopInfo | None = None
    usage: Usage | None = None
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)


class PingEvent(BaseModel):
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error_type: str = ""
    message: str = ""
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)


CanonicalStreamEvent = (
    MessageStartEvent
    | PartStartEvent
    | PartDeltaEvent
    | PartStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
    | PingEvent
    | ErrorEvent
)
