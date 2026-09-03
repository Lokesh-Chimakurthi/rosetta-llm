"""Canonical request IR — model-agnostic representation of an LLM request."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class _IRBase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    _raw: dict[str, Any] = PrivateAttr(default_factory=dict)


class TextPart(_IRBase):
    type: Literal["text"] = "text"
    text: str


class ImagePart(_IRBase):
    type: Literal["image"] = "image"
    source_type: Literal["url", "base64"] = "base64"
    media_type: str = "image/png"
    data: str = ""


class DocumentPart(_IRBase):
    type: Literal["document"] = "document"
    media_type: str = "application/pdf"
    data: str = ""


class ToolCallPart(_IRBase):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    # Always a JSON-serialized string; codecs convert to/from object on the wire.
    arguments_json_text: str = "{}"


class ToolResultPart(_IRBase):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content_parts: list[ContentPart] = Field(default_factory=list)
    is_error: bool = False


class ServerToolUsePart(_IRBase):
    type: Literal["server_tool_use"] = "server_tool_use"
    call_id: str = ""
    name: str = ""
    input: dict[str, Any] = Field(default_factory=dict)


class ToolSearchResultPart(_IRBase):
    type: Literal["tool_search_result"] = "tool_search_result"
    call_id: str = ""
    tools: list[Tool] = Field(default_factory=list)


class ReasoningPart(_IRBase):
    type: Literal["reasoning"] = "reasoning"
    visibility: Literal["visible", "redacted"] = "visible"
    text: str = ""
    signature: str = ""
    encrypted_content: str = ""
    # Provider-side reasoning item id (Responses `id`, Anthropic-encoded as <enc>@<id>).
    reasoning_id: str = ""
    summary: str = ""


class RefusalPart(_IRBase):
    type: Literal["refusal"] = "refusal"
    text: str


ContentPart = Annotated[
    TextPart
    | ImagePart
    | DocumentPart
    | ToolCallPart
    | ToolResultPart
    | ServerToolUsePart
    | ToolSearchResultPart
    | ReasoningPart
    | RefusalPart,
    Field(discriminator="type"),
]


class Message(_IRBase):
    # `tool` role only appears in OpenAI Chat-style flow; Anthropic uses user-message tool_result blocks.
    role: Literal["system", "user", "assistant", "tool"]
    parts: list[ContentPart] = Field(default_factory=list)


class Tool(_IRBase):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    kind: Literal["function", "hosted", "search"] = "function"
    strict: bool = False
    # `defer_loading` on the wire: the tool definition stays out of the context
    # window until the tool search loads it. There is no `deferred` flag on
    # tool-call blocks; deferredness lives on the definition only.
    deferred: bool = False


class ReasoningConfig(BaseModel):
    effort: Literal["low", "medium", "high", "xhigh"] | None = None
    budget_tokens: int | None = None
    thinking_type: Literal["enabled", "adaptive", "disabled"] | None = None
    summary: Literal["auto", "detailed", "concise"] | None = None
    include_encrypted: bool = False


class SamplingConfig(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None


class CanonicalRequest(BaseModel):
    model: str = ""
    messages: list[Message] = Field(default_factory=list)
    system: str | list[TextPart] | None = None
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: Any = "auto"
    max_output_tokens: int | None = None
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    stop_sequences: list[str] = Field(default_factory=list)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_extras: dict[str, Any] = Field(default_factory=dict)


# Resolve forward references for recursive ToolResultPart.content_parts.
ToolResultPart.model_rebuild()
ToolSearchResultPart.model_rebuild()
Message.model_rebuild()
CanonicalRequest.model_rebuild()
