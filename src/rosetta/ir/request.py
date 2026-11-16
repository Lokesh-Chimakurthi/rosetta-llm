"""Canonical request IR — model-agnostic representation of an LLM request."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from rosetta.config import Format


class _IRBase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    _raw: dict[str, Any] | None = PrivateAttr(default=None)


class CacheControl(BaseModel):
    """A prompt-cache breakpoint: the prefix ending at the marked item is cached.

    Anthropic's `cache_control` object, which LiteLLM- and OpenRouter-style
    Chat Completions upstreams accept too. Keys beyond `type` (`ttl`, and any
    added later) are kept verbatim.
    """

    model_config = ConfigDict(extra="allow", frozen=True)
    type: str

    @classmethod
    def from_wire(cls, value: object) -> CacheControl | None:
        """Parse a wire `cache_control` value; absent or null means no breakpoint.

        Raises:
            TypeError: the value is not a JSON object.
            pydantic.ValidationError: the object has no string `type`.
        """
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError('cache_control must be an object such as {"type": "ephemeral"}')
        return cls.model_validate(value)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump()


class _Cacheable(_IRBase):
    cache_control: CacheControl | None = None


class TextPart(_Cacheable):
    type: Literal["text"] = "text"
    text: str


class ImagePart(_Cacheable):
    type: Literal["image"] = "image"
    source_type: Literal["url", "base64"] = "base64"
    media_type: str = "image/png"
    data: str = ""


class DocumentPart(_Cacheable):
    type: Literal["document"] = "document"
    media_type: str = "application/pdf"
    data: str = ""


class ToolCallPart(_Cacheable):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    # Always a JSON-serialized string; codecs convert to/from object on the wire.
    arguments_json_text: str = "{}"


class ToolResultPart(_Cacheable):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content_parts: list[ContentPart] = Field(default_factory=list)
    is_error: bool = False


class ReasoningPart(_Cacheable):
    type: Literal["reasoning"] = "reasoning"
    visibility: Literal["visible", "redacted"] = "visible"
    text: str = ""
    signature: str = ""
    encrypted_content: str = ""
    # Provider-side reasoning item id (Responses `id`, Anthropic-encoded as <enc>@<id>).
    reasoning_id: str = ""
    summary: str = ""


class RefusalPart(_Cacheable):
    type: Literal["refusal"] = "refusal"
    text: str


ContentPart = Annotated[
    TextPart
    | ImagePart
    | DocumentPart
    | ToolCallPart
    | ToolResultPart
    | ReasoningPart
    | RefusalPart,
    Field(discriminator="type"),
]


class Message(_IRBase):
    # `tool` role only appears in OpenAI Chat-style flow; Anthropic uses user-message tool_result blocks.
    role: Literal["system", "user", "assistant", "tool"]
    parts: list[ContentPart] = Field(default_factory=list)


class Tool(_Cacheable):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    kind: Literal["function", "hosted"] = "function"
    strict: bool = False


class ReasoningConfig(BaseModel):
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
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
    # The format whose request carried raw_extras; see codecs.extras.extras_for.
    raw_extras_format: Format | None = None

    def has_cache_control(self) -> bool:
        """Whether any system block, tool or content part carries a breakpoint."""
        system = self.system if isinstance(self.system, list) else []
        parts = [p for m in self.messages for p in m.parts]
        nested = [cp for p in parts if isinstance(p, ToolResultPart) for cp in p.content_parts]
        items = [*system, *self.tools, *parts, *nested]
        return any(i.cache_control is not None for i in items)

    def without_cache_control(self) -> CanonicalRequest:
        """A copy with every breakpoint removed; `_raw` sidecars are kept."""
        system = (
            [_uncached(p) for p in self.system] if isinstance(self.system, list) else self.system
        )
        messages = [
            m.model_copy(update={"parts": [_uncached(p) for p in m.parts]}) for m in self.messages
        ]
        return self.model_copy(
            update={
                "system": system,
                "tools": [_uncached(t) for t in self.tools],
                "messages": messages,
            }
        )


def _uncached[T: BaseModel](item: T) -> T:
    update: dict[str, Any] = {}
    if isinstance(item, ToolResultPart):
        update["content_parts"] = [_uncached(cp) for cp in item.content_parts]
    if isinstance(item, _Cacheable):
        update["cache_control"] = None
    return item.model_copy(update=update) if update else item


# Resolve forward references for recursive ToolResultPart.content_parts.
ToolResultPart.model_rebuild()
Message.model_rebuild()
CanonicalRequest.model_rebuild()
