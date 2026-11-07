"""Which untranslated request fields a renderer may forward.

Each codec keeps the top-level request fields it does not translate in
`raw_extras`. Their meaning is defined only by the format they arrived in, so
another format receives just the fields both define identically. Anything else
is rejected by strict upstreams (OpenAI and Anthropic refuse unknown fields)
or silently misread by lenient ones.
"""

from __future__ import annotations

from typing import Any

from rosetta.config import Format
from rosetta.ir.request import CanonicalRequest

# Same name and meaning on both create endpoints, per
# https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
# and https://developers.openai.com/api/reference/cli/resources/responses/methods/create.
# Anthropic shares none with either: even `service_tier` takes other values.
_SHARED_FIELDS: dict[frozenset[Format], frozenset[str]] = {
    frozenset({"openai_chat", "openai_responses"}): frozenset(
        {
            "parallel_tool_calls",
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
            "safety_identifier",
            "service_tier",
            "store",
        }
    ),
}


def extras_for(ir: CanonicalRequest, target: Format) -> dict[str, Any]:
    """The `raw_extras` a `target`-format request may carry.

    An IR with no recorded source format forwards nothing.
    """
    source = ir.raw_extras_format
    if source is None:
        return {}
    if source == target:
        return dict(ir.raw_extras)
    shared = _SHARED_FIELDS.get(frozenset({source, target}), frozenset())
    return {k: v for k, v in ir.raw_extras.items() if k in shared}
