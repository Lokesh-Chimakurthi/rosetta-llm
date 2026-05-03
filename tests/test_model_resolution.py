"""Tests for `_resolve_model` shorthand + alias + fuzzy resolution."""

from __future__ import annotations

import os

import pytest

from rosetta.config import Config
from rosetta.pipeline import _resolve_model


def _cfg(providers: dict[str, dict]) -> Config:
    os.environ["UP_KEY"] = "sk-up"
    base_provider = {
        "format": "openai_chat",
        "base_url": "https://upstream.test/v1",
        "api_key_env": "UP_KEY",
    }
    return Config.model_validate(
        {"providers": {k: {**base_provider, **v} for k, v in providers.items()}}
    )


def test_strict_provider_model_still_wins() -> None:
    cfg = _cfg(
        {
            "anth": {"models": [{"id": "claude-haiku-4-5"}]},
            "openrouter": {
                "models": [{"id": "anthropic/claude-haiku-4-5", "aliases": ["claude-haiku-4-5"]}]
            },
        }
    )
    _, key, upstream = _resolve_model("anth/claude-haiku-4-5", cfg)
    assert key == "anth"
    assert upstream == "claude-haiku-4-5"


def test_strict_unknown_model_under_known_provider_passes_through() -> None:
    cfg = _cfg({"anth": {"models": [{"id": "claude-haiku-4-5"}]}})
    _, key, upstream = _resolve_model("anth/some-other-model", cfg)
    assert key == "anth"
    assert upstream == "some-other-model"


def test_shorthand_exact_id_resolves_unique_provider() -> None:
    cfg = _cfg(
        {
            "anth": {"models": [{"id": "claude-haiku-4-5"}]},
            "openai": {"models": [{"id": "gpt-5"}]},
        }
    )
    _, key, upstream = _resolve_model("claude-haiku-4-5", cfg)
    assert key == "anth"
    assert upstream == "claude-haiku-4-5"


def test_shorthand_alias_list_resolves() -> None:
    cfg = _cfg(
        {
            "openrouter": {
                "models": [
                    {
                        "id": "anthropic/claude-opus-4.1",
                        "upstream_name": "anthropic/claude-opus-4.1",
                        "aliases": ["claude-opus-4-7", "opus"],
                    }
                ]
            }
        }
    )
    _, key, upstream = _resolve_model("claude-opus-4-7", cfg)
    assert key == "openrouter"
    assert upstream == "anthropic/claude-opus-4.1"


def test_shorthand_regex_resolves() -> None:
    cfg = _cfg(
        {
            "openrouter": {
                "models": [
                    {
                        "id": "anthropic/claude-sonnet-4.5",
                        "upstream_name": "anthropic/claude-sonnet-4.5",
                        "alias_pattern": r"^claude-sonnet-.*",
                    }
                ]
            }
        }
    )
    _, key, upstream = _resolve_model("claude-sonnet-4-5", cfg)
    assert key == "openrouter"
    assert upstream == "anthropic/claude-sonnet-4.5"


def test_shorthand_substring_resolves_when_unique() -> None:
    cfg = _cfg(
        {
            "anth": {
                "models": [
                    {"id": "claude-haiku-4-5"},
                    {"id": "claude-opus-4-7"},
                ]
            }
        }
    )
    _, key, upstream = _resolve_model("haiku", cfg)
    assert key == "anth"
    assert upstream == "claude-haiku-4-5"


def test_shorthand_partial_substring_resolves() -> None:
    cfg = _cfg(
        {
            "anth": {
                "models": [
                    {"id": "claude-haiku-4-5"},
                    {"id": "claude-opus-4-7"},
                ]
            }
        }
    )
    _, _, upstream = _resolve_model("claude-haiku", cfg)
    assert upstream == "claude-haiku-4-5"


def test_exact_match_wins_over_substring() -> None:
    """If a model's id exactly matches and another's id contains it as substring,
    the exact match resolves cleanly without ambiguity."""
    cfg = _cfg(
        {
            "anth": {
                "models": [
                    {"id": "haiku"},
                    {"id": "claude-haiku-4-5"},
                ]
            }
        }
    )
    _, _, upstream = _resolve_model("haiku", cfg)
    assert upstream == "haiku"


def test_ambiguous_shorthand_lists_candidates() -> None:
    cfg = _cfg(
        {
            "anth": {"models": [{"id": "claude-haiku-4-5"}]},
            "openrouter": {
                "models": [{"id": "anthropic/claude-haiku-4-5", "aliases": ["claude-haiku-4-5"]}]
            },
        }
    )
    with pytest.raises(ValueError) as exc:
        _resolve_model("claude-haiku-4-5", cfg)
    msg = str(exc.value)
    assert "anth/claude-haiku-4-5" in msg
    assert "openrouter/anthropic/claude-haiku-4-5" in msg
    assert "exactly" in msg


def test_ambiguous_substring_labels_tier() -> None:
    cfg = _cfg(
        {
            "anth": {
                "models": [
                    {"id": "claude-haiku-4-5"},
                    {"id": "claude-haiku-4-7"},
                ]
            }
        }
    )
    with pytest.raises(ValueError) as exc:
        _resolve_model("haiku", cfg)
    msg = str(exc.value)
    assert "as a substring" in msg
    assert "anth/claude-haiku-4-5" in msg
    assert "anth/claude-haiku-4-7" in msg


def test_no_match_raises() -> None:
    cfg = _cfg({"anth": {"models": [{"id": "claude-haiku-4-5"}]}})
    with pytest.raises(ValueError) as exc:
        _resolve_model("gpt-5", cfg)
    assert "did not match" in str(exc.value)


def test_claude_code_prefix_with_bare_name_recurses_to_shorthand() -> None:
    cfg = _cfg({"anth": {"models": [{"id": "claude-haiku-4-5"}]}})
    _, key, upstream = _resolve_model("claude-code/claude-haiku-4-5", cfg)
    assert key == "anth"
    assert upstream == "claude-haiku-4-5"


def test_claude_code_prefix_with_provider_pair_still_works() -> None:
    cfg = _cfg({"anth": {"models": [{"id": "claude-haiku-4-5"}]}})
    _, key, upstream = _resolve_model("claude-code/anth/claude-haiku-4-5", cfg)
    assert key == "anth"
    assert upstream == "claude-haiku-4-5"


def test_invalid_alias_pattern_fails_config_load() -> None:
    with pytest.raises(ValueError) as exc:
        _cfg({"anth": {"models": [{"id": "x", "alias_pattern": "(unclosed"}]}})
    assert "alias_pattern" in str(exc.value)


def test_upstream_name_used_when_alias_resolves() -> None:
    cfg = _cfg(
        {
            "openrouter": {
                "models": [
                    {
                        "id": "anthropic/claude-opus-4.1",
                        "upstream_name": "anthropic/claude-opus-4.1:beta",
                        "aliases": ["claude-opus-4-7"],
                    }
                ]
            }
        }
    )
    _, _, upstream = _resolve_model("claude-opus-4-7", cfg)
    assert upstream == "anthropic/claude-opus-4.1:beta"
