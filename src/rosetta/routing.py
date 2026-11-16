"""Model id → provider resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from rosetta.config import Config, ModelConfig, ProviderConfig

CLAUDE_CODE_GW_PREFIX = "claude-code/"


@dataclass(frozen=True, slots=True)
class Route:
    """Where a model id is sent."""

    provider: ProviderConfig
    provider_key: str
    upstream_name: str
    # The provider's `models` entry for this id; None when it lists no such model.
    model: ModelConfig | None

    @property
    def forwards_cache_control(self) -> bool:
        """Whether translated requests keep their prompt-cache breakpoints.

        Native to Anthropic; on `openai_chat` only for a listed model that sets
        `forward_cache_control`; never on `openai_responses`, which has no field.
        """
        match self.provider.format:
            case "anthropic":
                return True
            case "openai_chat":
                return self.model is not None and self.model.forward_cache_control
            case "openai_responses":
                return False
            case unreachable:
                assert_never(unreachable)


def resolve_model(model_id: str, config: Config) -> Route:
    """Resolve `<provider>/<model>` to its provider, upstream model name and model entry.

    A `claude-code/` prefix is stripped first; a bare name routes to
    `default_provider` when one is set. `upstream_name` remaps the model name.

    Raises:
        ValueError: the id has no provider, names an unknown one, or is empty.
            The message says how to fix it.
    """
    if model_id.startswith(CLAUDE_CODE_GW_PREFIX):
        inner = model_id[len(CLAUDE_CODE_GW_PREFIX) :]
        if "/" in inner:
            return resolve_model(inner, config)
        raise ValueError(f"Model id '{model_id}' has gateway prefix but no provider/model")
    if "/" not in model_id:
        # Some clients (e.g. OpenAI Codex CLI) strip the provider prefix, so the
        # operator may name a default provider. An empty id is never routed.
        if model_id and config.default_provider is not None:
            return resolve_model(f"{config.default_provider}/{model_id}", config)
        raise ValueError(
            f"Model id '{model_id}' must be in format '<provider>/<model>' "
            "(or set 'default_provider' in config to route bare model names)"
        )
    provider_key, model_name = model_id.split("/", 1)
    if not model_name:
        raise ValueError(f"Model id '{model_id}' must be in format '<provider>/<model>'")
    provider = config.providers.get(provider_key)
    if provider is None:
        raise ValueError(f"Unknown provider '{provider_key}'")
    model = next((m for m in provider.static_models if m.id == model_name), None)
    upstream_name = model.effective_upstream_name if model else model_name
    return Route(provider, provider_key, upstream_name, model)
