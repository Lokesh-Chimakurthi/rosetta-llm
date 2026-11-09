"""Model id → provider resolution."""

from __future__ import annotations

from rosetta.config import Config, ProviderConfig

CLAUDE_CODE_GW_PREFIX = "claude-code/"


def resolve_model(model_id: str, config: Config) -> tuple[ProviderConfig, str, str]:
    """Resolve `<provider>/<model>` to (provider config, provider key, upstream model name).

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
    upstream_name = model_name
    for m in provider.static_models:
        if m.id == model_name:
            upstream_name = m.effective_upstream_name
            break
    return provider, provider_key, upstream_name
