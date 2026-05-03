"""Pydantic-validated configuration loader.

Reads a strict-JSON config.json and produces a validated Config object.
Supports api_key_env resolution, model route table construction, and
auto-discovery markers for providers with models=["*"].
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Format = Literal["openai_chat", "openai_responses", "anthropic"]


class Capabilities(BaseModel):
    tools: bool = True
    vision: bool = True
    thinking: bool = True
    structured_output: bool = True


class ModelConfig(BaseModel):
    id: str
    upstream_name: str | None = None
    supports: Capabilities = Field(default_factory=Capabilities)
    thinking_budget_default: int = 12288

    @property
    def effective_upstream_name(self) -> str:
        return self.upstream_name or self.id


class TimeoutConfig(BaseModel):
    connect: float = 30.0
    read: float = 600.0


class ProviderConfig(BaseModel):
    format: Format
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    models: list[ModelConfig | Literal["*"]] = Field(default_factory=list)
    models_ttl_seconds: int = 300

    @model_validator(mode="after")
    def resolve_api_key(self) -> ProviderConfig:
        if self.api_key is None and self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ValueError(f"Environment variable {self.api_key_env} is not set")
            self.api_key = key
        if self.api_key is None:
            raise ValueError("Provider requires api_key or api_key_env")
        return self

    @property
    def is_auto_discover(self) -> bool:
        return len(self.models) == 1 and self.models[0] == "*"

    @property
    def static_models(self) -> list[ModelConfig]:
        return [m for m in self.models if isinstance(m, ModelConfig)]


class ProxyConfig(BaseModel):
    api_keys: list[str] = Field(default_factory=list)


class Config(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_providers(self) -> Config:
        for key, _prov in self.providers.items():
            if "/" in key:
                raise ValueError(f"Provider key '{key}' must not contain '/'")
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate config from a JSON file.

    Strips // comments (single-line only) to support config.example.jsonc style.
    """
    raw = Path(path).read_text()
    lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    clean = "\n".join(lines)
    data = json.loads(clean)
    return Config.model_validate(data)
