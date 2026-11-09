"""LLM Proxy — Multi-format bidirectional translation service."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rosetta-llm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
