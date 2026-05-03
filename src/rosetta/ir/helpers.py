"""Helpers for constructing IR objects with `_raw` sidecar attached."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def with_raw[T: BaseModel](obj: T, raw: dict[str, Any]) -> T:
    """Attach a `_raw` sidecar to a Pydantic IR object via PrivateAttr."""
    object.__setattr__(obj, "_raw", raw)
    return obj
