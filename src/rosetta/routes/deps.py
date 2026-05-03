"""FastAPI dependency injection for shared application state."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from rosetta.config import Config
from rosetta.upstream import UpstreamClient


def get_config(request: Request) -> Config:
    cfg = request.app.state.config
    if cfg is None:
        raise RuntimeError("app.state.config is not initialized")
    return cast(Config, cfg)


def get_upstream(request: Request) -> UpstreamClient:
    upstream = request.app.state.upstream
    if upstream is None:
        raise RuntimeError("app.state.upstream is not initialized")
    return cast(UpstreamClient, upstream)


ConfigDep = Annotated[Config, Depends(get_config)]
UpstreamDep = Annotated[UpstreamClient, Depends(get_upstream)]
