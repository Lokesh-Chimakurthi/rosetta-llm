"""FastAPI app factory with lifespan management."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rosetta.auth import AuthMiddleware
from rosetta.config import Config
from rosetta.observability import RequestContextMiddleware
from rosetta.routes.chat_completions import router as chat_router
from rosetta.routes.health import router as health_router
from rosetta.routes.messages import router as messages_router
from rosetta.routes.models import refresh_models
from rosetta.routes.models import router as models_router
from rosetta.routes.responses import router as responses_router
from rosetta.upstream import UpstreamClient

_REFRESH_INTERVAL = 300  # seconds


async def _models_refresh_loop(app: FastAPI, interval: float) -> None:
    while True:
        try:
            await asyncio.sleep(interval)
            await refresh_models(app.state.config, app.state.upstream, app.state.model_snapshot)
        except asyncio.CancelledError:
            break
        except Exception:
            from rosetta.observability import get_logger

            get_logger().exception("models_refresh_loop_error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.upstream = UpstreamClient(app.state.config.providers)
    await app.state.upstream.start()

    refresh_task = asyncio.create_task(
        refresh_models(app.state.config, app.state.upstream, app.state.model_snapshot)
    )
    loop_task = asyncio.create_task(_models_refresh_loop(app, _REFRESH_INTERVAL))

    try:
        yield
    finally:
        loop_task.cancel()
        refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task
            await refresh_task
        await app.state.upstream.close()


def create_app(config: Config) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    app.state.provider_status = {}
    app.state.model_snapshot = {}
    app.state.model_refresh_lock = asyncio.Lock()

    app.add_middleware(AuthMiddleware)
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health_router)
    app.include_router(messages_router)
    app.include_router(chat_router)
    app.include_router(responses_router)
    app.include_router(models_router)

    return app
