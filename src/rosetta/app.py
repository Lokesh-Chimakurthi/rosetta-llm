"""FastAPI app factory with lifespan management."""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rosetta import __version__
from rosetta.auth import AuthMiddleware
from rosetta.config import Config
from rosetta.observability import RequestContextMiddleware, get_logger
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
        except Exception:  # noqa: BLE001 - the loop must outlive any single failed refresh; logged with traceback
            get_logger().exception("models_refresh_loop_error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    config: Config = app.state.config
    get_logger().info(
        "startup",
        version=__version__,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        providers=[
            {"key": key, "format": prov.format, "discover": prov.is_auto_discover}
            for key, prov in config.providers.items()
        ],
    )
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
        for task in (loop_task, refresh_task):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await app.state.upstream.close()


def create_app(config: Config, *, upstream: UpstreamClient | None = None) -> FastAPI:
    """Build the ASGI app.

    Args:
        upstream: Provider client to use instead of one built from `config`;
            it is started and closed by the app's lifespan.
    """
    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    app.state.upstream = upstream if upstream is not None else UpstreamClient(config)
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
