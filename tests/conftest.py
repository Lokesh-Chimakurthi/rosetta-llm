"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import structlog


@pytest.fixture
def log_records() -> Iterator[list[dict[str, Any]]]:
    """Capture structlog records emitted during the test as a list of dicts.

    Reconfigures structlog with a capture processor for the test's duration
    and restores the previous configuration afterward.
    """
    captured: list[dict[str, Any]] = []

    def _capture(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(event_dict))
        return event_dict

    saved = structlog.get_config()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _capture,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    try:
        yield captured
    finally:
        structlog.configure(**saved)
