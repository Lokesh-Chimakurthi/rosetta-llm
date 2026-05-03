"""ASGI entrypoint — creates the app at module level so uvicorn can import it.

uvicorn rosetta.asgi:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from pathlib import Path

from rosetta.app import create_app
from rosetta.config import load_config
from rosetta.observability import setup_logging

config_path = os.environ.get("ROSETTA_CONFIG", str(Path.home() / ".rosetta-llm" / "config.json"))
config = load_config(config_path)
setup_logging(config.log_level)
app = create_app(config)
