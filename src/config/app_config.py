"""Configuration loader.

Task: FND-7 (P01) – Config Loader
Spec Ref: Sections 4.2 Configuration Parameters; Storage Strategy Updated.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "gpt-4o-mini-audio-preview"
DEFAULT_API_VERSION = "2025-01-01-preview"

@dataclass
class AppConfig:
    endpoint: str
    api_key: Optional[str]
    api_version: str
    model: str
    region: Optional[str] = None
    realtime_enabled: bool = False


def load_config(env: dict | None = None) -> AppConfig:
    env = env or os.environ
    return AppConfig(
        endpoint=env.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=env.get("AZURE_OPENAI_KEY"),
        api_version=env.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
        model=env.get("AZURE_OPENAI_MODEL", DEFAULT_MODEL),
        region=env.get("AZURE_OPENAI_REGION"),
        realtime_enabled=env.get("REALTIME_ENABLED", "false").lower() == "true",
    )
