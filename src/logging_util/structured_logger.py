"""Structured logging utility.

Task: FND-6 (P01) – Logging Setup
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict

APP_NAME = "ielts-speaking-eval"

class Logger:
    def _emit(self, level: str, msg: str, **fields: Any) -> None:
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "msg": msg,
            "app": APP_NAME,
        }
        record.update(fields)
        sys.stdout.write(json.dumps(record) + "\n")

    def info(self, msg: str, **fields: Any) -> None:
        self._emit("INFO", msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit("ERROR", msg, **fields)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit("DEBUG", msg, **fields)

logger = Logger()
