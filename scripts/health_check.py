"""Health check script.

Task: FND-8 (P01) – Health Script
This script validates environment configuration and prints a simple readiness JSON.
Azure API call is mocked for now (no network) – extend later to perform real request.
"""
from __future__ import annotations
import json
import sys
from config.app_config import load_config
from logging_util.structured_logger import logger


def main() -> int:
    cfg = load_config()
    problems = []
    if not cfg.endpoint:
        problems.append("Missing AZURE_OPENAI_ENDPOINT")
    if not cfg.api_key:
        problems.append("Missing AZURE_OPENAI_KEY (or AAD token path)")
    status = "ok" if not problems else "degraded"
    logger.info("health_check", status=status, model=cfg.model, api_version=cfg.api_version)
    print(json.dumps({
        "status": status,
        "model": cfg.model,
        "api_version": cfg.api_version,
        "problems": problems,
    }, indent=2))
    return 0 if status == "ok" else 1

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
