#!/usr/bin/env python3
"""Run the miner-operated cookie refresh service."""

import os

import uvicorn
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(
        "chronoseek.miner.cookie_refresh_server:app",
        host=os.getenv("COOKIE_REFRESH_HOST", "127.0.0.1"),
        port=int(os.getenv("COOKIE_REFRESH_PORT", "8091")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
    )
