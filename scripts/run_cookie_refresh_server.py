#!/usr/bin/env python3
"""Run the miner-operated cookie refresh service."""

import os

import uvicorn
from dotenv import load_dotenv

from chronoseek.constants import (
    DEFAULT_COOKIE_REFRESH_SERVER_HOST,
    DEFAULT_COOKIE_REFRESH_SERVER_PORT,
    DEFAULT_LOG_LEVEL,
)


if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(
        "chronoseek.miner.cookie_refresh_server:app",
        host=DEFAULT_COOKIE_REFRESH_SERVER_HOST,
        port=DEFAULT_COOKIE_REFRESH_SERVER_PORT,
        log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).lower(),
        access_log=True,
    )
