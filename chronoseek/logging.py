"""ChronoSeek application logging.

Bittensor 11 no longer provides the legacy ``bt.logging`` facade, so logging
is configured and owned by the application.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


class ChronoSeekLogger:
    """Small compatibility layer for the log methods used by ChronoSeek."""

    def __init__(self, name: str = "chronoseek") -> None:
        self._logger = logging.getLogger(name)

    def debug(self, message: Any, *args, **kwargs) -> None:
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: Any, *args, **kwargs) -> None:
        self._logger.info(message, *args, **kwargs)

    def success(self, message: Any, *args, **kwargs) -> None:
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: Any, *args, **kwargs) -> None:
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: Any, *args, **kwargs) -> None:
        self._logger.error(message, *args, **kwargs)

    def exception(self, message: Any, *args, **kwargs) -> None:
        self._logger.exception(message, *args, **kwargs)


logger = ChronoSeekLogger()


def configure_logging(
    level: str = "INFO",
    *,
    logging_dir: str | None = None,
    filename: str = "chronoseek.log",
) -> None:
    """Configure console logging and an optional ChronoSeek log file."""

    normalized_level = str(level or "INFO").upper()
    numeric_level = logging.DEBUG if normalized_level in {"DEBUG", "TRACE"} else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    root_logger.setLevel(numeric_level)

    logging.getLogger("chronoseek").setLevel(numeric_level)
    logging.getLogger("bittensor").setLevel(numeric_level)

    if not logging_dir:
        return

    directory = Path(logging_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / filename
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
