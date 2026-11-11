"""Centralized logging infrastructure for the autex reduction pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER: logging.Logger | None = None


def setup_logger(
    name: str = "autex",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and initialize the centralized pipeline logger.

    Parameters
    ----------
    name : str, optional
        Name of the top-level logger instance, by default "autex".
    log_file : str | Path | None, optional
        Filesystem path to write runtime logs. If None, logs only to console.
    level : int, optional
        Logging severity threshold, by default logging.INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance with attached streaming and file handlers.
    """
    global _LOGGER

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        _LOGGER = logger
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-7s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logger.propagate = False
    _LOGGER = logger
    return logger


def get_logger(child_name: str | None = None) -> logging.Logger:
    """Retrieve the active logger or create a child logger."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = setup_logger()

    if child_name:
        return _LOGGER.getChild(child_name)
    return _LOGGER


def log_section(title: str, char: str = "=", length: int = 70) -> None:
    """Write an eye-catching visual section divider into the active log."""
    logger = get_logger()
    separator = char * length
    centered_title = f" {title.strip()} ".center(length, char)
    logger.info(separator)
    logger.info(centered_title)
    logger.info(separator)
    