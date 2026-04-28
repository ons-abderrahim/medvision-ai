"""Shared logging configuration for MedVision AI."""

from __future__ import annotations

import logging
import sys


def setup_logger(level: int = logging.INFO) -> None:
    """Configure root logger with a clean, timestamped format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
