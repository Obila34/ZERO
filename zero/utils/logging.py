"""Single place to configure logging so every module logs consistently."""
from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.environ.get("ZERO_LOG", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s",
        datefmt="%H:%M:%S",
    )
