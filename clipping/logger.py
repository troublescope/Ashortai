"""
clipping.logger — Centralized logging configuration.

Usage:
    from clipping.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened")
    log.debug("Debug detail: %s", value)
"""

import logging
import sys


class _EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emoji prefixes based on log level."""

    LEVEL_EMOJI = {
        logging.DEBUG:    "🐛",
        logging.INFO:     "ℹ️ ",
        logging.WARNING:  "⚠️ ",
        logging.ERROR:    "❌",
        logging.CRITICAL: "💥",
    }

    def format(self, record):
        emoji = self.LEVEL_EMOJI.get(record.levelno, "")
        record.emoji = emoji
        return super().format(record)


_DEFAULT_FMT = "%(emoji)s [%(name)s] %(message)s"
_DATE_FMT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with emoji formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers when re-imported
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = _EmojiFormatter(_DEFAULT_FMT, datefmt=_DATE_FMT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def configure_root(level: int = logging.INFO):
    """Configure root logger; useful for CLI entry points."""
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = _EmojiFormatter("%(emoji)s %(message)s", datefmt=_DATE_FMT)
        handler.setFormatter(formatter)
        root.addHandler(handler)
