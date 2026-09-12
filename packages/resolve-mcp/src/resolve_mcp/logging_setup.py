"""Logging setup for resolve-mcp.

Single structlog config that the server process enables at startup. Tests don't need
this; they import the backend directly.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib + structlog to write JSON lines to stderr.

    stderr is not a preference here: on the stdio transport, stdout carries the
    JSON-RPC frames, so a single log line printed there corrupts the stream and
    the client fails to parse the session. structlog's default
    ``PrintLoggerFactory`` writes to stdout, hence the explicit factory below.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
