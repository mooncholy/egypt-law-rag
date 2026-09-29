from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime as dt
from typing import Any, Final

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

SERVER_LOGGERS: Final[tuple[str, ...]] = ("uvicorn", "uvicorn.error")
ACCESS_LOGGER: Final[str] = "uvicorn.access"
_RESERVED_KEYS = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", (), None))
)


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON, one object per line.

    Every line carries the current correlation ID, so one request can be traced
    across the log even when requests interleave. Any extra field passed through
    ``logger.info(..., extra={...})`` is promoted to a top-level key.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Render one log record as a single JSON object on one line.

        returns:
        - line (str): the record as JSON, carrying the timestamp, level, logger
          name, correlation ID, message, and any extra fields the caller passed
        """
        corr_id = correlation_id_var.get() or "N/A"
        payload: dict[str, Any] = {
            "timestamp": dt.fromtimestamp(record.created, tz=datetime.UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": corr_id,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_KEYS and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return the named logger, attaching a JSON handler on stdout the first time
    it is asked for.

    Repeat calls return the same logger untouched, so importing a module twice
    cannot produce duplicated log lines.

    returns:
    - logger (logging.Logger): a logger writing JSON to stdout at the level in
      ``PRODML_LOG_LEVEL``, defaulting to INFO
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("PRODML_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


def configure_server_logging() -> None:
    """Bring uvicorn's own log lines into the same JSON format as ours.

    uvicorn configures its logging before it imports the application, so
    replacing the handlers at import time is what makes this stick. Without it
    the process emits two formats at once, and anything reading the stream line
    by line, ``jq`` or a log shipper, fails on the plain-text lines.

    ``uvicorn.access`` is silenced rather than reformatted. The correlation ID
    middleware already logs every request with its path, method, status, latency
    and correlation ID, while uvicorn's access record is emitted after that
    context is torn down and so carries no correlation ID at all. One request
    log per request, with the ID attached, beats two without.
    """
    for name in SERVER_LOGGERS:
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        server_logger.addHandler(handler)
        server_logger.setLevel(os.getenv("PRODML_LOG_LEVEL", "INFO").upper())
        server_logger.propagate = False

    access_logger = logging.getLogger(ACCESS_LOGGER)
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.disabled = True
