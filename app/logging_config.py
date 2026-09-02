"""Structured logging setup.

Logs are emitted as one JSON object per line so they can be shipped straight
into a log aggregator without a regex parser in between. Application code
attaches structured fields by passing extra={"context": {...}}.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from app.config import LOG_LEVEL


class JsonLogFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Install the JSON formatter on the root logger.

    Existing handlers are replaced rather than added to, so that uvicorn's
    default text handlers do not produce a second, unparseable copy of every
    line when the app is started through uvicorn.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)

    for uvicorn_logger in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(uvicorn_logger)
        logger.handlers = []
        logger.propagate = True
