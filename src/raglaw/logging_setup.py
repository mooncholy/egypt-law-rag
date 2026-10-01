"""Logging for the pipeline's DVC stages.

The API logs JSON to stdout through ``raglaw.logging_conf``. A pipeline stage
instead logs readable lines to the console and every detail to a JSONL file,
which the stage's MLflow run keeps as an artifact.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raglaw.config import Settings
from raglaw.logging_conf import JsonFormatter
from raglaw.schema import LogEvent

CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
# Loggers whose DEBUG records reach the file. ``__main__`` is the stage module
# itself when run as ``python -m raglaw.ingest.<stage>``. Everything else stays
# at INFO, so boto3 and MLflow internals don't flood the file.
DEBUG_LOGGERS = ("raglaw", "__main__")
_HANDLER_TAG = "_raglaw_pipeline"


def setup_logging(stage: str, logs_dir: Path | None = None) -> Path:
    """
    Send pipeline logs to the console (INFO, readable) and to a JSONL file
    (DEBUG) named after the stage.

    Calling it again replaces the handlers it added before, so a process never
    writes each line twice.

    returns:
    - log_path (Path): ``<logs_dir>/<stage>-<UTC timestamp>.jsonl``, for the
      stage's MLflow run to attach
    """
    close_logging()
    logs_dir = logs_dir if logs_dir is not None else Settings().paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"{stage}-{timestamp}.jsonl"

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for name in DEBUG_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)
    for handler in (console, file_handler):
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)
    return log_path


def close_logging() -> None:
    """
    Remove and close the handlers ``setup_logging`` added.

    A stage calls it before attaching its log file to MLflow, so the file is
    flushed and complete.
    """
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(handler)
        handler.close()


def log_anomaly(
    logger: logging.Logger,
    message: str,
    *,
    page: int,
    row_index: int,
    row_class: str,
    article_number: int | None,
    **fields: Any,
) -> None:
    """
    Log one anomaly at WARNING with the fields every anomaly must carry.

    The keyword-only fields make a call without them fail, so the report
    generated from the logs can always place each anomaly.
    ``article_number`` is None for rows that belong to no article.
    """
    logger.warning(
        message,
        extra={
            "event_type": LogEvent.ANOMALY,
            "page": page,
            "row_index": row_index,
            "row_class": row_class,
            "article_number": article_number,
            **fields,
        },
    )
