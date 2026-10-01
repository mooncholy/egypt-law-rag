"""U16: pipeline logging writes a console stream and a JSONL file per stage."""

import json
import logging

import pytest

from raglaw.logging_setup import close_logging, log_anomaly, setup_logging
from raglaw.schema import LogEvent

pytestmark = pytest.mark.unit

logger = logging.getLogger("raglaw.test_stage")

ANOMALY = {"page": 7, "row_index": 3, "row_class": "anomaly", "article_number": 54}


def stage_handlers(log_path):
    """The console and file handlers setup_logging put on the root logger."""
    root = logging.getLogger()
    files = [
        h
        for h in root.handlers
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
    ]
    # Exact type: pytest's own capture handlers subclass StreamHandler.
    consoles = [h for h in root.handlers if type(h) is logging.StreamHandler]
    return consoles, files


def read_jsonl(log_path):
    return [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]


def test_log_file_is_named_after_the_stage(pipeline_log, tmp_path):
    assert pipeline_log.parent == tmp_path
    assert pipeline_log.name.startswith("test-stage-")
    assert pipeline_log.suffix == ".jsonl"


def test_attaches_console_at_info_and_file_at_debug(pipeline_log):
    consoles, files = stage_handlers(pipeline_log)

    assert [h.level for h in consoles] == [logging.INFO]
    assert [h.level for h in files] == [logging.DEBUG]


def test_calling_twice_adds_no_duplicate_handlers(pipeline_log, tmp_path):
    second = setup_logging("test-stage", logs_dir=tmp_path)

    consoles, files = stage_handlers(second)
    assert len(consoles) == 1
    assert len(files) == 1


def test_close_logging_removes_its_handlers(pipeline_log):
    close_logging()

    assert stage_handlers(pipeline_log) == ([], [])


def test_anomaly_line_in_jsonl_carries_its_fields(pipeline_log):
    log_anomaly(logger, "Row matches no class", **ANOMALY)
    close_logging()

    [line] = [ln for ln in read_jsonl(pipeline_log) if ln.get("event_type")]
    assert line["level"] == "WARNING"
    assert line["event_type"] == LogEvent.ANOMALY
    assert {k: line[k] for k in ANOMALY} == ANOMALY


def test_debug_reaches_the_file_without_formatter_fields(pipeline_log):
    logger.info("visible on the console")
    logger.debug("file only")
    close_logging()

    lines = read_jsonl(pipeline_log)
    assert [ln["message"] for ln in lines] == ["visible on the console", "file only"]
    # The console formatter sets asctime on the record; it must not leak in.
    assert all("asctime" not in ln for ln in lines)


def test_anomaly_without_article_still_carries_the_key(log_records):
    log_anomaly(
        logger,
        "Continuation row with no previous page",
        page=46,
        row_index=0,
        row_class="continuation",
        article_number=None,
    )

    [record] = log_records(level=logging.WARNING, event_type=LogEvent.ANOMALY)
    assert record["article_number"] is None
    assert record["row_class"] == "continuation"
