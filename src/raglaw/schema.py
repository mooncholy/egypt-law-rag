from enum import StrEnum
from typing import Final

SCHEMA_VERSION: Final = "1.0"


class LogEvent(StrEnum):
    """Machine-readable event names attached to log lines.

    Every log line carries one of these as ``event_type``, so a log search can
    filter on the event rather than on wording that may change.
    """

    # System events
    STARTUP = "system_startup"
    SHUTDOWN = "system_shutdown"

    # HTTP Request events
    REQUEST_START = "request_start"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"

    VALIDATION_FAILED = "validation_failed"

    # Pipeline events: each DVC stage logs its input hash at start and its
    # output counts at completion; anomalies are never silently absorbed.
    STAGE_START = "stage_start"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    ANOMALY = "anomaly"
