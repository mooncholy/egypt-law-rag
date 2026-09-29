from enum import StrEnum


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

    # Pipeline events
    DATA_INGESTION_START = "data_ingestion_start"
    DATA_INGESTION_COMPLETED = "data_ingestion_completed"
    DATA_INGESTION_FAILED = "data_ingestion_failed"
    DATA_MANIPULATION_FAILED = "data_manipulation_failed"
    TRAINING_START = "training_start"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    EXPORT_COMPLETED = "export_completed"
    EXPORT_FAILED = "export_failed"
    PREDICTION_COMPLETED = "prediction_completed"
    PREDICTION_FAILED = "prediction_failed"
    VALIDATION_FAILED = "validation_failed"
