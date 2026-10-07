"""One MLflow run per pipeline stage execution.

A stage wraps its work in ``stage_run``; the run then ties what went in (git
SHA, input hash, the schema of the records it writes) to what came out
(metrics, the stage's full JSONL log). Runs of one ``dvc repro`` share a git SHA, which groups them.
Stage code records numbers through the yielded handle and never imports
MLflow itself.
"""

import hashlib
import logging
import os
import subprocess
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# MLflow logs a hint about its tracing tools on import; this pipeline doesn't
# trace, and the line would land in every stage's console and log file.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

from raglaw.config import Settings
from raglaw.logging_setup import close_logging, setup_logging
from raglaw.schema import LogEvent, Record

logger = logging.getLogger(__name__)

# The paths that define what a stage computes. A `dvc repro` rewrites dvc.lock
# and the metrics between stages, so those don't count as uncommitted code.
CODE_PATHS = ("src", "scripts", "params.yaml", "dvc.yaml", "pyproject.toml", "uv.lock")


class TrackingConfigError(RuntimeError):
    """The MLflow experiment can't be used as configured."""


@dataclass(frozen=True)
class StageRun:
    """The handle a stage gets inside ``stage_run``."""

    run_id: str
    log_path: Path
    _client: MlflowClient = field(repr=False)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Record the stage's numbers on its run, all at one timestamp."""
        timestamp = int(time.time() * 1000)
        self._client.log_batch(
            self.run_id,
            metrics=[Metric(k, float(v), timestamp, 0) for k, v in metrics.items()],
        )

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Attach a file to the run (e.g., a report), under ``artifact_path``."""
        self._client.log_artifact(self.run_id, str(path), artifact_path=artifact_path)

    def log_params(self, params: Mapping[str, object]) -> None:
        """Record the choices a stage ran with (e.g., a strategy), to compare runs by."""
        self._client.log_batch(
            self.run_id, params=[Param(k, str(v)) for k, v in params.items()]
        )


def sha256_file(path: Path) -> str:
    """
    Hash a file in chunks, so a large input never has to fit in memory.

    returns:
    - digest (str): the file's SHA-256, as hex
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> tuple[str, bool | None]:
    """
    Read the commit the code runs from, and whether it has uncommitted changes.

    returns:
    - state (tuple[str, bool | None]): the HEAD SHA, and whether any file
      under ``CODE_PATHS`` is modified, staged, deleted or new (untracked)
      compared with it; ``("unknown", None)`` outside a git checkout or
      without git
    """
    try:
        sha = _git("rev-parse", "HEAD")
        # Untracked files count: a new module the stage imports is code the
        # commit doesn't contain. Gitignored files (__pycache__) never show.
        changes = _git("status", "--porcelain", "--", *CODE_PATHS)
    except OSError, subprocess.CalledProcessError:
        return "unknown", None
    return sha, bool(changes)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _experiment_id(client: MlflowClient, settings: Settings, name: str) -> str:
    """
    Find or create the named experiment, refusing one whose artifacts live
    elsewhere. Takes the name, so each kind of work passes its own.

    MLflow fixes an experiment's artifact location when it is created, so an
    experiment made before the S3 root was configured would keep writing
    locally without any error. That case fails here instead.

    exceptions:
    - TrackingConfigError: the experiment is deleted, or its artifact
      location differs from the configured one
    """
    wanted = (
        f"{settings.mlflow_artifact_root.rstrip('/')}/{name}"
        if settings.mlflow_artifact_root
        else None
    )
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return client.create_experiment(name, artifact_location=wanted)
    if experiment.lifecycle_stage == "deleted":
        raise TrackingConfigError(
            f"Experiment {name!r} is deleted; restore it "
            f"(`mlflow experiments restore --experiment-id {experiment.experiment_id}`) "
            "or remove it for good (`mlflow gc`)."
        )
    if wanted is not None and experiment.artifact_location != wanted:
        raise TrackingConfigError(
            f"Experiment {name!r} stores artifacts at {experiment.artifact_location}, "
            f"but RAGLAW_MLFLOW_ARTIFACT_ROOT expects {wanted}. MLflow can't move "
            "an experiment's artifacts; delete the experiment and rerun."
        )
    return experiment.experiment_id


# MLflow caps a tag value at 5,000 characters; the reason fits well within it.
ERROR_TAG_CHARS = 1_000


def _tag_error(client: MlflowClient, run_id: str, exc: BaseException) -> None:
    """
    Tag a failed run with ``error: <type>: <message>``, shown on its Overview.

    The full traceback is in the run's log artifact. Tagging never masks the
    stage's own exception: a tracking failure here is only logged.
    """
    reason = f"{type(exc).__name__}: {exc}"[:ERROR_TAG_CHARS]
    try:
        client.set_tag(run_id, "error", reason)
    except Exception:
        logger.warning("Could not tag the failed run with its error", exc_info=True)


@contextmanager
def stage_run(
    stage: str,
    *,
    input_hash: str,
    output_model: type[Record] | None = None,
    settings: Settings | None = None,
) -> Generator[StageRun]:
    """
    Run one pipeline stage inside its own MLflow run.

    Sets up the stage's logging, records params and tags up front, and on
    exit attaches the stage's log file, even when the stage fails, since a
    failed run is when the log matters most. An exception marks the run
    FAILED (an interrupt, KILLED) and is re-raised, never swallowed; a failed
    run is also tagged ``error`` with the exception's type and message.
    ``output_model`` names the record type the stage writes, logged as the
    ``output_schema`` param; a stage that writes only metrics passes none.

    returns:
    - run (StageRun): the run's id, its log file, and ``log_metrics``

    exceptions:
    - TrackingConfigError: the experiment can't be used as configured
    """
    settings = settings or Settings()
    if settings.aws_profile:
        os.environ.setdefault("AWS_PROFILE", settings.aws_profile)
    log_path = setup_logging(stage, logs_dir=settings.paths.logs_dir)

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    sha, dirty = git_state()
    experiment_id = _experiment_id(
        client, settings, settings.tracking.experiments.corpus
    )
    run = client.create_run(
        experiment_id,
        run_name=stage,
        tags={
            "stage": stage,
            "git_dirty": "unknown" if dirty is None else str(dirty).lower(),
        },
    )
    run_id = run.info.run_id
    params = {"stage": stage, "git_sha": sha, "input_hash": input_hash}
    if output_model is not None:
        params["output_schema"] = output_model.schema_id()
    client.log_batch(run_id, params=[Param(k, v) for k, v in params.items()])

    stage_fields = {"stage": stage, "run_id": run_id}
    logger.info(
        "Stage started",
        extra={
            **stage_fields,
            "input_hash": input_hash,
            "event_type": LogEvent.STAGE_START,
        },
    )
    status = "FINISHED"
    stage_exc: BaseException | None = None
    try:
        yield StageRun(run_id=run_id, log_path=log_path, _client=client)
    except Exception as exc:
        status, stage_exc = "FAILED", exc
        logger.exception(
            "Stage failed", extra={**stage_fields, "event_type": LogEvent.STAGE_FAILED}
        )
        _tag_error(client, run_id, exc)
        raise
    except BaseException as exc:
        status, stage_exc = "KILLED", exc
        logger.warning(
            "Stage interrupted",
            extra={**stage_fields, "event_type": LogEvent.STAGE_FAILED},
        )
        raise
    else:
        logger.info(
            "Stage completed",
            extra={**stage_fields, "event_type": LogEvent.STAGE_COMPLETED},
        )
    finally:
        close_logging()
        try:
            client.log_artifact(run_id, str(log_path), artifact_path="logs")
        except Exception as upload_exc:
            if stage_exc is None:
                # The stage succeeded; the upload failure is the error to report.
                status = "FAILED"
                raise
            stage_exc.add_note(f"Uploading the stage log also failed: {upload_exc!r}")
        finally:
            client.set_terminated(run_id, status)
