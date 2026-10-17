"""Compare retrieval runs in MLflow against a baseline, question by question.

Reads the newest finished run of each name in the ``retrieval`` experiment that
logged a per-question file, keeps the runs scored on the baseline's half and on
its device (GPU arithmetic differs from CPU in the last digits, so a cross-device
difference isn't the variant's), and writes
``docs/reports/retrieval_comparison.md`` (``raglaw.retrieval.compare``).
It sits outside ``dvc repro``: its input is MLflow's run history, not a
tracked file.

Run as ``uv run python scripts/compare_retrieval.py [--baseline baseline]``.
"""

import os

# MLflow logs a hint about its tracing tools on import; this script doesn't trace.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import json
import logging
import tempfile
from pathlib import Path

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from raglaw.config import Settings
from raglaw.retrieval.compare import RunSummary, comparison_report

logger = logging.getLogger("compare_retrieval")

QUESTIONS = "questions/retrieval_questions.json"
# Runs from before per-question files were logged carry no `changed` tag.
MARKER_TAG = "changed"
# Runs from before the device was logged all ran on CPU.
DEFAULT_DEVICE = "cpu"


def device_of(run: Run) -> str:
    return run.data.params.get("device", DEFAULT_DEVICE)


def latest_by_name(client: MlflowClient, experiment_id: str) -> dict[str, Run]:
    """The newest finished run of each name that logged a per-question file."""
    runs = client.search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    latest: dict[str, Run] = {}
    for run in runs:
        if MARKER_TAG in run.data.tags:
            latest.setdefault(run.info.run_name, run)
    return latest


def summarize(client: MlflowClient, run: Run, tmp: Path) -> RunSummary:
    local = client.download_artifacts(run.info.run_id, QUESTIONS, str(tmp))
    return RunSummary(
        name=run.info.run_name,
        run_id=run.info.run_id,
        params=run.data.params,
        metrics=run.data.metrics,
        records=json.loads(Path(local).read_text(encoding="utf-8")),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", default="baseline", help="the baseline run's name")
    ap.add_argument(
        "--out",
        type=Path,
        default=settings.paths.reports_dir / "retrieval_comparison.md",
    )
    args = ap.parse_args(argv)
    if settings.aws_profile:  # run artifacts live in S3
        os.environ.setdefault("AWS_PROFILE", settings.aws_profile)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.tracking.experiments.retrieval)
    if experiment is None:
        raise SystemExit("No retrieval experiment yet: run `dvc repro` first")
    latest = latest_by_name(client, experiment.experiment_id)
    if args.baseline not in latest:
        raise SystemExit(
            f"No finished run named {args.baseline!r}; found {sorted(latest)}"
        )
    base_run = latest.pop(args.baseline)
    split, device = base_run.data.params["split"], device_of(base_run)
    kept = [
        run
        for run in latest.values()
        if run.data.params.get("split") == split and device_of(run) == device
    ]
    if left_out := sorted(set(latest) - {r.info.run_name for r in kept}):
        logger.info("Left out (another half or device): %s", ", ".join(left_out))
    with tempfile.TemporaryDirectory() as tmp:
        baseline = summarize(client, base_run, Path(tmp))
        runs = [summarize(client, run, Path(tmp)) for run in kept]
    runs.sort(key=lambda r: (-r.metrics["overall.recall_at_5"], r.name))
    args.out.write_text(comparison_report(baseline, runs), encoding="utf-8")
    logger.info("Compared %d runs against %s: %s", len(runs), baseline.name, args.out)


if __name__ == "__main__":
    main()
