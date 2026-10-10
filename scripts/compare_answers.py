"""Compare answer evaluation runs in MLflow: RAGAS means, shared questions, refusals.

Reads the newest finished run of each name in the ``answers`` experiment, and
writes ``docs/reports/answer_comparison.md`` (``raglaw.answer_eval``). With
``--repeat A B`` it also checks C20 on two runs of one config. It sits outside
``dvc repro``: its input is MLflow's run history, not a tracked file.

Run as ``uv run python scripts/compare_answers.py [--repeat A B]``.
"""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import json
import logging
import tempfile
from pathlib import Path

from mlflow.tracking import MlflowClient

from raglaw.answer_eval import comparison_report
from raglaw.config import Settings

logger = logging.getLogger("compare_answers")

QUESTIONS = "questions/answer_questions.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repeat", nargs=2, metavar=("A", "B"), help="two runs of one config (C20)"
    )
    ap.add_argument(
        "--out", type=Path, default=settings.paths.reports_dir / "answer_comparison.md"
    )
    args = ap.parse_args()
    client = MlflowClient(settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.tracking.experiments.answers)
    latest = {}
    for run in client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    ):
        latest[run.info.run_name] = run  # the newest of each name wins
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, run in latest.items():
            local = client.download_artifacts(
                run.info.run_id, QUESTIONS, f"{tmp}/{name}"
            )
            runs[name] = (run.data.metrics, json.loads(Path(local).read_text("utf-8")))
    args.out.write_text(
        comparison_report(runs, tuple(args.repeat) if args.repeat else None), "utf-8"
    )
    logger.info("Compared %d runs: %s", len(runs), args.out)


if __name__ == "__main__":
    main()
