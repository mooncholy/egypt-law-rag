"""CI's quality gate (Phase 11): exit 1 when answers or retrieval regress.

Reads ``docs/metrics/answers.json`` (the 20-question CI set), ``retrieval.json``
(the tuning half) and, with ``--base-retrieval``, the base branch's committed
``retrieval.json``; the decision is ``raglaw.gate.evaluate_gate``. Each line
also goes to the job summary when ``GITHUB_STEP_SUMMARY`` is set.

Run as ``uv run python scripts/ci_gate.py [--base-retrieval PATH]``.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from raglaw.config import Settings
from raglaw.gate import evaluate_gate

logger = logging.getLogger("ci_gate")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    metrics = Path(settings.paths.metrics_dir)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base-retrieval", type=Path, help="the base branch's retrieval.json"
    )
    args = ap.parse_args()
    base = (
        json.loads(args.base_retrieval.read_text("utf-8"))
        if args.base_retrieval and args.base_retrieval.exists()
        else None
    )
    result = evaluate_gate(
        json.loads((metrics / "answers.json").read_text("utf-8")),
        json.loads((metrics / "retrieval.json").read_text("utf-8")),
        base,
        settings.gate,
    )
    for line in result.lines:
        logger.info(line)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(
                "### Quality gate\n\n"
                + "\n".join(f"- {ln}" for ln in result.lines)
                + "\n"
            )
    if not result.passed:
        raise SystemExit("quality gate failed")


if __name__ == "__main__":
    main()
