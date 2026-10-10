"""Compare answer evaluation runs (Phase 9): refusals, shared questions, C20.

Kept apart from ``answer_eval`` (a dependency of the ``evaluate_answers``
stage), as ``retrieval.compare`` is from ``retrieval.scoring``: changing a
report never makes the stage stale. ``scripts/compare_answers.py`` feeds it
MLflow runs.
"""

import re
import statistics
from collections.abc import Iterable
from typing import Any

from raglaw.answer_eval import METRICS

# How the generator says it found nothing, in either language. Its answers are
# short; one below MIN_ANSWER_CHARS (e.g. a bare article label) says nothing.
REFUSAL = re.compile(
    r"do(es)? not (address|provide|directly address|specify)|not directly addressed|"
    r"لا تتناول|لا يتناول|لا يوجد في المقالات|لا تتضمن",
    re.IGNORECASE,
)
MIN_ANSWER_CHARS = 40


def answer_kind(row: dict[str, Any]) -> str:
    """
    What kind of reply a per-question record holds.

    returns:
    - kind (str): ``gate`` (the no-answer gate refused; no LLM call),
      ``refusal`` (the generator said the articles don't address it, or said
      nothing), or ``answer``
    """
    if row["unaddressed"]:
        return "gate"
    text = row["answer"]
    return (
        "refusal" if REFUSAL.search(text) or len(text) < MIN_ANSWER_CHARS else "answer"
    )


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def _cell(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def comparison_report(
    runs: dict[str, tuple[dict[str, float], list[dict[str, Any]]]],
    repeat: tuple[str, str] | None = None,
) -> str:
    """
    Compare answer runs: their RAGAS means, the same means on the questions
    every run answered with substance, and how often each refused.

    Means over all answered questions mix different question sets: the gate
    refuses only when reranking supplies a score. The shared-question view
    compares like with like. ``repeat`` names two runs of one config (C20).

    returns:
    - report (str): Markdown
    """
    lines = ["# Answer comparison", ""]
    lines += [
        "| Run | " + " | ".join(METRICS) + " | Answered | Gate refused "
        "| Generator refused | ...with the expected articles given |",
        "| --- |" + " --- |" * (len(METRICS) + 4),
    ]
    for name, (metrics, rows) in runs.items():
        refused = [r for r in rows if answer_kind(r) == "refusal"]
        given = sum(
            bool(r["expected"]) and set(r["expected"]) <= set(r["articles"])
            for r in refused
        )
        lines.append(
            f"| {name} | "
            + " | ".join(_cell(metrics.get(f"overall.{m}")) for m in METRICS)
            + f" | {sum(not r['unaddressed'] for r in rows)} "
            f"| {sum(r['unaddressed'] for r in rows)} | {len(refused)} | {given} |"
        )
    shared = set.intersection(
        *(
            {r["id"] for r in rows if answer_kind(r) == "answer"}
            for _, rows in runs.values()
        )
    )
    lines += [
        "",
        f"## On the {len(shared)} questions every run answered with substance",
        "",
        (
            "| Run | faithfulness | faithfulness (ar) | faithfulness (en) "
            "| context_precision | context_recall |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, (_, rows) in runs.items():
        sub = [r for r in rows if r["id"] in shared]

        def m(metric: str, lang: str | None = None, sub=sub) -> str:
            return _cell(
                _mean(
                    r["scores"].get(metric)
                    for r in sub
                    if lang is None or r["language"] == lang
                )
            )

        lines.append(
            f"| {name} | {m('faithfulness')} | {m('faithfulness', 'ar')} "
            f"| {m('faithfulness', 'en')} | {m('context_precision')} "
            f"| {m('context_recall')} |"
        )
    if repeat:
        (ma, ra), (mb, rb) = runs[repeat[0]], runs[repeat[1]]
        fa, fb = ma["overall.faithfulness"], mb["overall.faithfulness"]
        same = sum(
            a["answer"] == b["answer"]
            for a, b in zip(
                sorted(ra, key=lambda r: r["id"]),
                sorted(rb, key=lambda r: r["id"]),
                strict=True,
            )
        )
        lines += [
            "",
            f"## Repeatability (C20): {repeat[0]} against {repeat[1]}",
            "",
            (
                f"Faithfulness {fa:.4f} against {fb:.4f}: a difference of "
                f"{abs(fa - fb):.4f} (C20 allows 0.02). {same} of {len(ra)} answers "
                "are word-for-word identical, so the difference is the judge's."
            ),
        ]
    return "\n".join(lines) + "\n"
