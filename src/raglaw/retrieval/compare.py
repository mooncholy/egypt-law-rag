"""Compare retrieval runs against a baseline, question by question.

A run's aggregate recall hides which questions moved: on the 65 in-scope
tuning questions, one question is 1.5 points of recall@5, and +3 net could be
3 fixed or 6 fixed and 3 broken. Each run's per-question file (written by
``evaluate_retrieval``) lets the two be told apart. This module holds the pure
logic; ``scripts/compare_retrieval.py`` reads the runs from MLflow.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

K = 5
GROUPS = ("kind", "language", "register")
# Params that follow from others, so listing them would name a consequence as
# a cause: the chunks and input hashes change with any chunking or index
# setting, and `stage_run` logs the commit and the stage on every run.
DERIVED_PARAMS = frozenset({"chunks_sha256", "input_hash", "git_sha", "stage"})


@dataclass(frozen=True)
class RunSummary:
    """One retrieval run, as the comparison reads it."""

    name: str
    run_id: str
    params: Mapping[str, str]
    metrics: Mapping[str, float]  # flattened, e.g. ``overall.recall_at_5``
    records: Sequence[Mapping[str, Any]]  # ``QuestionScore.to_record`` per question


def params_changed(base: Mapping[str, str], run: Mapping[str, str]) -> list[str]:
    """
    The params a run sets differently from the baseline.

    A param the run doesn't log isn't a change: the run predates it, and every
    switch added later defaults to the behaviour before it existed.

    returns:
    - changes (list[str]): sorted ``key=value``; a key the baseline doesn't
      log is marked ``(new)``
    """
    out = []
    for key in sorted((run.keys()) - DERIVED_PARAMS):
        if key not in base:
            out.append(f"{key}={run[key]} (new)")
        elif base[key] != run[key]:
            out.append(f"{key}={run[key]}")
    return out


def params_unlogged(base: Mapping[str, str], run: Mapping[str, str]) -> list[str]:
    """
    The baseline's params a run didn't log, because it predates them.

    returns:
    - keys (list[str]): sorted
    """
    return sorted(base.keys() - run.keys() - DERIVED_PARAMS)


def _in_scope(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {r["id"]: r for r in records if r["in_scope"]}


def question_changes(
    base: Sequence[Mapping[str, Any]], run: Sequence[Mapping[str, Any]], k: int = K
) -> dict[str, list[Mapping[str, Any]]]:
    """
    The in-scope questions a run finds at ``k`` that the baseline didn't, and back.

    Only questions in both runs count.

    returns:
    - changes (dict[str, list]): ``fixed`` and ``broken`` records, by id
    """
    before, after = _in_scope(base), _in_scope(run)
    shared = sorted(before.keys() & after.keys())
    hit = str(k)
    return {
        "fixed": [
            after[i]
            for i in shared
            if after[i]["hits"][hit] and not before[i]["hits"][hit]
        ],
        "broken": [
            after[i]
            for i in shared
            if before[i]["hits"][hit] and not after[i]["hits"][hit]
        ],
    }


def group_deltas(
    base: Sequence[Mapping[str, Any]], run: Sequence[Mapping[str, Any]], k: int = K
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Questions found at ``k`` per group, before and after, for groups that moved.

    returns:
    - deltas (dict): per ``kind`` / ``language`` / ``register``, each group
      whose count changed: its ``questions``, and how many the ``baseline``
      and the ``run`` find
    """
    before, after = _in_scope(base), _in_scope(run)
    shared = sorted(before.keys() & after.keys())
    hit = str(k)
    deltas: dict[str, dict[str, dict[str, int]]] = {}
    for attr in GROUPS:
        groups: dict[str, dict[str, int]] = {}
        for i in shared:
            g = groups.setdefault(
                after[i][attr], {"questions": 0, "baseline": 0, "run": 0}
            )
            g["questions"] += 1
            g["baseline"] += before[i]["hits"][hit]
            g["run"] += after[i]["hits"][hit]
        deltas[attr] = {
            name: g for name, g in sorted(groups.items()) if g["baseline"] != g["run"]
        }
    return deltas


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _questions(records: Sequence[Mapping[str, Any]]) -> str:
    return (
        ", ".join(
            f"{r['id']} ({r['kind']}, {r['language']}, {r['register']})"
            for r in records
        )
        or "none"
    )


def comparison_report(
    baseline: RunSummary, runs: Sequence[RunSummary], k: int = K
) -> str:
    """
    Write the comparison of ``runs`` against ``baseline``.

    returns:
    - text (str): Markdown: a table of every run's changed params, recall@k
      and MRR (with deltas), and questions fixed, broken and net; then, per
      run, the fixed and broken questions and the groups that moved
    """
    recall, mrr = f"overall.recall_at_{k}", "overall.mrr"
    n = len(_in_scope(baseline.records))
    step = 1 / n if n else 0.0
    lines = [
        f"# Retrieval comparison: against `{baseline.name}` (run {baseline.run_id})",
        "",
        (
            f"Each run changes the listed params against the baseline. On {n} "
            f"in-scope questions: one question moves recall@{k} by {step:.3f}, so "
            "a net change of one question is within noise."
        ),
        "",
        f"| Run | Changed | Recall@{k} | MRR | Fixed | Broken | Net |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {baseline.name} | none | {baseline.metrics[recall]:.3f} | "
            f"{baseline.metrics[mrr]:.3f} | – | – | – |"
        ),
    ]
    sections = []
    for run in runs:
        changed = params_changed(baseline.params, run.params)
        changes = question_changes(baseline.records, run.records, k)
        net = len(changes["fixed"]) - len(changes["broken"])
        r, m = run.metrics[recall], run.metrics[mrr]
        lines.append(
            f"| {run.name} | {', '.join(changed) or 'none'} | "
            f"{r:.3f} ({_signed(r - baseline.metrics[recall])}) | "
            f"{m:.3f} ({_signed(m - baseline.metrics[mrr])}) | "
            f"{len(changes['fixed'])} | {len(changes['broken'])} | {net:+d} |"
        )
        sections += [
            f"## {run.name} (run {run.run_id})",
            "",
            f"- Changed: {', '.join(changed) or 'none'}",
            *(
                [f"- Not logged (the run predates them): {', '.join(unlogged)}"]
                if (unlogged := params_unlogged(baseline.params, run.params))
                else []
            ),
            f"- Fixed at {k} ({len(changes['fixed'])}): {_questions(changes['fixed'])}",
            f"- Broken at {k} ({len(changes['broken'])}): {_questions(changes['broken'])}",
            "",
        ]
        moved = [
            (f"{attr}: {name}", g)
            for attr, groups in group_deltas(baseline.records, run.records, k).items()
            for name, g in groups.items()
        ]
        if moved:
            sections += [
                "| Group | Questions | Baseline found | Run found | Δ |",
                "| --- | --- | --- | --- | --- |",
            ]
            sections += [
                f"| {label} | {g['questions']} | {g['baseline']} | {g['run']} | "
                f"{g['run'] - g['baseline']:+d} |"
                for label, g in moved
            ]
        else:
            sections.append("No group moved.")
        sections.append("")
    return "\n".join([*lines, "", *sections]).rstrip("\n") + "\n"
