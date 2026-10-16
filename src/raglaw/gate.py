"""The CI quality gate (Phase 11): fail a build whose answers or retrieval regress.

CI scores the 20-question CI set (``answer_evaluation.subset: ci``) and the
tuning half of the retrieval eval set, then ``scripts/ci_gate.py`` hands both
metrics files here. The build fails when:

- faithfulness on the CI set is below ``gate.min_faithfulness`` (the handbook's
  0.75), or the answers weren't scored on the whole CI set (``ci_size``);
- the retrieval metrics aren't the tuning half's, or recall@5 on it is more
  than ``gate.max_recall_drop`` questions below the base branch's committed
  score (one question is 1/65 of recall).
"""

from dataclasses import dataclass
from typing import Any

from raglaw.config import Gate


@dataclass(frozen=True)
class GateResult:
    """Whether the build passes, and one line per condition checked."""

    passed: bool
    lines: list[str]


def evaluate_gate(
    answers: dict[str, Any],
    retrieval: dict[str, Any],
    base_retrieval: dict[str, Any] | None,
    gate: Gate,
    ci_size: int,
) -> GateResult:
    """
    Decide the gate from ``docs/metrics/answers.json``, ``retrieval.json`` and
    the base branch's ``retrieval.json`` (None: no base to compare with);
    ``ci_size`` is the number of questions the CI set holds.

    returns:
    - result (GateResult): passed only when every condition holds
    """
    lines, passed = [], True
    faithfulness = answers["overall"]["faithfulness"]
    subset, scored = answers.get("subset"), answers.get("questions")
    ok = (
        subset == "ci"
        and scored == ci_size
        and faithfulness is not None
        and faithfulness >= gate.min_faithfulness
    )
    passed &= ok
    lines.append(
        f"{'pass' if ok else 'FAIL'}: faithfulness {faithfulness} on the {subset} set "
        f"({scored} of its {ci_size} questions), minimum {gate.min_faithfulness}"
    )
    recall = retrieval["overall"]["recall_at_5"]
    questions = retrieval["overall"]["questions"]
    if retrieval.get("split") != "tuning":
        passed = False
        lines.append(
            f"FAIL: recall@5 {recall:.3f} is on the {retrieval.get('split')} half; "
            "the gate compares the tuning half"
        )
    elif base_retrieval is None or base_retrieval.get("split") != "tuning":
        lines.append(
            f"skip: recall@5 {recall:.3f}, no tuning-half score on the base branch"
        )
    else:
        base = base_retrieval["overall"]["recall_at_5"]
        drop = round((base - recall) * questions)
        ok = drop <= gate.max_recall_drop
        passed &= ok
        lines.append(
            f"{'pass' if ok else 'FAIL'}: recall@5 {recall:.3f} against the base "
            f"branch's {base:.3f} ({drop:+d} questions lost of {questions}; at most "
            f"{gate.max_recall_drop})"
        )
    return GateResult(passed, lines)
