import pytest

from raglaw.retrieval.compare import (
    RunSummary,
    comparison_report,
    group_deltas,
    params_changed,
    question_changes,
)

pytestmark = pytest.mark.unit


def record(qid, hit, kind="rule", language="en", register="english", in_scope=True):
    """A per-question record as ``QuestionScore.to_record`` writes it."""
    return {
        "id": qid,
        "kind": kind,
        "language": language,
        "register": register,
        "in_scope": in_scope,
        "hits": {"5": hit} if in_scope else {},
    }


BASE = [
    record("q001", True),
    record("q002", False, kind="lay_term", language="ar", register="colloquial"),
    record("q003", True, kind="lay_term"),
    record("q004", False),
    record("q005", False, in_scope=False, kind="out_of_scope"),
]
RUN = [
    record("q001", True),
    record("q002", True, kind="lay_term", language="ar", register="colloquial"),
    record("q003", False, kind="lay_term"),
    record("q004", True),
    record("q005", False, in_scope=False, kind="out_of_scope"),
]


def summary(name, records, recall, params=None):
    return RunSummary(
        name=name,
        run_id=f"{name}-0123456789",
        params=params or {"retrieval_mode": "hybrid", "rrf_k": "60"},
        metrics={"overall.recall_at_5": recall, "overall.mrr": recall / 2},
        records=records,
    )


def test_questions_fixed_and_broken_are_named():
    changes = question_changes(BASE, RUN)

    assert [r["id"] for r in changes["fixed"]] == ["q002", "q004"]
    assert [r["id"] for r in changes["broken"]] == ["q003"]


def test_questions_missing_from_either_run_are_skipped():
    """Runs on different halves share no questions to compare."""
    changes = question_changes(BASE, [record("q999", True)])

    assert changes == {"fixed": [], "broken": []}


def test_group_deltas_count_questions_and_list_only_groups_that_moved():
    deltas = group_deltas(BASE, RUN)

    assert deltas["kind"] == {
        "rule": {"questions": 2, "baseline": 1, "run": 2},
        # lay_term: q002 fixed and q003 broken cancel out, so it isn't listed
    }
    # English: q001 and q003 found before, q001 and q004 after; 2 either way
    assert deltas["language"] == {"ar": {"questions": 1, "baseline": 0, "run": 1}}
    assert deltas["register"] == {
        "colloquial": {"questions": 1, "baseline": 0, "run": 1}
    }


def test_params_changed_ignores_what_follows_from_other_params():
    base = {
        "retrieval_mode": "hybrid",
        "rrf_k": "60",
        "chunks_sha256": "a",
        "input_hash": "h1",
        "git_sha": "s1",
        "stage": "evaluate_retrieval",
    }
    run = {
        "retrieval_mode": "dense",
        "rrf_k": "60",
        "chunks_sha256": "b",
        "input_hash": "h2",
        "git_sha": "s2",
        "stage": "evaluate_retrieval",
        "x": "1",
    }

    assert params_changed(base, run) == ["retrieval_mode=dense", "x=1 (new)"]


def test_the_report_puts_every_run_against_the_baseline():
    baseline = summary("baseline", BASE, 0.5)
    dense = summary("dense-only", RUN, 0.75, {"retrieval_mode": "dense", "rrf_k": "60"})

    report = comparison_report(baseline, [dense])

    assert report.startswith(
        "# Retrieval comparison: against `baseline` (run baseline-"
    )
    assert "4 in-scope questions: one question moves recall@5 by 0.250" in report
    assert "| baseline | none | 0.500 | 0.250 | – | – | – |" in report
    assert (
        "| dense-only | retrieval_mode=dense | 0.750 (+0.250) | 0.375 (+0.125) "
        "| 2 | 1 | +1 |"
    ) in report
    assert (
        "- Fixed at 5 (2): q002 (lay_term, ar, colloquial), q004 (rule, en, english)"
        in report
    )
    assert "- Broken at 5 (1): q003 (lay_term, en, english)" in report
    assert "| kind: rule | 2 | 1 | 2 | +1 |" in report


def test_a_run_identical_to_the_baseline_says_so():
    report = comparison_report(
        summary("baseline", BASE, 0.5), [summary("again", BASE, 0.5)]
    )

    assert "- Fixed at 5 (0): none" in report
    assert "No group moved." in report
