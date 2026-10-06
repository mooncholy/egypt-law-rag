"""The ``profile`` stage: check the raw PDF against the analyzed baselines.

Every repair downstream assumes the PDF analyzed in
``docs/reports/0_source_pdf_analysis.md``. This stage re-measures it (G1 to G6
of the corpus build brief) and fails on any mismatch, so a changed source stops
``dvc repro`` before a bad corpus becomes a DVC version.

The same measuring pass writes the evidence behind the report (``summary.json``
and the TSVs in ``docs/analysis/source_pdf/``), which
``docs/reports/0_source_pdf_analysis.sha256`` pins byte for byte.

On an excerpt (``--first-page``), only the per-page gates apply (G2 to G4):
an excerpt has no tag tree and holds a fraction of the rows and defects. Its
evidence is written only to an explicit ``--evidence-dir``, so it never
overwrites the full document's.

Run as ``python -m raglaw.ingest.profile``.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from raglaw.config import Settings
from raglaw.ingest.measure import (
    SourceFacts,
    defect_counts,
    measure_document,
    write_evidence,
)
from raglaw.logging_setup import log_anomaly
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)

PAGES = 170  # P1
FONTS = ["Arial-BoldMT", "ArialMT", "Calibri", "Calibri-Bold"]  # P2
ROWS_DETECTED = 1_463  # P4
ROWS_TAGGED = 1_464  # P4
# Page 1's borderless promulgation block is the one tagged row that detection
# doesn't find (P4, P7): [page, detected rows, tagged rows].
PAGES_DETECTED_NE_TAGGED = [[1, 9, 10]]
# Raw defect counts, before any repair (G6).
RAW_DEFECTS = {
    "lam_alef_signatures": 3_500,  # P23
    "rtl_digit_runs": 1_167,  # P24
    "header_typos": 2,  # P9
    "same_line_headers": 6,  # P8
    "spaced_mada_headers": 13,  # P10
}


class ProfileGateError(RuntimeError):
    """At least one profile gate failed: the PDF is not the one analyzed."""


@dataclass(frozen=True)
class Gate:
    """One gate's outcome; ``passed`` is ``None`` when the gate doesn't apply."""

    id: str
    passed: bool | None
    expected: Any
    actual: Any

    @property
    def outcome(self) -> str:
        return {True: "pass", False: "fail", None: "skip"}[self.passed]


def dvc_md5(dvc_file: Path) -> str:
    """The md5 that a ``.dvc`` file records for its single output."""
    meta = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    return meta["outs"][0]["md5"]


def md5_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def _gate(gate_id: str, expected: Any, actual: Any, applies: bool = True) -> Gate:
    return Gate(gate_id, (expected == actual) if applies else None, expected, actual)


def profile_metrics(
    facts: SourceFacts, *, excerpt: bool, dvc_file: Path | None = None
) -> tuple[dict[str, Any], list[Gate]]:
    """
    Compute the profile metrics and check each gate.

    ``dvc_file`` is the PDF's ``.dvc`` file, for G1; it's ignored on an excerpt.
    DVC records an md5 for the PDF, so G1 compares md5s. The PDF's sha256 is
    the stage run's ``input_hash``, so it isn't repeated here.

    returns:
    - metrics (dict[str, Any]): the values ``source_profile.json`` holds
    - gates (list[Gate]): G1 to G6, each passed, failed or skipped
    """
    pages = facts.pages
    md5 = md5_file(facts.path)
    raw = defect_counts(facts)
    detected = sum(p.n_rows for p in pages)
    tr_per_page = facts.struct["tr_per_page"]
    ne_tagged = [
        [p.number, p.n_rows, tr_per_page.get(p.number, 0)]
        for p in pages
        if p.n_rows != tr_per_page.get(p.number, 0)
    ]
    metrics: dict[str, Any] = {
        "pdf_md5": md5,
        "pages_total": facts.page_count,
        "pages_with_text_layer": sum(p.has_text for p in pages),
        "pages_one_2col_table": sum(
            p.n_tables == 1 and p.col_counts == [2] for p in pages
        ),
        "font_set": sorted(set().union(*(p.fonts for p in pages))),
        "rows_detected": detected,
        "rows_tagged": sum(tr_per_page.values()),
        "pages_detected_ne_tagged": ne_tagged,
        "rows_missing_a_side": sum(len(p.rows_missing_a_side) for p in pages),
        **{f"raw_{k}": v for k, v in raw.items()},
    }
    full = not excerpt
    gates = [
        _gate("G1", dvc_md5(dvc_file) if full and dvc_file else None, md5, full),
        _gate(
            "G2",
            [facts.page_count if excerpt else PAGES] * 2,
            [metrics["pages_total"], metrics["pages_with_text_layer"]],
        ),
        _gate("G3", facts.page_count, metrics["pages_one_2col_table"]),
        _gate("G4", FONTS, metrics["font_set"]),
        _gate(
            "G5",
            [ROWS_DETECTED, ROWS_TAGGED, PAGES_DETECTED_NE_TAGGED],
            [detected, metrics["rows_tagged"], ne_tagged],
            full,
        ),
        _gate("G6", RAW_DEFECTS, raw, full),
    ]
    metrics["gates"] = {g.id: g.outcome for g in gates}
    return metrics, gates


def report_rows_missing_a_side(facts: SourceFacts) -> None:
    """Log every row with no cell on one side; measuring skips them."""
    for page in facts.pages:
        for index in page.rows_missing_a_side:
            log_anomaly(
                logger,
                "Row has no cell on one side of the table",
                page=page.number,
                row_index=index,
                row_class="missing_side",
                article_number=None,
            )


def run_profile(
    pdf: Path,
    out: Path,
    *,
    first_page: int | None = None,
    evidence_dir: Path | None = None,
) -> tuple[dict[str, Any], list[Gate]]:
    """
    Measure ``pdf``, write its profile metrics to ``out``, and check the gates.

    ``first_page`` marks the PDF as an excerpt starting at that source page.
    ``evidence_dir``, when given, also receives the analysis evidence from the
    same measuring pass.

    returns:
    - metrics (dict[str, Any]): what was written to ``out``
    - gates (list[Gate]): every gate's outcome

    exceptions:
    - ProfileGateError: a gate failed; ``out`` is still written, for diagnosis
    """
    excerpt = first_page is not None
    facts = measure_document(pdf, first_page=first_page or 1)
    if evidence_dir is not None:
        write_evidence(facts, evidence_dir)
    report_rows_missing_a_side(facts)
    metrics, gates = profile_metrics(
        facts, excerpt=excerpt, dvc_file=pdf.with_name(pdf.name + ".dvc")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failed = [g for g in gates if g.passed is False]
    for gate in failed:
        logger.error(
            "Profile gate %s failed",
            gate.id,
            extra={"gate": gate.id, "expected": gate.expected, "actual": gate.actual},
        )
    if failed:
        raise ProfileGateError(
            f"gates {', '.join(g.id for g in failed)} failed: the source PDF "
            "differs from the one analyzed"
        )
    return metrics, gates


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """The metrics MLflow can hold: numbers, plus each gate as 1 (pass) or 0."""
    values = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
    values |= {
        f"gate_{gate}": float(outcome == "pass")
        for gate, outcome in metrics["gates"].items()
        if outcome != "skip"
    }
    return values


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", type=Path, default=settings.paths.raw_pdf)
    ap.add_argument(
        "--first-page",
        type=int,
        default=None,
        help="Source page number of the PDF's first page; marks it as an excerpt.",
    )
    ap.add_argument(
        "--out", type=Path, default=settings.paths.metrics_dir / "source_profile.json"
    )
    ap.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="Where to write the analysis evidence. Defaults to paths.analysis_dir "
        "for the full PDF; an excerpt writes none unless this is given.",
    )
    args = ap.parse_args(argv)
    evidence_dir = args.evidence_dir
    if evidence_dir is None and args.first_page is None:
        evidence_dir = settings.paths.analysis_dir

    try:
        with stage_run(
            "profile", input_hash=sha256_file(args.pdf), settings=settings
        ) as run:
            try:
                metrics, _ = run_profile(
                    args.pdf,
                    args.out,
                    first_page=args.first_page,
                    evidence_dir=evidence_dir,
                )
            except ProfileGateError:
                run.log_metrics(
                    numeric_metrics(json.loads(args.out.read_text("utf-8")))
                )
                raise
            run.log_metrics(numeric_metrics(metrics))
    except ProfileGateError as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
