import json
import subprocess

import pytest

from raglaw.ingest import profile
from raglaw.ingest.profile import (
    FONTS,
    PAGES,
    RAW_DEFECTS,
    ROWS_DETECTED,
    ROWS_TAGGED,
    ProfileGateError,
    run_profile,
)

# --- Unit: the stage on a synthetic PDF ----------------------------------------

# What the synthetic PDF (tests/conftest.py) measures as: two pages, seven rows,
# no tag tree, and one header with body text on its line ("Article 2 No ...").
SYNTHETIC_BASELINES = {
    "PAGES": 2,
    "FONTS": ["Helvetica", "Helvetica-Bold"],
    "ROWS_DETECTED": 7,
    "ROWS_TAGGED": 0,
    "PAGES_DETECTED_NE_TAGGED": [[1, 4, 0], [2, 3, 0]],
    "RAW_DEFECTS": {
        "lam_alef_signatures": 0,
        "rtl_digit_runs": 0,
        "header_typos": 0,
        "same_line_headers": 1,
        "spaced_mada_headers": 0,
    },
}


@pytest.mark.unit
def test_pdf_matching_the_baselines_passes_every_gate(
    tracked_pdf, tmp_path, monkeypatch
):
    for name, value in SYNTHETIC_BASELINES.items():
        monkeypatch.setattr(profile, name, value)
    out = tmp_path / "source_profile.json"

    metrics, gates = run_profile(tracked_pdf, out)

    assert {g.id: g.outcome for g in gates} == {f"G{i}": "pass" for i in range(1, 7)}
    assert metrics["rows_missing_a_side"] == 0
    assert json.loads(out.read_text("utf-8")) == metrics


@pytest.mark.unit
def test_other_pdf_fails_the_gates_and_still_writes_metrics_and_evidence(
    synthetic_pdf, tmp_path
):
    out, evidence = tmp_path / "source_profile.json", tmp_path / "evidence"

    with pytest.raises(ProfileGateError, match="G1, G2, G4, G5, G6"):
        run_profile(synthetic_pdf, out, evidence_dir=evidence)

    gates = json.loads(out.read_text("utf-8"))["gates"]
    assert gates["G1"] == "fail"  # it has no .dvc file at all
    assert gates["G3"] == "pass"  # every page is still one two-column table
    assert (evidence / "summary.json").exists()


@pytest.mark.unit
def test_evidence_comes_from_the_same_measuring_pass(synthetic_pdf, tmp_path):
    evidence = tmp_path / "evidence"
    with pytest.raises(ProfileGateError):
        run_profile(synthetic_pdf, tmp_path / "metrics.json", evidence_dir=evidence)

    summary = json.loads((evidence / "summary.json").read_text("utf-8"))
    assert summary["P4_rows"]["detected_lines_strict"] == 7
    assert summary["P5_continuation_first_rows"]["pages"] == 1  # page 2
    assert summary["P12_repeal_rows"] == [[2, 1, "3", "5"]]
    assert summary["P18_keyword_only_heading_as_last_row"] == [[1, "SECTION II"]]
    pages_tsv = (evidence / "pages.tsv").read_text("utf-8").splitlines()
    assert [line.split("\t")[0] for line in pages_tsv[1:]] == ["1", "2"]
    assert "page 1, row 3" in (evidence / "extractor_comparison.txt").read_text("utf-8")


@pytest.mark.unit
def test_numeric_metrics_turn_gates_into_ones_and_zeros():
    metrics = {
        "pages_total": 2,
        "font_set": ["x"],
        "gates": {"G1": "pass", "G2": "fail"},
    }

    assert profile.numeric_metrics(metrics) == {
        "pages_total": 2,
        "gate_G1": 1.0,
        "gate_G2": 0.0,
    }


# --- Profile gate: the full PDF (G1 to G6) ------------------------------------


@pytest.mark.profile
def test_every_gate_passes(source_profile):
    assert source_profile["gates"] == {f"G{i}": "pass" for i in range(1, 7)}


@pytest.mark.profile
def test_g1_hash_matches_the_dvc_file(source_profile, full_pdf):
    assert source_profile["pdf_md5"] == profile.dvc_md5(
        full_pdf.with_name(full_pdf.name + ".dvc")
    )


@pytest.mark.profile
def test_g2_to_g5_layout_baselines(source_profile):
    assert source_profile["pages_total"] == PAGES
    assert source_profile["pages_with_text_layer"] == PAGES
    assert source_profile["pages_one_2col_table"] == PAGES
    assert source_profile["font_set"] == FONTS
    assert source_profile["rows_detected"] == ROWS_DETECTED
    assert source_profile["rows_tagged"] == ROWS_TAGGED
    assert source_profile["pages_detected_ne_tagged"] == [[1, 9, 10]]
    assert source_profile["rows_missing_a_side"] == 0


@pytest.mark.profile
def test_g6_raw_defect_baselines(source_profile):
    assert {k: source_profile[f"raw_{k}"] for k in RAW_DEFECTS} == RAW_DEFECTS


@pytest.mark.profile
def test_analysis_evidence_is_unchanged(source_profile, repo_root):
    """`profile` rewrites the report's evidence byte for byte (Gate 1)."""
    checked = subprocess.run(
        ["sha256sum", "-c", "docs/reports/0_source_pdf_analysis.sha256"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
