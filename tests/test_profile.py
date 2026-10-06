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

# --- Smoke: excerpts (S1) ---------------------------------------------------


@pytest.mark.smoke
def test_excerpt_passes_the_per_page_gates(excerpt, tmp_path):
    pdf, first_page = excerpt
    out = tmp_path / "source_profile.json"

    metrics, gates = run_profile(pdf, out, first_page=first_page)

    assert {g.id: g.outcome for g in gates} == {
        "G1": "skip",
        "G2": "pass",
        "G3": "pass",
        "G4": "pass",
        "G5": "skip",
        "G6": "skip",
    }
    assert metrics["rows_missing_a_side"] == 0
    assert json.loads(out.read_text("utf-8")) == metrics
    assert list(tmp_path.iterdir()) == [out]  # no evidence unless asked for


@pytest.mark.smoke
def test_failed_gate_stops_the_stage_and_still_writes_metrics(
    fixture_pdfs, tmp_path, monkeypatch
):
    monkeypatch.setattr(profile, "FONTS", ["Times-Roman"])
    out = tmp_path / "source_profile.json"

    with pytest.raises(ProfileGateError, match="G4"):
        run_profile(fixture_pdfs["page_081"], out, first_page=81)

    assert json.loads(out.read_text("utf-8"))["gates"]["G4"] == "fail"


@pytest.mark.smoke
def test_evidence_comes_from_the_same_pass_with_source_page_numbers(
    fixture_pdfs, tmp_path
):
    """Pages 46-47 hold `Section II`, the P25 sample, ending page 46 (P19)."""
    evidence = tmp_path / "evidence"
    run_profile(
        fixture_pdfs["pages_046_047"],
        tmp_path / "source_profile.json",
        first_page=46,
        evidence_dir=evidence,
    )

    summary = json.loads((evidence / "summary.json").read_text("utf-8"))
    assert summary["P1_file"]["page_count"] == 2
    assert summary["P18_keyword_only_heading_as_last_row"][0][0] == 46
    pages_tsv = (evidence / "pages.tsv").read_text("utf-8").splitlines()
    assert [line.split("\t")[0] for line in pages_tsv[1:]] == ["46", "47"]
    assert "page 46" in (evidence / "extractor_comparison.txt").read_text("utf-8")


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
