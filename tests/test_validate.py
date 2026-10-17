import pytest

from raglaw.ingest import validate as v


@pytest.mark.unit
def test_c1_finds_a_gap_and_a_duplicate(make_article, monkeypatch):
    monkeypatch.setattr(v, "ARTICLE_COUNT", 4)
    check = v.check_numbers([make_article(n) for n in (1, 2, 2, 4)])

    assert not check.passed
    assert "duplicates [2], missing [3]" in check.detail


@pytest.mark.unit
def test_c2_flags_exactly_the_repealed_ranges(make_article, monkeypatch):
    monkeypatch.setattr(v, "REPEALED_RANGES", ((2, 3),))
    right = [make_article(1), make_article(2, is_repealed=True)]
    right.append(make_article(3, is_repealed=True))

    assert v.check_repealed(right).passed
    wrong = [make_article(1, is_repealed=True), make_article(2, is_repealed=True)]
    check = v.check_repealed(wrong + [make_article(3)])
    assert "unflagged [3], wrongly flagged [1]" in check.detail


@pytest.mark.unit
def test_c6_c7_c8_read_the_bodies(make_article):
    articles = [
        make_article(1, "(١) أولى\n(٢) ثانية"),
        make_article(2, "(١) أولى\n(٣) ثالثة"),  # skips (٢)
        make_article(3, text_en="Body.\nArticle 4"),  # a header line inside a body
        make_article(1022, ""),  # the print's empty Arabic cell (allowed)
        make_article(5, ""),
    ]

    c6, c7, c8 = v.check_bodies(articles)

    assert (c6.passed, c6.detail) == (False, "articles [3]")
    assert not c7.passed and "(2, [1, 3])" in c7.detail
    assert not c8.passed and "empty text_ar [5]" in c8.detail


@pytest.mark.unit
def test_c13_counts_presentation_forms_and_latin(make_article):
    clean = v.check_characters([make_article(1, "(١) نص عربي")])
    dirty = v.check_characters(
        [make_article(1, "نص ﻻ"), make_article(2, "نص Latin words here")]
    )

    assert clean.passed
    assert dirty.value["presentation_forms"] == 1
    assert dirty.value["latin_over_1pct"] == [2]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stored", "passed", "ar_cer"),
    [
        ("wxyz", True, 0.0),  # only the accepted span differs
        ("vxyz", False, 0.125),  # a change outside it still counts
    ],
)
def test_c14_puts_back_only_the_accepted_spans(
    make_article, monkeypatch, stored, passed, ar_cer
):
    monkeypatch.setattr(v, "CER_ACCEPTED", {2: ("accepted", (("yz", "yy"),))})
    articles = [make_article(1, "abcd"), make_article(2, stored)]
    gold = [
        {"article_number": 1, "text_en": "Body of article 1.", "text_ar": "abcd"},
        {"article_number": 2, "text_en": "Body of article 2.", "text_ar": "wxyy"},
    ]

    check = v.check_gold(articles, gold)

    assert check.passed is passed
    assert check.value["ar_cer"] == ar_cer
    assert check.value["ar_cer_with_accepted"] == (0.125 if passed else 0.25)
    assert check.value["boundary_recall"] == 1.0


@pytest.mark.unit
def test_c12_fails_without_article_1_instead_of_raising(make_article):
    check = v.check_root([make_article(2)], "Preliminary Chapter / General Provisions")

    assert (check.passed, check.detail) == (False, "Article 1 is missing")


@pytest.mark.unit
def test_edit_distance_counts_characters():
    assert v.edit_distance("فحواه.ا", "فحواها.") == 2
    assert v.edit_distance("", "abc") == 3


@pytest.mark.unit
def test_the_row_accounting_must_close():
    rows = {"rows_article": 1, "rows_heading": 222, "rows_continuation": 0}
    c4, c5, c9 = v.check_rows(rows | {"anomalies": 1})

    assert not c4.passed and not c5.passed
    assert c9.passed


@pytest.mark.corpus
def test_every_corpus_check_passes(stage_metrics):
    """Gate 4 on the built corpus: C1-C16 (C15 is the chunk check, Phase 5)."""
    checks = stage_metrics("corpus")["checks"]

    assert set(checks) == {f"C{n}" for n in range(1, 17)} - {"C15"}
    assert set(checks.values()) == {"pass"}
