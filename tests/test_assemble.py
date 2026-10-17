import pytest

from raglaw.ingest.assemble import (
    NUMBERED_RANK,
    PLAIN_RANK,
    RowClass,
    _ar_unit,
    _en_unit,
    align_units,
    assemble_rows,
    classify_row,
    heading_rank,
    heading_tree,
    heading_units,
)
from raglaw.schema import LogEvent, Row

ROOT = ("Preliminary Chapter", "باب تمهيدي / أحكام عامة")


def row(page: int, index: int, en: str, ar: str = "", bold: bool | None = False) -> Row:
    """A repaired row; ``bold`` applies to both sides, None for an empty side."""
    return Row(
        page=page,
        row_index=index,
        en_text=en,
        ar_text=ar,
        en_all_bold=bold if en else None,
        ar_all_bold=bold if ar else None,
    )


def article(page: int, index: int, n: int, ar_n: str, body: str = "Body.") -> Row:
    return row(page, index, f"Article {n}\n{body}", f"مادة {ar_n}\nنص.")


def heading(page: int, index: int, en: str, ar: str) -> Row:
    return row(page, index, en, ar, bold=True)


def paths(rows: list[Row]) -> dict[int, list[str]]:
    result = assemble_rows(rows, *ROOT)
    return {a.article_number: a.heading_path for a in result.articles}


# --- Classification (U3, U4) ------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("r", "kind"),
    [
        (article(5, 1, 12, "١٢"), RowClass.ARTICLE),
        (heading(5, 2, "SECTION I\nPersons", "الفصل الأول\nالأشخاص"), RowClass.HEADING),
        (row(5, 0, "of the contract.", "العقد."), RowClass.CONTINUATION),
        (row(5, 3, "stray text", "نص"), RowClass.ANOMALY),
        (row(53, 3, "Articles 389-417 repealed", "ملغاة", bold=True), RowClass.REPEAL),
        (row(81, 5, "", "موت المستأجر أو إعساره", bold=True), RowClass.HEADING),
    ],
    ids=[
        "article",
        "heading",
        "continuation",
        "anomaly",
        "repeal",
        "arabic-only heading",
    ],
)
def test_each_row_class(r, kind):
    """U3, plus the repeal row (R19) and an Arabic-only heading (P29)."""
    assert classify_row(r) is kind


@pytest.mark.unit
def test_an_inline_citation_never_starts_an_article():
    """U4: only a header-form first line does."""
    cited = row(5, 2, "paragraph 2 of Article 717.\nmore", "المادة ٧١٧")
    body = article(5, 1, 194, "١٩٤", body="as in paragraph 2 of Article 717.")

    assert classify_row(cited) is RowClass.ANOMALY
    assert [a.article_number for a in assemble_rows([body], *ROOT).articles] == [194]


# --- Headings (U8 to U12) ---------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("en", "ar", "rank"),
    [
        ("PART I", "القسم الأول", 1),
        ("BOOK II", "الكتاب الثاني", 2),
        ("Chapter III", "الباب الثالث", 3),
        ("SECTION IV", "الفصل الرابع", 4),
        ("Section IV", "الفصل الرابع", 4),
        ("1. Laws and Rights", "١ -القانون والحق", NUMBERED_RANK),
        ("2- Juristic persons", "٢ – الشخص الاعتباري", NUMBERED_RANK),
        ("Consent:", "الرضاء", PLAIN_RANK),
    ],
)
def test_heading_rank_by_keyword_then_numbering(en, ar, rank):
    """U8: keywords match case-insensitively; `1.`, `2-`, `١ -`, `١ –` number."""
    assert heading_rank(_en_unit(en), _ar_unit(ar))[0] == rank


@pytest.mark.unit
def test_a_keyword_line_takes_the_next_line_as_its_title():
    """U9: `SECTION I` + `Laws and their Applications` is one heading."""
    units = heading_units(["SECTION I", "Laws and their Applications"], _en_unit)

    assert [u.text for u in units] == ["SECTION I Laws and their Applications"]


@pytest.mark.unit
def test_a_numbered_line_and_the_next_line_are_two_headings():
    """U9: `1. Elements of Contracts` + `Consent:` is two, in both languages."""
    pairs, aligned = align_units(
        heading_units(["1. Elements of Contracts", "Consent:"], _en_unit),
        heading_units(["أولاً- أركان العقد", "١ -الرضاء"], _ar_unit),
    )

    assert aligned
    assert [(e.text, a.text) for e, a in pairs] == [
        ("1. Elements of Contracts", "أولاً- أركان العقد"),
        ("Consent:", "١ -الرضاء"),
    ]


@pytest.mark.unit
def test_a_wrapped_title_is_joined_when_the_other_language_has_one_heading():
    pairs, aligned = align_units(
        heading_units(
            ["1- The Relationship between the Surety and the", "Creditor"], _en_unit
        ),
        heading_units(["١ -العلاقة ما بين الكفيل والدائن"], _ar_unit),
    )

    assert aligned
    assert [e.text for e, _ in pairs] == [
        "1- The Relationship between the Surety and the Creditor"
    ]


@pytest.mark.unit
def test_heading_paths_follow_the_stack():
    """U10: `Object:` replaces `Consent:`, and `2.` pops both."""
    rows = [
        heading(
            8, 10, "1. Elements of Contracts\nConsent:", "أولاً- أركان العقد\n١ -الرضاء"
        ),
        article(8, 11, 89, "٨٩"),
        heading(14, 4, "Object:", "المحل"),
        article(14, 5, 131, "١٣١"),
        heading(16, 4, "2. The Effects of a Contract", "٢ - آثار العقد"),
        article(16, 5, 145, "١٤٥"),
    ]

    p = paths(rows)
    assert p[89] == [ROOT[0], "1. Elements of Contracts", "Consent:"]
    assert p[131] == [ROOT[0], "1. Elements of Contracts", "Object:"]
    assert p[145] == [ROOT[0], "2. The Effects of a Contract"]


@pytest.mark.unit
def test_a_keyword_heading_ending_a_page_takes_its_title_from_the_next():
    """U11: `Section II` ends page 46; its title opens page 47 (P19)."""
    rows = [
        article(46, 6, 349, "٣٤٩"),
        heading(46, 7, "Section II", "الفصل الثاني"),
        heading(47, 0, "Methods of Extinction", "انقضاء الالتزام"),
        heading(47, 1, "1. Giving in Payment", "١ -الوفاء بمقابل"),
        article(47, 2, 350, "٣٥٠"),
    ]

    result = assemble_rows(rows, *ROOT)
    assert [h.en for h in result.headings] == [
        ROOT[0],
        "Section II Methods of Extinction",
        "1. Giving in Payment",
    ]
    assert result.headings[1].ar == "الفصل الثاني انقضاء الالتزام"


@pytest.mark.unit
def test_numbered_in_either_language_ranks_as_numbered(log_records):
    """U12: `Associations` / `٣ -الجمعيات` is rank 5, with one numbering_mismatch."""
    rows = [
        heading(6, 10, "2. Juristic persons", "٢ -الشخص الاعتباري"),
        heading(7, 2, "Associations", "٣ -الجمعيات"),
        article(7, 3, 54, "٥٤"),
    ]

    result = assemble_rows(rows, *ROOT)
    assert [h.rank for h in result.headings[1:]] == [NUMBERED_RANK, NUMBERED_RANK]
    assert result.articles[0].heading_path == [ROOT[0], "Associations"]
    mismatches = log_records(event_type=LogEvent.ANOMALY)
    assert [m["anomaly_type"] for m in mismatches] == ["numbering_mismatch"]


@pytest.mark.unit
def test_a_later_heading_in_one_cell_nests_under_the_earlier():
    """Two plain headings in one cell: the second is a child, not a replacement."""
    rows = [
        heading(
            138,
            4,
            "Protection of Possession\n(The three possessory actions)",
            "حماية الحيازة\n( دعاوى الحيازة الثلاث(",
        ),
        article(138, 5, 958, "٩٥٨"),
    ]

    assert paths(rows)[958] == [
        ROOT[0],
        "Protection of Possession",
        "(The three possessory actions)",
    ]


@pytest.mark.unit
def test_an_arabic_only_heading_enters_the_arabic_path():
    rows = [
        row(81, 5, "", "موت المستأجر أو إعساره", bold=True),
        article(81, 6, 601, "٦٠١"),
    ]

    [a] = assemble_rows(rows, *ROOT).articles
    assert a.heading_path == [ROOT[0], ""]
    assert a.heading_path_ar == [ROOT[1], "موت المستأجر أو إعساره"]


# --- Repeals (U13) ------------------------------------------------------------------


@pytest.mark.unit
def test_both_repeal_formats_expand_to_one_record_per_article():
    rows = [
        row(
            7,
            3,
            "Article 54\n* Articles 54-80 have been repealed by Presidential Decree.",
            "المواد من ٥٤ إلى ٨٠ ملغاة",
        ),
        row(
            53, 3, "Articles 389-417 repealed", "المواد من ٣٨٩ إلى ٤١٧ ملغاة", bold=True
        ),
    ]

    result = assemble_rows(rows, *ROOT)
    numbers = [a.article_number for a in result.articles]
    assert numbers == [*range(54, 81), *range(389, 418)]
    assert all(a.is_repealed for a in result.articles)
    assert result.articles[0].text_en.startswith("* Articles 54-80")


# --- Continuations (R13, U24) -------------------------------------------------------


@pytest.mark.unit
def test_a_continuation_merges_into_the_previous_pages_article():
    rows = [
        article(10, 8, 77, "٧٧", body="The debtor shall"),
        row(11, 0, "pay in full.", "يدفع كاملاً."),
        article(11, 1, 78, "٧٨"),
    ]

    result = assemble_rows(rows, *ROOT)
    first = result.articles[0]
    assert first.text_en == "The debtor shall\npay in full."
    assert first.source_pages == [10, 11]
    assert first.source_rows == [(10, 8), (11, 0)]
    assert result.counts["cross_page_merges"] == 1


@pytest.mark.unit
def test_a_continuation_with_nothing_before_it_is_an_anomaly(log_records):
    """U24: logged as orphan_continuation, not dropped silently, never merged."""
    rows = [
        row(46, 0, "remainder of an article", "باقي المادة"),
        article(46, 1, 345, "٣٤٥"),
    ]

    result = assemble_rows(rows, *ROOT)
    [anomaly] = log_records(event_type=LogEvent.ANOMALY)
    assert anomaly["anomaly_type"] == "orphan_continuation"
    assert result.articles[0].text_en == "Body."


# --- Records (R20) --------------------------------------------------------------------


@pytest.mark.unit
def test_articles_carry_their_bodies_and_the_root_path():
    [a] = assemble_rows([article(1, 2, 1, "١")], *ROOT).articles

    assert a.text_en == "Body."
    assert a.text_ar == "نص."
    assert a.heading_path == [ROOT[0]]
    assert a.heading_path_ar == [ROOT[1]]


@pytest.mark.unit
def test_an_arabic_number_that_differs_is_an_anomaly(log_records):
    assemble_rows([article(3, 1, 12, "٢١")], *ROOT)

    [anomaly] = log_records(event_type=LogEvent.ANOMALY)
    assert (anomaly["anomaly_type"], anomaly["ar_number"]) == ("number_mismatch", 21)


@pytest.mark.unit
def test_heading_tree_indents_by_rank():
    rows = [
        heading(1, 0, "SECTION I\nLaws", "الفصل الأول\nالقانون"),
        article(1, 1, 1, "١"),
    ]

    tree = heading_tree(assemble_rows(rows, *ROOT).headings).splitlines()
    assert tree[0].startswith("[1] Preliminary Chapter")
    assert tree[1].startswith(
        "      [4] SECTION I Laws | الفصل الأول القانون  (p1 r0; articles 1–1)"
    )


# --- Corpus: the full PDF -------------------------------------------------------------


@pytest.mark.corpus
def test_assembly_counts(stage_metrics):
    counts = stage_metrics("assemble")

    assert counts["articles"] == 1_149
    assert counts["repealed"] == 56
    rows = sum(
        counts[f"rows_{k}"] for k in ("article", "heading", "continuation", "repeal")
    )
    assert rows + counts.get("rows_anomaly", 0) == 1_463  # C5
    assert counts["cross_page_merges"] == counts["rows_continuation"]


# --- Untranslated text (P32, R28) ------------------------------------------------------


def highlighted(r: Row, en: list[str] = (), ar: list[str] = ()) -> Row:
    return r.model_copy(update={"en_highlighted": list(en), "ar_highlighted": list(ar)})


@pytest.mark.unit
def test_highlighted_lines_become_untranslated_passages():
    """Article 1021: Arabic paragraphs (٢)-(٣) with no English counterpart."""
    r = row(
        147,
        5,
        "Article 1021\nThe owner is not bound.",
        "مادة ١٠٢١\n(١) لا يلزم المالك.\n(٢) فإذا كان المالك هو المكلف\nعلى نفقته.",
    )
    r = highlighted(r, ar=["(٢) فإذا كان المالك هو المكلف", "على نفقته."])

    [a] = assemble_rows([r], *ROOT).articles

    assert a.only_in_ar == ["(٢) فإذا كان المالك هو المكلف\nعلى نفقته."]
    assert a.only_in_en == []


@pytest.mark.unit
def test_a_highlighted_header_line_is_not_a_passage():
    """Article 1022: its whole English is highlighted, the header line included."""
    r = row(147, 6, "Article 1022\nIn the absence of an agreement.", "")
    r = highlighted(r, en=["Article1022", "In the absence of an agreement."])

    [a] = assemble_rows([r], *ROOT).articles

    assert a.only_in_en == ["In the absence of an agreement."]


@pytest.mark.unit
def test_highlights_continue_across_a_page():
    rows = [
        highlighted(
            article(147, 6, 1022, "١٠٢٢", body="first part"), en=["first part"]
        ),
        highlighted(row(148, 0, "second part", ""), en=["second part"]),
    ]

    [a] = assemble_rows(rows, *ROOT).articles

    assert a.only_in_en == ["first part\nsecond part"]


@pytest.mark.unit
def test_a_highlight_not_found_in_its_article_is_an_anomaly(log_records):
    r = highlighted(article(8, 1, 84, "٨٤"), en=["text that is not there"])

    assemble_rows([r], *ROOT)

    [anomaly] = log_records(event_type=LogEvent.ANOMALY)
    assert anomaly["anomaly_type"] == "highlight_not_found"


@pytest.mark.unit
def test_a_heading_in_one_language_only_is_marked():
    rows = [
        highlighted(
            row(81, 5, "", "موت المستأجر أو إعساره", bold=True),
            ar=["موت المستأجر أو إعساره"],
        ),
        article(81, 6, 601, "٦٠١"),
    ]

    result = assemble_rows(rows, *ROOT)

    assert result.headings[1].only_in == "ar"
    assert "only in Arabic" in heading_tree(result.headings)
