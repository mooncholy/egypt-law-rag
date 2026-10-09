"""Find the articles a text names by number (``المادة ٢٢٢``, ``Article 147``).

- ``article_references``: the articles a *question* names. Embeddings are poor
  at exact numbers, so the retriever fetches them directly and ranks them first.
  A plural (``Articles 221 and 222``) is left to search, since a question that
  names a list asks about a topic.
- ``cited_articles``: the articles an *article's text* cites, plural and dual
  forms included (``المادتين ٢٢١ ، ٢٢٢``), for ``search.cite_expansion``.

Only a number right after an article keyword counts: ``7 percent`` or a year is
never a reference.
"""

import re

from raglaw.text.normalize import fold_arabic

# Matched on fold_arabic text, so مادة / المادة arrive as ماده / الماده and
# Arabic-Indic digits as ASCII. An optional و or ف, then ال, بال or لل.
ARABIC_REFERENCE = re.compile(r"(?<!\w)[وف]?(?:ال|بال|لل)?ماده\s*\(?\s*(\d+)")
ENGLISH_REFERENCE = re.compile(r"\b(?:article|art\.?)\s*\(?\s*(\d+)", re.IGNORECASE)


def article_references(question: str) -> list[int]:
    """
    The article numbers a question names, in order of first appearance.

    returns:
    - numbers (list[int]): each referenced article once; empty when the
      question names none
    """
    text = fold_arabic(question)
    found = [
        (m.start(), int(m.group(1)))
        for pattern in (ARABIC_REFERENCE, ENGLISH_REFERENCE)
        for m in pattern.finditer(text)
    ]
    return list(dict.fromkeys(number for _, number in sorted(found)))


# An article's own citations, beyond the singular: `Articles 221 and 222`,
# `Articles 466 to 468` (its ends only), the Arabic dual `المادتين ٢٢١ ، ٢٢٢` and
# the Arabic range below.
NUMBER_LIST = r"(\d+(?:\s*(?:,|،|and|or|to|و|-|–)\s*\d+)*)"
ENGLISH_PLURAL = re.compile(rf"\barticles\s+{NUMBER_LIST}", re.IGNORECASE)
ARABIC_DUAL = re.compile(rf"(?<!\w)[وف]?(?:ال|بال|لل)?مادت(?:ين|ان)\s*{NUMBER_LIST}")
# `المواد من ١٠٤٠ إلى ١٠٤٢` (إلى folds to الي). A number must follow `من`, since
# `مواد من عنده` means materials.
ARABIC_RANGE = re.compile(
    r"(?<!\w)[وف]?(?:ال|بال|لل)?مواد\s*من\s*(\d+)(?:\s*(?:الي|حتي)\s*(\d+))?"
)


def cited_articles(text: str) -> list[int]:
    """
    The articles a passage of the Code cites by number, in order of first appearance.

    A range (``Articles 466 to 468``) contributes its two ends, never the
    articles between, so a wide range can't flood the results.

    returns:
    - numbers (list[int]): each cited article once
    """
    folded = fold_arabic(text)
    found = [
        (m.start(1) + n.start(), int(n.group()))
        for pattern in (ENGLISH_PLURAL, ARABIC_DUAL)
        for m in pattern.finditer(folded)
        for n in re.finditer(r"\d+", m.group(1))
    ]
    found += [
        (m.start(1), int(m.group(1)))
        for pattern in (ARABIC_REFERENCE, ENGLISH_REFERENCE)
        for m in pattern.finditer(folded)
    ]
    found += [
        (m.start(i), int(m.group(i)))
        for m in ARABIC_RANGE.finditer(folded)
        for i in (1, 2)
        if m.group(i)
    ]
    return list(dict.fromkeys(n for _, n in sorted(found)))
