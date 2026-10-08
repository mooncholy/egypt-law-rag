"""Find the articles a question names by number (``المادة ٢٢٢``, ``Article 147``).

Embeddings are poor at exact numbers, so the retriever fetches a named article
directly and ranks it first. Only a number right after an article keyword
counts: ``7 percent`` or a year is never a reference. A plural (``Articles 221
and 222``, ``المواد``) is left to search, since it names a range or a list.
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
