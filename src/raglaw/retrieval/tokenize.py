"""The BM25 tokenizer: one folding key for Arabic and English alike.

Text goes through ``fold_arabic`` (F1 to F5 in ``docs/normalization.md``), so a
question typed without hamza, diacritics or Arabic-Indic digits matches the
printed text. It is then lower-cased and split on anything that isn't a letter
or a digit. Indexing and queries call this same function, so they can't drift.

Light prefix stripping is a variant measured by retrieval evaluation, not the
default: it can merge words that differ only in their first letter.
"""

import re

from raglaw.text.normalize import fold_arabic

# Letters and digits in any script; underscores are not word characters here.
TOKEN = re.compile(r"[^\W_]+")
# Light stripping, a subset of Larkey et al.'s light10: the conjunctions و and ف,
# then the definite article, alone or after ب. A bare ب is never stripped, since
# it starts many roots (بيع, بطلان).
CONJUNCTIONS = "وف"
ARTICLES = ("بال", "ال")
MIN_LETTERS_BEFORE_CONJUNCTION = 4  # e.g. وعقد; وقف keeps its و
MIN_STEM_LETTERS = 2


def _strip_prefixes(token: str) -> str:
    if len(token) >= MIN_LETTERS_BEFORE_CONJUNCTION and token[0] in CONJUNCTIONS:
        token = token[1:]
    for article in ARTICLES:
        if token.startswith(article) and len(token) - len(article) >= MIN_STEM_LETTERS:
            return token[len(article) :]
    return token


def fold_tokens(text: str, *, strip_prefixes: bool = False) -> list[str]:
    """
    Split text into folded search tokens.

    returns:
    - tokens (list[str]): ``fold_arabic`` text, lower-cased, split on non-word
      characters; with ``strip_prefixes``, Arabic tokens lose a leading و or ف
      and a definite article (``ال``, ``بال``)
    """
    tokens = TOKEN.findall(fold_arabic(text).lower())
    if strip_prefixes:
        tokens = [_strip_prefixes(t) for t in tokens]
    return tokens
