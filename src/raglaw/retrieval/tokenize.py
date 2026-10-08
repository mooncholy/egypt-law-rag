"""The BM25 tokenizer: one folding key for Arabic and English alike.

Text goes through ``fold_arabic`` (F1 to F5 in ``docs/normalization.md``), so a
question typed without hamza, diacritics or Arabic-Indic digits matches the
printed text. It is then lower-cased and split on anything that isn't a letter
or a digit. Indexing and queries call this same function, so they can't drift.

Light prefix stripping is a variant measured by retrieval evaluation, not the
default: it can merge words that differ only in their first letter. So is the
embedding model's own subword tokenizer (``bm25_tokenizer``), which doesn't fold
and splits words into pieces shared across unrelated words.
"""

import re
from collections.abc import Callable
from typing import Protocol

from raglaw.text.normalize import fold_arabic, normalize_arabic

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


class SubwordTokenizer(Protocol):
    """The part of a Hugging Face tokenizer the ``model_subwords`` choice uses."""

    def tokenize(self, text: str) -> list[str]: ...


def bm25_tokenizer(
    name: str, subword_tokenizer: SubwordTokenizer | None = None
) -> Callable[[str], list[str]]:
    """
    The tokenizer named by ``retrieval.bm25_tokenizer``, for indexing and queries.

    ``model_subwords`` tokenizes ``normalize_arabic`` text without folding, as
    the model's vocabulary was learned on text as printed.

    returns:
    - tokenize (Callable[[str], list[str]]): text to BM25 terms

    exceptions:
    - ValueError: an unknown name, or ``model_subwords`` without a tokenizer
    """
    match name:
        case "words":
            return fold_tokens
        case "words_light_stem":
            return lambda text: fold_tokens(text, strip_prefixes=True)
        case "model_subwords":
            if subword_tokenizer is None:
                raise ValueError("model_subwords needs the model's tokenizer")
            return lambda text: subword_tokenizer.tokenize(normalize_arabic(text))
        case _:
            raise ValueError(f"Unknown BM25 tokenizer {name!r}")
