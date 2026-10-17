"""Arabic normalization, shared by ingestion and query time (``docs/normalization.md``).

- ``normalize_arabic`` makes stored text consistent at the code-point level
  without changing a letter a reader would see (N1 to N5).
- ``fold_arabic`` builds a matching key for lexical search, conflating spelling
  variants on purpose (F1 to F5). Its output is never stored or shown.

Both are idempotent, and query-time code imports these same functions, so
ingestion and queries cannot drift apart.
"""

import re
import unicodedata

# N1: Arabic Presentation Forms-A and -B.
PRESENTATION_FORMS = re.compile("[\\ufb50-\\ufdff\\ufe70-\\ufeff]")
# N3 and N4: tatweel, zero-width and bidi control characters.
INVISIBLE = re.compile(
    "[\\u0640\\u200b-\\u200f\\u202a-\\u202e\\u2066-\\u2069\\u061c\\ufeff]"
)
# N5: every whitespace character except the line break.
SPACES = re.compile(r"[^\S\n]+")

# F1 to F5.
ALEF_VARIANTS = str.maketrans("أإآٱ", "ا" * 4)
DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
DIACRITICS = re.compile("[\\u064b-\\u0652\\u0670]")


def normalize_arabic(text: str) -> str:
    """
    Normalize stored Arabic text at the code-point level (N1 to N5).

    No letter changes: alef variants, ``ى``, ``ة``, diacritics and digits are
    kept as printed.

    returns:
    - text (str): the normalized text; lines stripped, empty lines dropped
    """
    text = INVISIBLE.sub("", text)  # N4 first: U+FEFF also sits in N1's block
    text = PRESENTATION_FORMS.sub(
        lambda m: unicodedata.normalize("NFKC", m.group()), text
    )  # N1
    text = unicodedata.normalize("NFC", text)  # N2
    text = INVISIBLE.sub("", text)  # N3, and anything N1 or N2 produced
    lines = (SPACES.sub(" ", line).strip() for line in text.split("\n"))  # N5
    return "\n".join(line for line in lines if line)


def fold_arabic(text: str) -> str:
    """
    Build the lexical matching key for Arabic text (F1 to F5).

    Never store or display the result: it changes letters on purpose.

    returns:
    - key (str): ``normalize_arabic`` text with alef variants unified, ``ى`` as
      ``ي``, ``ة`` as ``ه``, diacritics removed and digits in ASCII
    """
    text = normalize_arabic(text)
    text = text.translate(ALEF_VARIANTS)  # F1
    text = text.replace("ى", "ي").replace("ة", "ه")  # F2, F3
    text = DIACRITICS.sub("", text)  # F4
    return text.translate(DIGITS)  # F5
