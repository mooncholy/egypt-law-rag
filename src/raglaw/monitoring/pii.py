"""The PII guardrail (Phase 14): redact personal data from every /ask answer.

The Code's articles hold no personal data, so an answer can only carry it by
echoing the question. Four recognizers, in the style of Presidio's pattern
recognizers (Presidio itself needs a spaCy NER model, and none reads Arabic
names):

- ``national_id``: an Egyptian national ID, 14 digits whose structure is
  checked (century 2 or 3, a real birth date, a governorate code), so an
  arbitrary 14-digit number isn't redacted;
- ``phone``: an Egyptian mobile number (``01[0125]`` and 8 digits, with or
  without ``+20``/``0020``, spaces or dashes);
- ``email``;
- ``name``: up to three Arabic words after an honorific (``السيد``, ``المواطنة``,
  ...), the first without the article ``ال`` (a given name never has it, a
  role does: ``المهندس المعماري``), which is how a name appears in a question
  about one's own case.

Digits may be Western, Arabic-Indic (``٠``-``٩``) or Persian (``۰``-``۹``);
an honorific may carry a prefix (``للسيد``).
``StreamRedactor`` does the same on a streamed answer, holding back the last
words until they can't be part of a match.
"""

import re
import time
from dataclasses import dataclass
from datetime import date

_TO_ASCII = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9",
    "01234567890123456789",
)
# Governorates of birth, plus 88 (born abroad).
GOVERNORATES = {
    *(f"{n:02d}" for n in (1, 2, 3, 4)),
    *(str(n) for n in range(11, 20)),
    *(str(n) for n in range(21, 30)),
    *(str(n) for n in range(31, 36)),
    "88",
}
# Honorifics without their article, which may carry a prefix (للسيد, وبالمواطن).
HONORIFICS = (
    "سيد",
    "سيدة",
    "أستاذ",
    "أستاذة",
    "دكتور",
    "دكتورة",
    "مواطن",
    "مواطنة",
    "مدعو",
    "مدعوة",
    "مهندس",
    "مهندسة",
)
ARTICLE = "(?:[وف]?[بك]?ال|[وف]?لل)"
ARABIC_WORD = "[\u0621-\u064a]+"

# Matched on the text with its digits made ASCII (one character for one, so
# the positions hold).
NATIONAL_ID = re.compile(r"(?<!\d)\d{14}(?!\d)")
PHONE = re.compile(r"(?<!\d)(?:(?:\+|00)20[\s-]?)?0?1[0125](?:[\s-]?\d){8}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
NAME = re.compile(
    rf"(?<![\u0621-\u064a]){ARTICLE}(?:{'|'.join(HONORIFICS)})\s+"
    rf"((?!ال){ARABIC_WORD}(?:\s+{ARABIC_WORD}){{0,2}})"
)
TAGS = {
    "national_id": "[national_id]",
    "phone": "[phone]",
    "email": "[email]",
    "name": "[name]",
}
# The most words one match can span: a phone number written with ``+20`` and a
# separator before each of its 8 last digits is 10 words. A streamed answer
# holds that many back, so no match can be released half seen.
HOLD_WORDS = 10


@dataclass(frozen=True)
class Finding:
    """One piece of personal data: its kind and where it was in the text."""

    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class Redaction:
    """The text with each finding replaced by its tag."""

    text: str
    findings: list[Finding]

    @property
    def redacted(self) -> bool:
        return bool(self.findings)


def is_national_id(digits: str) -> bool:
    """Whether 14 digits have a national ID's structure."""
    d = digits.translate(_TO_ASCII)
    if len(d) != 14 or d[0] not in "23":
        return False
    year = (1900 if d[0] == "2" else 2000) + int(d[1:3])
    try:
        date(year, int(d[3:5]), int(d[5:7]))
    except ValueError:
        return False
    return d[7:9] in GOVERNORATES


def find_pii(text: str) -> list[Finding]:
    """Every finding in ``text``, in order, none overlapping."""
    ascii_digits = text.translate(_TO_ASCII)
    found = [
        Finding("national_id", m.start(), m.end())
        for m in NATIONAL_ID.finditer(ascii_digits)
        if is_national_id(m.group())
    ]
    found += [
        Finding("phone", m.start(), m.end()) for m in PHONE.finditer(ascii_digits)
    ]
    found += [Finding("email", m.start(), m.end()) for m in EMAIL.finditer(text)]
    found += [Finding("name", m.start(1), m.end(1)) for m in NAME.finditer(text)]
    kept: list[Finding] = []
    for f in sorted(found, key=lambda f: (f.start, -f.end)):
        if not kept or f.start >= kept[-1].end:
            kept.append(f)
    return kept


def redact(text: str) -> Redaction:
    """
    Replace every finding in ``text`` with its tag (``[phone]``, ...).

    returns:
    - redaction (Redaction): the redacted text, and what was found
    """
    findings = find_pii(text)
    parts, last = [], 0
    for f in findings:
        parts += [text[last : f.start], TAGS[f.kind]]
        last = f.end
    return Redaction("".join([*parts, text[last:]]), findings)


class StreamRedactor:
    """Redact a streamed answer: release text only once its last
    ``HOLD_WORDS`` words are past, so no match is cut in two. ``seconds`` adds
    up the time spent checking, for the guardrail's latency metric."""

    def __init__(self) -> None:
        self._pending = ""
        self.findings: list[Finding] = []
        self.seconds = 0.0

    def feed(self, piece: str) -> str:
        """Take the next piece; return the text that is safe to send now."""
        start = time.perf_counter()
        try:
            return self._feed(piece)
        finally:
            self.seconds += time.perf_counter() - start

    def _feed(self, piece: str) -> str:
        self._pending += piece
        cuts = [m.start() for m in re.finditer(r"\s+", self._pending)]
        if len(cuts) <= HOLD_WORDS:
            return ""
        cut = cuts[-HOLD_WORDS]
        for f in find_pii(self._pending):  # a match across the cut stays held
            if f.start < cut < f.end:
                cut = f.start
        ready, self._pending = self._pending[:cut], self._pending[cut:]
        return self._redact(ready)

    def flush(self) -> str:
        """Return whatever is still held back, redacted."""
        start = time.perf_counter()
        ready, self._pending = self._pending, ""
        try:
            return self._redact(ready)
        finally:
            self.seconds += time.perf_counter() - start

    def _redact(self, text: str) -> str:
        result = redact(text)
        self.findings += result.findings
        return result.text
