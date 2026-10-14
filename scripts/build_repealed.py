"""Build ``data/repealed/repealed.json``: the repealed ranges' former text and repeal records.

The former text is the Internet Archive's OCR of the 1948 Code
(https://archive.org/details/alqanon-almadany-algaded-1948, file
``alqanon-almadany-algaded-1948_djvu.txt``). The OCR drops letters
(``الجعية`` for ``الجمعية``), so it is used only to match questions, never to
answer them. Each range is cut at its section headings, found by their OCR
forms below; each section is headed by its English heading (from the
Andersen translation of Law 131 of 1948, which prints the ranges as repealed)
and its Arabic heading (the 1948 heading, spelled out where the OCR dropped
letters).

The repeal records come from the current Code's repeal note where it gives the
law and the date (Articles 54-80), else from the repealing law (Articles
389-417, which the Code marks only ``ملغاة``). They are the owner's to confirm.

Run as ``uv run python scripts/build_repealed.py <djvu.txt> [--out PATH]``, then
``uv run dvc add data/repealed/repealed.json``. It sits outside ``dvc repro``:
its input is a download, not a tracked file.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from raglaw.config import Settings
from raglaw.retrieval.repealed import RepealedSet

logger = logging.getLogger("build_repealed")

SOURCE = (
    "Internet Archive, alqanon-almadany-algaded-1948 (the 1948 Civil Code, Law "
    "131 of 1948), OCR text; noisy, for matching only"
)

# (range's first article, English heading, Arabic heading, the OCR line that
# opens the section). A heading of None marks where the last section ends.
SECTIONS = (
    (54, "Legal Persons: Associations", "الأشخاص الاعتبارية: الجمعيات", r"^امعيات :"),
    (
        54,
        "Legal Persons: Foundations (Institutions)",
        "الأشخاص الاعتبارية: المؤسسات",
        r"^المؤسسات \.\s*$",
    ),
    (
        54,
        "Common Provisions for Associations and Foundations",
        "أحكام مشتركة بين الجمعيات والمؤسسات",
        r"^أحكام مشتركة بين",
    ),
    (54, None, None, r"^الفصل الثالك"),
    (389, "Proof of Obligations", "إثبات الالتزام", r"^إثنات الاليزام"),
    (
        389,
        "Proof of Obligations: Proof by Writing",
        "إثبات الالتزام: الإثبات بالكتابة",
        r"^الإثبات بالسكتاية",
    ),
    (
        389,
        "Proof of Obligations: Proof by Testimony (Oral Evidence)",
        "إثبات الالتزام: الإثبات بالبينة",
        r"^الإثيات بالبيئة",
    ),
    (
        389,
        "Proof of Obligations: Presumptions",
        "إثبات الالتزام: القرائن",
        r"^القرائ\s*$",
    ),
    (
        389,
        "Proof of Obligations: Admission (Acknowledgment)",
        "إثبات الالتزام: الإقرار",
        r"^الفصل الرابع",
    ),
    (389, "Proof of Obligations: Oath", "إثبات الالتزام: اليمين", r"^الهيبين"),
    (389, None, None, r"^العقود المسماة"),
)

RANGES = (
    {
        "first": 54,
        "last": 80,
        "subject_en": "associations and foundations",
        "subject_ar": "الجمعيات والمؤسسات",
        "repealed_on": "1956-11-03",
        "repealed_by_en": "by Presidential Decree-Law No. 348 of 1956",
        "repealed_by_ar": "بالقرار الجمهوري بالقانون رقم ٣٤٨ لسنة ١٩٥٦",
        "source": (
            "The Code's repeal note for Articles 54-80 (Official Gazette no. 88 "
            "bis (c), 3/11/1956; that law was itself repealed by Presidential "
            "Decree No. 32 of 1964). The note prints the law's number as 348; "
            "other sources give 384"
        ),
    },
    {
        "first": 389,
        "last": 417,
        "subject_en": "the proof of obligations",
        "subject_ar": "إثبات الالتزام",
        "repealed_on": "1968-11-30",
        "repealed_by_en": (
            "by the Law of Evidence in Civil and Commercial Matters (Law No. 25 of 1968)"
        ),
        "repealed_by_ar": "بقانون الإثبات في المواد المدنية والتجارية رقم ٢٥ لسنة ١٩٦٨",
        "source": (
            "The Code marks Articles 389-417 only as repealed. Law No. 25 of 1968, "
            "Article 1, repealed Chapter VI of Book I of Part I of the Civil Code; "
            "published in the Official Gazette no. 22 on 30 May 1968 and in force "
            "six months later"
        ),
    },
)


def sections(lines: list[str]) -> list[dict]:
    """
    Cut the OCR text into the sections of ``SECTIONS``, in order.

    exceptions:
    - StopIteration: a section's opening line isn't in the text
    """
    starts, at = [], 0
    for *_, pattern in SECTIONS:
        at = next(i for i in range(at, len(lines)) if re.search(pattern, lines[i]))
        starts.append(at)
    out = []
    for k, (first, en, ar, _) in enumerate(SECTIONS):
        if en is None:
            continue
        body = [ln.strip() for ln in lines[starts[k] + 1 : starts[k + 1]]]
        text = "\n".join(ln for ln in body if ln)
        out.append({"first": first, "heading_en": en, "heading_ar": ar, "text": text})
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ocr", type=Path, help="the 1948 Code's OCR text (djvu.txt)")
    ap.add_argument("--out", type=Path, default=Path(settings.paths.repealed))
    args = ap.parse_args()
    lines = args.ocr.read_text(encoding="utf-8").split("\n")
    data = RepealedSet.model_validate(
        {"source": SOURCE, "ranges": list(RANGES), "sections": sections(lines)}
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=1) + "\n",
        "utf-8",
    )
    for s in data.sections:
        logger.info("%d: %s (%d characters)", s.first, s.heading_en, len(s.text))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
