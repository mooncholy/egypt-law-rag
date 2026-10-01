"""Cut the committed test excerpts out of the source PDF.

CI has no access to the full PDF (D6), so the smoke tests run the pipeline on
these four excerpts, committed to git. Each covers cases the brief lists in
its section 4. Output is byte-reproducible: rerunning on the same PDF writes
identical files, so git shows a change only when the source or the page list
changes.

Usage (from the repo root, after `dvc pull`):
    uv run python scripts/make_fixtures.py
"""

import argparse
import hashlib
import logging
from pathlib import Path

import pymupdf

from raglaw.config import Settings
from raglaw.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Excerpt file stem -> first and last source page (1-based, inclusive).
EXCERPTS: dict[str, tuple[int, int]] = {
    "page_001": (1, 1),
    "pages_007_009": (7, 9),
    "pages_046_047": (46, 47),
    "page_081": (81, 81),
}
# The pre-commit large-file guard blocks anything above 1024 KB.
MAX_BYTES = 1024 * 1024


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut_excerpt(source: pymupdf.Document, first: int, last: int, out: Path) -> None:
    """
    Copy source pages ``first`` to ``last`` into a new PDF at ``out``.

    The copy loses the source's tag tree, so the fixture tests skip the
    checks that read it (G1, G5).
    """
    excerpt = pymupdf.open()
    excerpt.insert_pdf(source, from_page=first - 1, to_page=last - 1)
    # No metadata and no fresh file ID: the same pages always give the same bytes.
    excerpt.set_metadata({})
    excerpt.save(out, garbage=4, deflate=True, no_new_id=True)


def check_excerpt(
    source: pymupdf.Document, first: int, last: int, out: Path
) -> list[str]:
    """
    Compare a written excerpt with the source pages it should hold.

    returns:
    - problems (list[str]): a wrong page count, a page whose text differs from
      its source page, or a file too large to commit; empty when it's right
    """
    problems = []
    excerpt = pymupdf.open(out)
    expected = last - first + 1
    if excerpt.page_count != expected:
        problems.append(f"{excerpt.page_count} pages, expected {expected}")
    for offset in range(min(excerpt.page_count, expected)):
        if excerpt[offset].get_text() != source[first - 1 + offset].get_text():
            problems.append(
                f"page {offset + 1} differs from source page {first + offset}"
            )
    size = out.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"{size} bytes, over the {MAX_BYTES}-byte commit limit")
    return problems


def main() -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=settings.paths.raw_pdf)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args()
    setup_logging("make_fixtures", logs_dir=settings.paths.logs_dir)

    if not args.pdf.exists():
        logger.error("Source PDF %s is missing; run `dvc pull`", args.pdf)
        return 1
    source = pymupdf.open(args.pdf)
    logger.info(
        "Cutting %d excerpts from %s",
        len(EXCERPTS),
        args.pdf,
        extra={
            "source_sha256": sha256_bytes(args.pdf),
            "source_pages": source.page_count,
        },
    )
    args.out.mkdir(parents=True, exist_ok=True)

    failed = False
    for stem, (first, last) in EXCERPTS.items():
        out = args.out / f"{stem}.pdf"
        cut_excerpt(source, first, last, out)
        fields = {"fixture": stem, "first_page": first, "last_page": last}
        problems = check_excerpt(source, first, last, out)
        for problem in problems:
            logger.error("%s: %s", out, problem, extra=fields)
        failed = failed or bool(problems)
        logger.info(
            "Wrote %s (source pages %d-%d)",
            out,
            first,
            last,
            extra={**fields, "bytes": out.stat().st_size, "sha256": sha256_bytes(out)},
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
