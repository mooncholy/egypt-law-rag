"""What is embedded and BM25-indexed for each chunk (D14).

The default carries both languages and both heading paths, so a question in
either language, or one that names a topic only its heading states (e.g.,
``Leases``), can match. The variants exist to be measured against it.

An empty heading label (a heading printed in one language only, R28) is left
out: the placeholder the answer prompt uses would only add noise to a search.
"""

from raglaw.schema import Chunk

HEADING_SEPARATOR = " > "


def _headings(path: list[str]) -> str:
    return HEADING_SEPARATOR.join(label for label in path if label)


def document_text(chunk: Chunk, variant: str) -> str:
    """
    Build the searchable text of one chunk.

    returns:
    - text (str): the variant's parts, one per line, empty parts left out:
      ``both_with_headings`` is Arabic heading path, ``text_ar``, English
      heading path, ``text_en``; ``ar_only`` is the first two;
      ``both_without_headings`` is ``text_ar`` and ``text_en``

    exceptions:
    - ValueError: ``variant`` is not one of the three
    """
    ar = [_headings(chunk.heading_path_ar), chunk.text_ar]
    en = [_headings(chunk.heading_path), chunk.text_en]
    match variant:
        case "both_with_headings":
            parts = ar + en
        case "ar_only":
            parts = ar
        case "both_without_headings":
            parts = [chunk.text_ar, chunk.text_en]
        case _:
            raise ValueError(f"Unknown document text variant {variant!r}")
    return "\n".join(part for part in parts if part)
