"""What is embedded and BM25-indexed for each chunk (D14).

The default carries both languages and both heading paths, so a question in
either language, or one that names a topic only its heading states (e.g.,
``Leases``), can match. The variants exist to be measured against it.

An empty heading label (a heading printed in one language only, R28) is left
out: the placeholder the answer prompt uses would only add noise to a search.

A repealed article's text is its repeal note (R19). With
``retrieval.repealed_text: heading`` its search text keeps only the heading
paths and ``Article N repealed``, since the decree's wording would otherwise
outweigh the headings, the only text naming the subject. The stored record is
never changed (R10); this is a search key.
"""

from raglaw.schema import Chunk

HEADING_SEPARATOR = " > "


def _headings(path: list[str]) -> str:
    return HEADING_SEPARATOR.join(label for label in path if label)


def _repealed_marker(chunk: Chunk) -> tuple[str, str]:
    """``المادة ٥٤ ملغاة`` and ``Article 54 repealed``, from the chunk's citation."""
    en, ar = chunk.citation.split(" | ")
    return f"{ar} ملغاة", f"{en} repealed"


def document_text(chunk: Chunk, variant: str, repealed_text: str = "note") -> str:
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
    text_ar, text_en = chunk.text_ar, chunk.text_en
    if chunk.is_repealed and repealed_text == "heading":
        text_ar, text_en = _repealed_marker(chunk)
    ar = [_headings(chunk.heading_path_ar), text_ar]
    en = [_headings(chunk.heading_path), text_en]
    match variant:
        case "both_with_headings":
            parts = ar + en
        case "ar_only":
            parts = ar
        case "both_without_headings":
            parts = [text_ar, text_en]
        case _:
            raise ValueError(f"Unknown document text variant {variant!r}")
    return "\n".join(part for part in parts if part)
