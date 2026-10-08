"""LangChain's entry point to the corpus: one ``Document`` per article.

Parsing stays on PyMuPDF (R1 to R4): LangChain's PDF loaders read page text,
which would lose the table rows that carry the structure. LangChain starts
here, from the validated ``articles.json``. The ``chunk`` stage uses this
loader now, and retrieval will reuse it.
"""

from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from raglaw.records import read_records
from raglaw.schema import Article


class CivilCodeArticleLoader(BaseLoader):
    """Load ``articles.json`` as one LangChain ``Document`` per article."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def lazy_load(self) -> Iterator[Document]:
        """
        Yield each article as a ``Document``.

        returns:
        - documents (Iterator[Document]): ``page_content`` is the Arabic text;
          the metadata holds the number, the English text, both heading paths,
          the source pages, the repeal flag and the untranslated passages
        """
        for article in read_records(self.path, Article):
            yield Document(
                id=f"art-{article.article_number}",
                page_content=article.text_ar,
                metadata={
                    "article_number": article.article_number,
                    "text_en": article.text_en,
                    "heading_path": article.heading_path,
                    "heading_path_ar": article.heading_path_ar,
                    "source_pages": article.source_pages,
                    "is_repealed": article.is_repealed,
                    "only_in_en": article.only_in_en,
                    "only_in_ar": article.only_in_ar,
                },
            )
