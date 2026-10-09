"""The manifest written next to each half of the search index.

``data/index/`` holds two indexes over the same chunks, built by two stages so
that changing one never rebuilds the other: ``dense/`` (``raglaw.retrieval.dense``)
and ``bm25/`` (``raglaw.retrieval.lexical``). Each writes a ``manifest.json``
naming what it was built from. A retriever compares both with its config, and
with each other, before searching, so a query is never embedded or tokenized
differently from the documents, and the two halves always cover the same chunks.
"""

import json
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

MANIFEST = "manifest.json"


class IndexManifest(BaseModel):
    """What both halves of the index record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunks_sha256: str = Field(
        description="SHA-256 of the `chunks.json` the index was built from."
    )
    chunk_ids: list[str] = Field(
        description="Every indexed `chunk_id`, in build order. BM25 results are "
        "positions in this list."
    )
    document_text: str = Field(
        description="The `retrieval.document_text` variant indexed per chunk (D14)."
    )
    repealed_text: str = Field(
        default="note",
        description="The `retrieval.repealed_text` choice for repealed chunks. "
        "Defaults to `note`, which every index built before the choice existed holds.",
    )


def write_manifest(directory: Path, manifest: IndexManifest) -> None:
    """Write ``manifest`` as sorted, indented JSON, so a rebuild gives the same bytes."""
    (directory / MANIFEST).write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest[M: IndexManifest](directory: Path, model: type[M]) -> M:
    """
    Read the manifest an index half wrote.

    returns:
    - manifest (IndexManifest): validated as ``model``
    """
    return model.model_validate_json((directory / MANIFEST).read_text("utf-8"))


def fresh_dir(directory: Path) -> None:
    """Remove and recreate ``directory``, so a rebuild never mixes with an old build."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
