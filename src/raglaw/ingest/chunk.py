"""The ``chunk`` stage: articles become retrievable chunks, never across an article.

Assembly already split the code along its structure: one article per record,
with its heading path (document-structure chunking). This stage only splits
articles longer than ``max_chars``, and only at their Arabic paragraph markers
(R22): a paragraph is never cut, and an oversize one is kept whole and logged.

Two strategies, chosen in ``params.yaml``:

- ``structural`` (default): LangChain's ``RecursiveCharacterTextSplitter`` with
  the paragraph-marker boundary as its only separator, packing consecutive
  paragraphs up to ``max_chars``.
- ``structural_semantic``: paragraphs are also split where neighbouring
  paragraphs drift apart in meaning. The threshold is a percentile of the
  distances over the whole corpus, since a per-article percentile always splits
  at the largest gap of a short article.

Every part carries the article's full English text (D4), its heading paths, and
its untranslated passages (R28).

Besides counts, the stage records the *quality* of a chunking, so two runs can be
compared in MLflow: chunk sizes, how small the parts of split articles are, and
how many parts open with a word tying them to the paragraph before (an exception
or a condition, e.g. ``ومع ذلك``), against how often paragraphs do in the corpus.
Each split article is listed, with its parts, in a ``split_articles.md``
artifact on the run.

Run as ``python -m raglaw.ingest.chunk``.
"""

import argparse
import json
import logging
import math
import re
import statistics
import tempfile
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from raglaw.config import Chunking, Embedding, Settings
from raglaw.ingest.loader import CivilCodeArticleLoader
from raglaw.logging_setup import log_anomaly
from raglaw.records import write_records
from raglaw.retrieval.embeddings import huggingface_embeddings
from raglaw.schema import Chunk, LogEvent
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)

# A paragraph marker opening a line: (١), )١(, (٢(, ( ٣( and (٢.) after the
# right-to-left repairs. Every marker in the corpus starts its line (P14).
MARKER = r"[()]\s*([١-٩][٠-٩]*)\s*[().]?\s*[()]"
# The same pattern without a capturing group: LangChain wraps its separator in
# a group of its own before calling re.split, so a second group would misalign.
PARAGRAPH_START = r"\n(?=[()]\s*[١-٩][٠-٩]*\s*[().]?\s*[()])"
# A marker opening a paragraph, as a prefix to strip before reading its words.
MARKER_PREFIX = re.compile(r"^[()\s]*[١-٩][٠-٩]*\s*[().]?\s*[()]\s*،?\s*")
# Opening words that tie a paragraph to the one before it: a condition, an
# exception or a permission qualifying the previous paragraph's rule.
CONNECTIVE = re.compile(
    r"^(ومع ذلك|ومع هذا|غير أن|إلا أن|على أن|ولكن|فإذا|وإذا|فإن|"
    r"وفي هذه الحالة|وفى هذه الحالة|ويجوز|ولا يجوز)"
)
SHORT_CHUNK_CHARS = 100  # a chunk this short carries little context on its own
AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
EN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


class ChunkError(RuntimeError):
    """The chunks lose text, or miss an article."""


def citation(number: int) -> str:
    """How an answer cites an article, in both languages."""
    return f"Article {number} | المادة {str(number).translate(AR_DIGITS)}"


def range_citation(first: int, last: int) -> str:
    """How an answer cites a repealed range, in both languages."""
    ar = [str(n).translate(AR_DIGITS) for n in (first, last)]
    return f"Articles {first}-{last} | المواد من {ar[0]} إلى {ar[1]}"


def merge_repealed_ranges(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """
    Replace each run of consecutive repealed articles sharing one note by one chunk.

    Assembly expands a repeal row into one record per article (R19), each
    holding the same note; ``chunking.repealed: per_range`` indexes the range
    once instead.

    returns:
    - chunks (list[Chunk]): in order, each range as ``art-{first}-{last}``
      with ``range_end``
    - merged (int): how many ranges were merged
    """
    runs: list[list[Chunk]] = []
    for c in chunks:
        prev = runs[-1] if runs else None
        if (
            prev is not None
            and c.is_repealed
            and prev[0].is_repealed
            and c.article_number == prev[-1].article_number + 1
            and (c.text_ar, c.text_en) == (prev[0].text_ar, prev[0].text_en)
        ):
            prev.append(c)
        else:
            runs.append([c])
    out = []
    for run in runs:
        if len(run) == 1:
            out.append(run[0])
            continue
        first, last = run[0].article_number, run[-1].article_number
        out.append(
            run[0].model_copy(
                update={
                    "chunk_id": f"art-{first}-{last}",
                    "citation": range_citation(first, last),
                    "source_pages": sorted({p for c in run for p in c.source_pages}),
                    "range_end": last,
                }
            )
        )
    return out, sum(len(run) > 1 for run in runs)


def covered_articles(chunk: Chunk) -> range:
    """The articles a chunk stands for: one, or a whole repealed range."""
    return range(chunk.article_number, (chunk.range_end or chunk.article_number) + 1)


def paragraph_numbers(text: str) -> list[int]:
    """The numbers of the paragraph markers that open lines of ``text``."""
    return [
        int(m.group(1).translate(EN_DIGITS))
        for m in re.finditer(rf"(?m)^{MARKER}", text)
    ]


def split_paragraphs(text: str) -> list[str]:
    """Split Arabic text before each line that opens with a paragraph marker."""
    return [p for p in re.split(PARAGRAPH_START, text) if p.strip()]


# Parts never repeat text: an article's parts rebuild it exactly, which the
# reconstruction check relies on.
CHUNK_OVERLAP = 0


def structural_splitter(max_chars: int) -> RecursiveCharacterTextSplitter:
    """
    LangChain's recursive splitter, allowed to split at paragraph markers only.

    With no smaller separator to fall back to, an oversize paragraph comes back
    whole (R22). Pieces keep their marker and join with nothing in between, so
    the parts rebuild the article exactly.

    returns:
    - splitter (RecursiveCharacterTextSplitter): packs paragraphs up to
      ``max_chars``
    """
    return RecursiveCharacterTextSplitter(
        separators=[PARAGRAPH_START],
        is_separator_regex=True,
        keep_separator="start",
        chunk_size=max_chars,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        strip_whitespace=True,
    )


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return round(1.0 - dot / norm, 6) if norm else 0.0


def _percentile(values: list[float], q: float) -> float:
    """The ``q``-th percentile, by linear interpolation (as numpy's default)."""
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    low, high = math.floor(k), math.ceil(k)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


class ParagraphSemanticSplitter(TextSplitter):
    """Split articles at paragraph boundaries where meaning shifts.

    LangChain's semantic-chunking method (embed the units, measure the cosine
    distance between neighbours, split above a threshold), with two changes for
    short legal articles: the units are paragraphs, so a split never cuts one
    (R22), and the threshold is a percentile over the whole corpus, set by
    ``fit``. Groups longer than ``max_chars`` are then packed by the structural
    splitter.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        breakpoint_percentile: float,
        max_chars: int,
        batch_size: int = 32,
    ) -> None:
        super().__init__(chunk_size=max_chars, chunk_overlap=CHUNK_OVERLAP)
        self.embeddings = embeddings
        self.breakpoint_percentile = breakpoint_percentile
        self.batch_size = batch_size
        self.packer = structural_splitter(max_chars)
        self.threshold: float | None = None
        self.breakpoints = 0
        self._distances: dict[str, list[float]] = {}

    def fit(self, texts: list[str]) -> float | None:
        """
        Embed every paragraph of every multi-paragraph text, and set the threshold.

        returns:
        - threshold (float | None): the corpus percentile of neighbour
          distances; None when no text has two paragraphs
        """
        multi = {t: split_paragraphs(t) for t in texts}
        multi = {t: ps for t, ps in multi.items() if len(ps) > 1}
        flat = [p for ps in multi.values() for p in ps]
        vectors: list[list[float]] = []
        for i in range(0, len(flat), self.batch_size):
            vectors += self.embeddings.embed_documents(flat[i : i + self.batch_size])
        position = 0
        for text, paragraphs in multi.items():
            own = vectors[position : position + len(paragraphs)]
            position += len(paragraphs)
            self._distances[text] = [_cosine_distance(a, b) for a, b in pairwise(own)]
        every = [d for ds in self._distances.values() for d in ds]
        self.threshold = (
            round(_percentile(every, self.breakpoint_percentile), 6) if every else None
        )
        return self.threshold

    def split_text(self, text: str) -> list[str]:
        """
        Split one article at its meaning shifts, then pack by ``max_chars``.

        returns:
        - parts (list[str]): the article's parts, in order
        """
        paragraphs = split_paragraphs(text)
        distances = self._distances.get(text)
        if len(paragraphs) < 2 or distances is None or self.threshold is None:
            return self.packer.split_text(text)
        groups, current = [], [paragraphs[0]]
        for paragraph, distance in zip(paragraphs[1:], distances, strict=True):
            if distance > self.threshold:
                groups.append(current)
                current = []
                self.breakpoints += 1
            current.append(paragraph)
        groups.append(current)
        parts: list[str] = []
        for group in groups:
            parts += self.packer.split_text("\n".join(group))
        return parts


def build_splitter(
    chunking: Chunking, documents: list[Document], embeddings: Embeddings | None = None
) -> TextSplitter:
    """
    The splitter for the configured strategy, fitted to the corpus if semantic.

    returns:
    - splitter (TextSplitter): ready to split each article's Arabic text

    exceptions:
    - ValueError: the semantic strategy was asked for without embeddings
    """
    if chunking.strategy == "structural":
        return structural_splitter(chunking.max_chars)
    if embeddings is None:
        raise ValueError("structural_semantic needs embeddings to split by meaning")
    splitter = ParagraphSemanticSplitter(
        embeddings,
        chunking.semantic.breakpoint_percentile,
        chunking.max_chars,
    )
    splitter.fit([d.page_content for d in documents if not d.metadata["is_repealed"]])
    return splitter


def chunk_document(
    document: Document, splitter: TextSplitter, strategy: str, max_chars: int
) -> tuple[list[Chunk], int]:
    """
    Split one article into chunks; a repealed article stays one chunk.

    returns:
    - chunks (list[Chunk]): the article's parts, in order
    - oversize (int): parts longer than ``max_chars``, each logged (R22)
    """
    meta = document.metadata
    number = meta["article_number"]
    text = document.page_content
    if meta["is_repealed"] or not text:
        parts = [text]
    else:
        parts = splitter.split_text(text)
    oversize = 0
    for part in parts:
        if len(part) > max_chars:
            oversize += 1
            log_anomaly(
                logger,
                "Paragraph longer than max_chars kept whole (R22)",
                page=meta["source_pages"][0],
                row_index=-1,
                row_class="article",
                article_number=number,
                anomaly_type="oversize_paragraph",
                chars=len(part),
            )
    chunks = [
        Chunk(
            chunk_id=f"art-{number}-p{i}",
            article_number=number,
            citation=citation(number),
            heading_path=meta["heading_path"],
            heading_path_ar=meta["heading_path_ar"],
            source_pages=meta["source_pages"],
            is_repealed=meta["is_repealed"],
            part_index=i,
            part_count=len(parts),
            paragraphs=paragraph_numbers(part),
            text_ar=part,
            text_en=meta["text_en"],
            only_in_en=meta["only_in_en"],
            only_in_ar=[p for p in meta["only_in_ar"] if p.split("\n")[0] in part],
            strategy=strategy,
        )
        for i, part in enumerate(parts, start=1)
    ]
    return chunks, oversize


def opening_connective(paragraph: str) -> str | None:
    """
    The connective a paragraph opens with, after its marker.

    returns:
    - connective (str | None): e.g. ``ومع ذلك``; None when it opens otherwise
    """
    m = CONNECTIVE.match(MARKER_PREFIX.sub("", paragraph))
    return m.group(1) if m else None


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def quality_metrics(chunks: list[Chunk], documents: list[Document]) -> dict[str, float]:
    """
    Measure a chunking's shape and how often a split separates dependent text.

    returns:
    - metrics (dict[str, float]): chunk-size distribution; the size of split
      articles' parts; how many parts after a split open with a connective, and
      the share; and the corpus base rate of paragraphs opening with one
    """
    lengths = sorted(len(c.text_ar) for c in chunks)
    split = {c.article_number for c in chunks if c.part_count > 1}
    parts = [c for c in chunks if c.article_number in split]
    later = [c for c in parts if c.part_index > 1]
    tied = sum(opening_connective(c.text_ar) is not None for c in later)
    paragraphs = [
        p
        for d in documents
        if not d.metadata["is_repealed"]
        for p in split_paragraphs(d.page_content)[1:]
    ]
    base = sum(opening_connective(p) is not None for p in paragraphs)
    return {
        "median_chunk_chars": _median(lengths),
        "p10_chunk_chars": float(lengths[len(lengths) // 10]) if lengths else 0.0,
        "chunks_under_100_chars": sum(n < SHORT_CHUNK_CHARS for n in lengths),
        "split_parts": len(parts),
        "split_part_median_chars": _median([len(c.text_ar) for c in parts]),
        "split_parts_under_100_chars": sum(
            len(c.text_ar) < SHORT_CHUNK_CHARS for c in parts
        ),
        "split_parts_with_connective": tied,
        "split_parts_with_connective_share": round(tied / len(later), 6)
        if later
        else 0.0,
        "paragraphs_with_connective_share": round(base / len(paragraphs), 6)
        if paragraphs
        else 0.0,
    }


def split_articles_report(chunks: list[Chunk]) -> str:
    """
    List every split article with its parts, for the run's artifact.

    returns:
    - text (str): Markdown, one section per split article: its size, heading,
      and each part's size, paragraphs, opening connective and start
    """
    by_article: dict[int, list[Chunk]] = {}
    for c in chunks:
        by_article.setdefault(c.article_number, []).append(c)
    split = {n: cs for n, cs in by_article.items() if len(cs) > 1}
    strategy = chunks[0].strategy if chunks else ""
    lines = [f"# Split articles ({strategy}): {len(split)}", ""]
    for n, cs in sorted(split.items()):
        whole = sum(len(c.text_ar) for c in cs)
        heading = cs[0].heading_path_ar[-1] if cs[0].heading_path_ar else ""
        lines += [f"## Article {n} ({whole} chars, {len(cs)} parts; {heading})", ""]
        for c in cs:
            tie = opening_connective(c.text_ar) if c.part_index > 1 else None
            start = c.text_ar.replace("\n", " ")[:100]
            lines.append(
                f"- part {c.part_index}: {len(c.text_ar)} chars, paragraphs "
                f"{c.paragraphs}{f', opens with {tie}' if tie else ''}: {start}"
            )
        lines.append("")
    return "\n".join(lines)


def chunk_documents(
    documents: list[Document], chunking: Chunking, embeddings: Embeddings | None = None
) -> tuple[list[Chunk], dict[str, Any]]:
    """
    Chunk every article and check that no text was lost.

    returns:
    - chunks (list[Chunk]): ordered by article, then part
    - counts (dict[str, Any]): the stage metrics

    exceptions:
    - ChunkError: an article's parts don't rebuild its text, or an article
      has no chunk
    """
    splitter = build_splitter(chunking, documents, embeddings)
    chunks: list[Chunk] = []
    counts: Counter = Counter()
    for document in documents:
        parts, oversize = chunk_document(
            document, splitter, chunking.strategy, chunking.max_chars
        )
        rebuilt = "".join(c.text_ar for c in parts)
        if re.sub(r"\s", "", rebuilt) != re.sub(r"\s", "", document.page_content):
            raise ChunkError(
                f"Article {document.metadata['article_number']}: its chunks "
                "don't rebuild its Arabic text"
            )
        chunks += parts
        counts["oversize_paragraph_anomalies"] += oversize
        counts["split_articles"] += len(parts) > 1
    merged = 0
    if chunking.repealed == "per_range":
        chunks, merged = merge_repealed_ranges(chunks)
    covered = len({n for c in chunks for n in covered_articles(c)})
    if covered != len(documents):
        raise ChunkError(f"{len(documents) - covered} articles have no chunk")
    metrics: dict[str, Any] = {
        "strategy": chunking.strategy,
        "chunks": len(chunks),
        "articles_covered": covered,
        "chunks_with_article_number": sum(c.article_number > 0 for c in chunks),
        "max_chunk_chars": max(len(c.text_ar) for c in chunks),
        "split_articles": counts["split_articles"],
        "split_article_share": round(counts["split_articles"] / covered, 6),
        "oversize_paragraph_anomalies": counts["oversize_paragraph_anomalies"],
        "chunks_with_untranslated_text": sum(
            bool(c.only_in_en or c.only_in_ar) for c in chunks
        ),
        "repealed_ranges_merged": merged,
    }
    metrics |= quality_metrics(chunks, documents)
    if isinstance(splitter, ParagraphSemanticSplitter):
        metrics["semantic_threshold"] = splitter.threshold
        metrics["semantic_breakpoints"] = splitter.breakpoints
    return chunks, metrics


def run_chunk(
    articles_in: Path,
    out: Path,
    metrics_out: Path,
    chunking: Chunking,
    embeddings: Embeddings | None = None,
    report_out: Path | None = None,
) -> dict[str, Any]:
    """
    Chunk ``articles_in`` into ``out`` (``Chunk`` records), with metrics.

    ``report_out``, when given, receives the split-articles report.

    returns:
    - metrics (dict[str, Any]): what was written to ``metrics_out``, including
      ``chunks_sha256`` (C16 compares it across two clean runs)
    """
    documents = CivilCodeArticleLoader(articles_in).load()
    chunks, metrics = chunk_documents(documents, chunking, embeddings)
    write_records(out, Chunk, chunks)
    metrics["chunks_sha256"] = sha256_file(out)
    if report_out is not None:
        report_out.write_text(split_articles_report(chunks), encoding="utf-8")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "Chunked %d articles into %d chunks",
        metrics["articles_covered"],
        metrics["chunks"],
        extra={"event_type": LogEvent.STAGE_COMPLETED, **metrics},
    )
    return metrics


def run_params(chunking: Chunking, embedding: Embedding) -> dict[str, object]:
    """
    The chunking config a run is compared by, under the names used across runs.

    Only the semantic variant uses an embedding model, so only its runs record
    one; the embed stage records the model it embeds chunks with.

    returns:
    - params (dict[str, object]): strategy, chunk size and overlap, how
      repealed ranges are chunked, plus the embedding model, its revision and the breakpoint percentile for the
      semantic variant
    """
    params: dict[str, object] = {
        "strategy": chunking.strategy,
        "chunk_size": chunking.max_chars,
        "chunk_overlap": CHUNK_OVERLAP,
        "repealed": chunking.repealed,
    }
    if chunking.strategy == "structural_semantic":
        params |= {
            "embedding_model": embedding.model,
            "embedding_revision": embedding.revision,
            "breakpoint_percentile": chunking.semantic.breakpoint_percentile,
        }
    return params


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--articles", type=Path, default=paths.corpus_dir / "articles.json")
    ap.add_argument("--out", type=Path, default=paths.corpus_dir / "chunks.json")
    ap.add_argument("--metrics", type=Path, default=paths.metrics_dir / "chunks.json")
    args = ap.parse_args(argv)
    chunking = settings.chunking
    with stage_run(
        "chunk",
        input_hash=sha256_file(args.articles),
        output_model=Chunk,
        settings=settings,
    ) as run:
        run.log_params(run_params(chunking, settings.embedding))
        embeddings = (
            huggingface_embeddings(settings.embedding)
            if chunking.strategy == "structural_semantic"
            else None
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "split_articles.md"
            metrics = run_chunk(
                args.articles,
                args.out,
                args.metrics,
                chunking,
                embeddings,
                report_out=report,
            )
            run.log_artifact(report, artifact_path="reports")
        run.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        )


if __name__ == "__main__":
    main()
