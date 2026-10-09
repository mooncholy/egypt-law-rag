"""Reciprocal rank fusion (RRF): combine ranked lists by rank alone.

A document scores the sum of ``1 / (k + rank)`` over the lists it appears in,
with ranks counted from 1. Only ranks are combined, so a dense cosine and a BM25
score never need to be put on one scale. A larger ``k`` flattens the weight of
the top ranks; 60 is the value from Cormack et al. (2009).
"""

from collections.abc import Sequence

DEFAULT_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = DEFAULT_K
) -> list[tuple[str, float]]:
    """
    Fuse ranked lists of ids into one ranking.

    An id repeated within one list counts only at its best rank there.

    returns:
    - fused (list[tuple[str, float]]): ``(id, score)`` from the highest score
      down; ties ordered by id, so the result never depends on list order

    exceptions:
    - ValueError: ``k`` is negative
    """
    if k < 0:
        raise ValueError(f"RRF k must be 0 or more, got {k}")
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id not in seen:
                seen.add(doc_id)
                scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
