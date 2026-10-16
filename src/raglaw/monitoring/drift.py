"""Query drift (Phase 14): are recent questions still about what the eval set covers?

The baseline is the eval set's questions embedded with the serving embedder:
their centroid, and the mean and spread of each question's cosine to it. A
window of recent questions scores the mean of their cosines to that centroid,
as a ratio to the baseline mean, so the eval set itself scores 1.0 and
questions about other topics score lower.

The alert line comes from the baseline's spread: a window of ``n`` in-domain
questions has a mean cosine whose standard deviation is about ``std / sqrt(n)``,
so a window more than ``Z`` of those below the baseline mean is drift, not
chance. ``scripts/drift_baseline.py`` writes the baseline; the API keeps the
window and exports both numbers to Prometheus.
"""

import json
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

Z = 3.0  # standard errors below the baseline mean that count as drift


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass(frozen=True)
class DriftBaseline:
    """The eval set's centroid and its questions' cosines to it."""

    centroid: list[float]
    mean: float
    std: float
    questions: int

    @classmethod
    def from_vectors(cls, vectors: Sequence[Sequence[float]]) -> DriftBaseline:
        """
        exceptions:
        - ValueError: fewer than 2 vectors
        """
        if len(vectors) < 2:
            raise ValueError("a baseline needs at least 2 questions")
        dims = len(vectors[0])
        centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dims)]
        cosines = [cosine(v, centroid) for v in vectors]
        mean = sum(cosines) / len(cosines)
        std = math.sqrt(sum((c - mean) ** 2 for c in cosines) / (len(cosines) - 1))
        return cls(centroid, mean, std, len(vectors))

    def ratio(self, vectors: Sequence[Sequence[float]]) -> float:
        """The mean cosine of ``vectors`` to the centroid, over the baseline's."""
        return sum(cosine(v, self.centroid) for v in vectors) / len(vectors) / self.mean

    def threshold(self, window: int) -> float:
        """The ratio below which a window of ``window`` questions is drift."""
        return 1 - Z * self.std / (self.mean * math.sqrt(window))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)) + "\n", "utf-8")

    @classmethod
    def load(cls, path: Path) -> DriftBaseline:
        return cls(**json.loads(path.read_text("utf-8")))


class DriftWindow:
    """The last ``size`` question vectors, scored against ``baseline``."""

    def __init__(self, baseline: DriftBaseline, size: int) -> None:
        self.baseline = baseline
        self.size = size
        self._vectors: deque[Sequence[float]] = deque(maxlen=size)

    def add(self, vector: Sequence[float]) -> float:
        """Add one question's vector; return the window's ratio."""
        self._vectors.append(vector)
        return self.baseline.ratio(self._vectors)

    @property
    def threshold(self) -> float:
        """The alert line for this window's current length."""
        return self.baseline.threshold(max(len(self._vectors), 1))
