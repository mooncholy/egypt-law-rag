"""The benchmark harness (Phase 13): time one operation the same way every time.

A latency measured once means little: the first calls pay for lazy loading,
kernel compilation and cache misses, and a GPU call returns before its work is
done. So the harness runs ``warmup`` untimed calls first, then ``iterations``
timed ones, and calls ``sync`` (``torch.cuda.synchronize`` on a GPU) before it
reads the clock, so each time covers the work and not only its launch.

``scripts/benchmark.py`` uses it for the rerankers and the generator.
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Timing:
    """Latency over the timed iterations, in milliseconds."""

    warmup: int
    iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(
    run: Callable[[int], object],
    *,
    warmup: int,
    iterations: int,
    sync: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.perf_counter,
) -> Timing:
    """
    Call ``run(i)`` ``warmup`` times untimed, then ``iterations`` times timed;
    ``i`` counts every call from 0, so ``run`` can cycle through its inputs.

    returns:
    - timing (Timing): the timed iterations' mean and percentiles

    exceptions:
    - ValueError: fewer than 2 timed iterations, or a negative warmup
    """
    if iterations < 2 or warmup < 0:
        raise ValueError("benchmark needs warmup >= 0 and iterations >= 2")
    for i in range(warmup):
        run(i)
    sync()
    times = []
    for i in range(warmup, warmup + iterations):
        start = clock()
        run(i)
        sync()
        times.append((clock() - start) * 1000)
    cuts = statistics.quantiles(times, n=100, method="inclusive")
    return Timing(
        warmup=warmup,
        iterations=iterations,
        mean_ms=statistics.fmean(times),
        p50_ms=statistics.median(times),
        p95_ms=cuts[94],
        p99_ms=cuts[98],
    )
