import pytest

from raglaw.bench import benchmark

pytestmark = pytest.mark.unit


class StubGPU:
    """A model on a fake clock: a call returns after 1 ms of launch, and its
    other 9 ms of work finish only at ``sync``. The first 3 calls (cold) take
    100 ms."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pending = 0.0
        self.calls: list[int] = []

    def clock(self) -> float:
        return self.now

    def run(self, i: int) -> None:
        self.calls.append(i)
        cold = len(self.calls) <= 3
        self.now += 0.001
        self.pending += 0.099 if cold else 0.009

    def sync(self) -> None:
        self.now += self.pending
        self.pending = 0.0


def test_warmup_is_discarded_and_the_timer_stops_after_sync():
    """U40: the cold calls are not timed, and each time includes the work
    still queued when the call returned (10 ms, not 1 ms)."""
    gpu = StubGPU()

    timing = benchmark(
        gpu.run, warmup=3, iterations=500, sync=gpu.sync, clock=gpu.clock
    )

    assert gpu.calls == list(range(503))
    assert timing.iterations == 500
    assert timing.mean_ms == pytest.approx(10)
    assert timing.p50_ms == pytest.approx(10)
    assert timing.p99_ms == pytest.approx(10)


def test_without_warmup_the_cold_calls_show_in_the_tail():
    gpu = StubGPU()

    timing = benchmark(
        gpu.run, warmup=0, iterations=100, sync=gpu.sync, clock=gpu.clock
    )

    assert timing.p50_ms == pytest.approx(10)
    assert timing.p99_ms > 90


@pytest.mark.parametrize(("warmup", "iterations"), [(-1, 10), (0, 1)])
def test_too_few_iterations_are_refused(warmup, iterations):
    with pytest.raises(ValueError):
        benchmark(lambda i: None, warmup=warmup, iterations=iterations)
