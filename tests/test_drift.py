import math
import random

import pytest

from raglaw.monitoring.drift import DriftBaseline, DriftWindow

pytestmark = pytest.mark.unit


def around(
    direction: list[float], spread: float, n: int, seed: int
) -> list[list[float]]:
    """``n`` unit vectors scattered around ``direction``."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        v = [d + rng.gauss(0, spread) for d in direction]
        norm = math.sqrt(sum(x * x for x in v))
        out.append([x / norm for x in v])
    return out


CIVIL = [1.0] + [0.0] * 15  # the eval set's topic
FAMILY = [0.0, 1.0] + [0.0] * 14  # an off-topic one (family law)


def test_u41_the_eval_set_scores_near_one_and_an_off_topic_set_falls():
    """U41: drift is ~1.0 on the eval set itself and drops for another topic."""
    eval_set = around(CIVIL, 0.15, 140, seed=1)
    baseline = DriftBaseline.from_vectors(eval_set)
    off_topic = around(FAMILY, 0.15, 20, seed=2)

    assert baseline.ratio(eval_set) == pytest.approx(1.0)
    assert baseline.ratio(off_topic) < baseline.threshold(20) < 1.0
    same_topic = around(CIVIL, 0.15, 20, seed=3)  # new questions, same topic
    assert baseline.ratio(same_topic) > baseline.threshold(20)


def test_the_window_keeps_only_the_latest_questions():
    baseline = DriftBaseline.from_vectors(around(CIVIL, 0.15, 50, seed=1))
    window = DriftWindow(baseline, size=5)
    for v in around(FAMILY, 0.15, 5, seed=2):
        window.add(v)

    recovered = [window.add(v) for v in around(CIVIL, 0.15, 5, seed=3)][-1]

    assert recovered > window.threshold


def test_a_saved_baseline_loads_the_same(tmp_path):
    baseline = DriftBaseline.from_vectors(around(CIVIL, 0.15, 10, seed=1))
    baseline.save(tmp_path / "drift.json")

    assert DriftBaseline.load(tmp_path / "drift.json") == baseline
