import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    "ci_scope", Path(__file__).parents[1] / "scripts" / "ci_scope.py"
)
ci_scope = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_scope)


@pytest.mark.parametrize(
    ("paths", "runs"),
    [
        (["README.md", "docs/images/grafana_faithfulness.png"], False),
        (["CONTRIBUTING.md", "docs/serving.md", "docs/reports/optimization.md"], False),
        (["README.md", "src/raglaw/rag.py"], True),
        (["src/raglaw/prompts/answer_v2.md"], True),  # a prompt is code
        (["docs/metrics/drift_baseline.json"], True),  # baked into the API image
        (["docker/api/Dockerfile"], True),
        ([], True),  # nothing to tell from: run it
    ],
)
def test_only_documentation_skips_the_gpu_job(paths, runs):
    assert ci_scope.needs_deliver(paths) is runs


@pytest.mark.parametrize("base", ["", "0" * 40])
def test_without_a_base_the_gpu_job_runs(base):
    """A manual run, or the first push of a branch, has nothing to diff against."""
    assert ci_scope.changed(base, "HEAD") is None
