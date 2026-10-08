import os

import pytest

import raglaw.retrieval.embeddings  # noqa: F401  (its import sets the variable)

pytestmark = pytest.mark.unit


def test_loading_the_model_never_fetches_the_bot_safetensors():
    """D12: transformers' auto-conversion thread would download refs/pr/130."""
    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"
