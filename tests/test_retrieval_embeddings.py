import os

import pytest

import raglaw.retrieval.embeddings  # noqa: F401  (its import sets the variable)

pytestmark = pytest.mark.unit


def test_loading_the_model_never_fetches_the_bot_safetensors():
    """D12: transformers' auto-conversion thread would download refs/pr/130."""
    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_cuda_without_a_gpu_fails_before_any_model_loads(monkeypatch):
    """A machine set to `cuda` but running CPU-only torch must not fall back silently."""
    import torch

    from raglaw.retrieval.embeddings import DeviceError, check_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceError, match="torch-gpu"):
        check_device("cuda")
    check_device("cpu")  # always available
