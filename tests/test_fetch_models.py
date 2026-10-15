import pytest

from raglaw.config import Settings
from raglaw.serving import fetch_models

pytestmark = pytest.mark.unit


def test_both_pinned_models_are_loaded_once_on_cpu(monkeypatch, clean_env):
    """compose's models-pull: the service's own loaders, so it fetches exactly
    the files the service reads, at the pinned revisions."""
    loaded = []
    monkeypatch.setattr(
        fetch_models,
        "huggingface_embeddings",
        lambda embedding, device: loaded.append((embedding.revision, device)),
    )
    monkeypatch.setattr(
        fetch_models,
        "CrossEncoderReranker",
        lambda reranker, device: loaded.append((reranker.revision, device)),
    )

    fetch_models.main()

    settings = Settings(_env_file=None)
    assert loaded == [
        (settings.embedding.revision, "cpu"),
        (settings.reranker.revision, "cpu"),
    ]
