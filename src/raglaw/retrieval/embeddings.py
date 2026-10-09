"""The pinned embedding model, shared by the semantic chunking variant and ``embed``.

Imported only by code that embeds, so the structural strategy and the API never
load ``torch``.
"""

import os
from typing import Any, Protocol

# BAAI's pinned revision ships only pytorch_model.bin. After loading a .bin,
# transformers starts a background thread that finds the SFconvertbot pull
# request (refs/pr/130) and downloads its model.safetensors "for next time"
# (2.2 GB). It never changes the weights loaded here (use_safetensors=False),
# but it fetches the very files D12 rules out, on every load.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from langchain_core.embeddings import Embeddings

from raglaw.config import Embedding


class DeviceError(RuntimeError):
    """The configured device isn't available in this environment."""


def check_device(device: str) -> None:
    """
    Fail before loading a model if ``device`` can't be used.

    exceptions:
    - DeviceError: ``cuda`` without a usable GPU, e.g. with the default
      CPU-only torch installed
    """
    if device != "cuda":
        return
    import torch  # embed group only

    if not torch.cuda.is_available():
        raise DeviceError(
            "RAGLAW_DEVICE=cuda, but torch sees no GPU. Install the CUDA build "
            "with `uv sync --no-group torch-cpu --group torch-gpu` (README), "
            "or set RAGLAW_DEVICE=cpu."
        )


class ModelTokenizer(Protocol):
    """The parts of the model's Hugging Face tokenizer the pipeline uses."""

    model_max_length: int

    def tokenize(self, text: str) -> list[str]: ...

    def __call__(self, text: str) -> Any: ...


def huggingface_embeddings(embedding: Embedding, device: str = "cpu") -> Embeddings:
    """
    Load the pinned local embedding model, importing it only when called.

    returns:
    - embeddings (Embeddings): ``HuggingFaceEmbeddings`` on ``device``,
      normalized, encoding ``embedding.batch_size`` texts per batch

    exceptions:
    - ImportError: the ``embed`` dependency group isn't installed
    - DeviceError: ``device`` isn't available
    """
    check_device(device)
    from langchain_huggingface import HuggingFaceEmbeddings  # embed group only

    # BAAI's revision ships only pytorch_model.bin. Without this flag,
    # transformers fetches model.safetensors from an unmerged bot pull request
    # (refs/pr/130) instead of the pinned revision (D12). torch loads the .bin
    # with weights_only=True, which refuses pickled code.
    return HuggingFaceEmbeddings(
        model_name=embedding.model,
        model_kwargs={
            "device": device,
            "revision": embedding.revision,
            "model_kwargs": {"use_safetensors": False},
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": embedding.batch_size,
        },
    )


def model_tokenizer(embedding: Embedding) -> ModelTokenizer:
    """
    Load the embedding model's own tokenizer, at the pinned revision.

    It counts each chunk's tokens against ``model_max_length`` (the length past
    which the model truncates), and is the ``model_subwords`` BM25 tokenizer.

    returns:
    - tokenizer (ModelTokenizer): the Hugging Face tokenizer

    exceptions:
    - ImportError: the ``embed`` dependency group isn't installed
    """
    from transformers import AutoTokenizer  # embed group only

    return AutoTokenizer.from_pretrained(embedding.model, revision=embedding.revision)
