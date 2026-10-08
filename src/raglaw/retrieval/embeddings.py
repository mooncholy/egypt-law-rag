"""The pinned embedding model, shared by the semantic chunking variant and ``index``.

Imported only by code that embeds, so the structural strategy and the API never
load ``torch``.
"""

from langchain_core.embeddings import Embeddings

from raglaw.config import Embedding


def huggingface_embeddings(embedding: Embedding) -> Embeddings:
    """
    Load the pinned local embedding model, importing it only when called.

    returns:
    - embeddings (Embeddings): ``HuggingFaceEmbeddings`` on CPU, normalized,
      encoding ``embedding.batch_size`` texts per batch

    exceptions:
    - ImportError: the ``embed`` dependency group isn't installed
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # embed group only

    # BAAI's revision ships only pytorch_model.bin. Without this flag,
    # transformers fetches model.safetensors from an unmerged bot pull request
    # (refs/pr/130) instead of the pinned revision (D12). torch loads the .bin
    # with weights_only=True, which refuses pickled code.
    return HuggingFaceEmbeddings(
        model_name=embedding.model,
        model_kwargs={
            "device": "cpu",
            "revision": embedding.revision,
            "model_kwargs": {"use_safetensors": False},
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": embedding.batch_size,
        },
    )
