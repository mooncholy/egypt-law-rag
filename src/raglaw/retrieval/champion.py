"""``champion`` (D16): the chosen retrieval config, in the MLflow Model Registry.

The registered model is a pyfunc wrapping ``HybridRetriever``. Its artifacts are
the config sections that define retrieval and both index manifests; the index
itself stays in DVC. A deployment loads it by alias, never by path:
``models:/civil-code-retriever@champion``. Opening it checks that the index on
that machine is the one the champion was scored on, so a registered score can't
silently describe another index.

This is the only retrieval module that imports MLflow.

Register with ``python -m raglaw.retrieval.champion`` (see ``main``).
"""

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Self

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.pyfunc
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from raglaw.config import Embedding, Reranker, Retrieval, Search, Settings
from raglaw.retrieval.dense import DenseManifest
from raglaw.retrieval.lexical import Bm25Manifest
from raglaw.retrieval.manifest import MANIFEST, read_manifest
from raglaw.retrieval.rerank import RerankScorer
from raglaw.retrieval.retriever import HybridRetriever, IndexMismatchError

logger = logging.getLogger(__name__)

MODEL_NAME = "civil-code-retriever"
ALIAS = "champion"
# What defines a retrieval config; `evaluation` (the split, the target) doesn't.
CONFIG_SECTIONS = ("chunking", "embedding", "reranker", "retrieval", "search")
# The run params that needn't agree with the config being registered: the
# machine the run scored on, and which half of the eval set it scored. Every
# other one must, including the chunks and device of the index it searched.
UNCHECKED_PARAMS = frozenset({"device", "split"})
# What the registered model imports beyond the project itself. The project's
# dependency groups don't install with it, so a model environment names them.
RUNTIME_PACKAGES = (
    "mlflow",
    "langchain-core",
    "langchain-qdrant",
    "qdrant-client",
    "bm25s",
    "langchain-huggingface",
    "sentence-transformers",
    "torch",
)


class ChampionError(RuntimeError):
    """The run doesn't match the config being registered."""


class ChampionRetriever(mlflow.pyfunc.PythonModel):
    """The registered retriever: question in, article citations out."""

    def load_context(self, context: Any) -> None:
        artifacts = context.artifacts
        self.config = json.loads(Path(artifacts["config"]).read_text("utf-8"))
        self.manifests = {
            "dense": DenseManifest.model_validate_json(
                Path(artifacts["dense_manifest"]).read_text("utf-8")
            ),
            "bm25": Bm25Manifest.model_validate_json(
                Path(artifacts["bm25_manifest"]).read_text("utf-8")
            ),
        }
        self._retriever: HybridRetriever | None = None

    def open(
        self,
        settings: Settings,
        *,
        embeddings: Embeddings | None = None,
        reranker: RerankScorer | None = None,
    ) -> HybridRetriever:
        """
        Open the index on this machine with the registered config.

        ``embeddings`` and ``reranker`` default to the registered models on
        ``settings.device``.

        exceptions:
        - IndexMismatchError: the index differs from the registered one
        """
        index = Path(settings.paths.index_dir)
        found = {
            "dense": read_manifest(index / "dense", DenseManifest),
            "bm25": read_manifest(index / "bm25", Bm25Manifest),
        }
        if found != self.manifests:
            raise IndexMismatchError(
                f"the index at {index} isn't the registered one (its manifests "
                "differ); pull the index the champion was scored on"
            )
        embedding = Embedding.model_validate(self.config["embedding"])
        search = Search.model_validate(self.config["search"])
        if embeddings is None:
            from raglaw.retrieval.embeddings import huggingface_embeddings

            embeddings = huggingface_embeddings(embedding, settings.device)
        if reranker is None and search.rerank:
            from raglaw.retrieval.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                Reranker.model_validate(self.config["reranker"]), settings.device
            )
        self._retriever = HybridRetriever.from_index(
            index / "dense",
            index / "bm25",
            embedding=embedding,
            retrieval=Retrieval.model_validate(self.config["retrieval"]),
            search=search,
            embeddings=embeddings,
            reranker=reranker,
        )
        return self._retriever

    def predict(
        self, context: Any, model_input: list[str], params: dict | None = None
    ) -> list[list[str]]:
        """
        Retrieve for each question.

        returns:
        - citations (list[list[str]]): per question, the article citation of
          each returned chunk, in rank order
        """
        retriever = self._retriever or self.open(Settings())
        return [
            [d.metadata["citation"] for d in retriever.invoke(q)] for q in model_input
        ]


def _check_run(
    client: MlflowClient, run_id: str, settings: Settings, index: DenseManifest
) -> None:
    from raglaw.ingest.evaluate_retrieval import run_params  # stage code, pipeline only

    logged = client.get_run(run_id).data.params
    wanted = {
        k: str(v)
        for k, v in run_params(
            settings, chunks_sha256=index.chunks_sha256, index_device=index.device
        ).items()
        if k not in UNCHECKED_PARAMS
    }
    differ = [
        f"{k}: run {logged.get(k)!r}, config {v!r}"
        for k, v in wanted.items()
        if logged.get(k) != v
    ]
    if differ:
        raise ChampionError(
            f"run {run_id} wasn't scored with this config and index "
            f"({'; '.join(differ)})"
        )


def _pip_requirements() -> list[str]:
    """The project, and each package the model imports pinned to its version here."""
    # A local build tag (torch's `+cpu`) names an index, not a PyPI release.
    return ["egypt-law-rag"] + [
        f"{p}=={version(p).split('+')[0]}" for p in RUNTIME_PACKAGES
    ]


def register_champion(
    settings: Settings, run_id: str, heldout: dict[str, float] | None = None
) -> ModelVersion:
    """
    Register the config ``run_id`` was scored with as the new ``champion``.

    ``heldout`` holds the held-out half's scores (D15), kept as version tags.

    returns:
    - version (ModelVersion): the new version, which the alias now names

    exceptions:
    - ChampionError: the run's params differ from ``settings``' config, or it
      searched another index than the one at ``paths.index_dir``
    """
    if settings.aws_profile:  # run artifacts live in S3
        os.environ.setdefault("AWS_PROFILE", settings.aws_profile)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(settings.mlflow_tracking_uri)
    index = Path(settings.paths.index_dir)
    _check_run(client, run_id, settings, read_manifest(index / "dense", DenseManifest))
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "config.json"
        sections = {
            s: getattr(settings, s).model_dump(mode="json") for s in CONFIG_SECTIONS
        }
        config.write_text(
            json.dumps(sections, indent=2, sort_keys=True) + "\n", "utf-8"
        )
        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(
                name="retriever",
                python_model=ChampionRetriever(),
                artifacts={
                    "config": str(config),
                    "dense_manifest": str(index / "dense" / MANIFEST),
                    "bm25_manifest": str(index / "bm25" / MANIFEST),
                },
                pip_requirements=_pip_requirements(),
            )
    version = mlflow.register_model(info.model_uri, MODEL_NAME)
    for key, value in (heldout or {}).items():
        client.set_model_version_tag(
            MODEL_NAME, version.version, f"heldout_{key}", str(value)
        )
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    logger.info("Registered %s version %s as %s", MODEL_NAME, version.version, ALIAS)
    return client.get_model_version(MODEL_NAME, version.version)


def load_champion(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
    reranker: RerankScorer | None = None,
) -> Champion:
    """
    Load ``champion`` by alias and open it on this machine's index.

    returns:
    - champion (Champion): call ``predict(questions)``

    exceptions:
    - IndexMismatchError: this machine's index isn't the registered one
    """
    if settings.aws_profile:  # the model's artifacts live in S3
        os.environ.setdefault("AWS_PROFILE", settings.aws_profile)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    champion = model.unwrap_python_model()
    champion.open(settings, embeddings=embeddings, reranker=reranker)
    return Champion(champion)


class Champion:
    """The loaded champion, answering ``predict(questions)`` without pyfunc's context.

    It holds the index's Qdrant lock until ``close`` (or the end of a ``with``).
    """

    def __init__(self, champion: ChampionRetriever) -> None:
        self.champion = champion

    def predict(self, questions: Sequence[str]) -> list[list[str]]:
        return self.champion.predict(None, list(questions))

    def retrieve(self, question: str) -> list[Document]:
        """
        The chunks for one question, with their scores (``HybridRetriever``).

        returns:
        - documents (list[Document]): in rank order, at most ``search.top_k``
        """
        retriever = self.champion._retriever
        if retriever is None:
            raise RuntimeError("the champion is closed")
        return retriever.invoke(question)

    @property
    def documents_indexed(self) -> int:
        """Chunks in the index the champion was registered with."""
        return len(self.champion.manifests["dense"].chunk_ids)

    def close(self) -> None:
        """Release the index."""
        if self.champion._retriever is not None:
            self.champion._retriever.close()
            self.champion._retriever = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _latest_run(client: MlflowClient, experiment: str, name: str) -> Any:
    exp = client.get_experiment_by_name(experiment)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.run_name = '{name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No finished run named {name!r} in {experiment!r}")
    return runs[0]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="champion", help="the tuning run to register")
    ap.add_argument(
        "--heldout-run", default="champion-heldout", help="its one held-out run (D15)"
    )
    args = ap.parse_args(argv)
    client = MlflowClient(settings.mlflow_tracking_uri)
    experiment = settings.tracking.experiments.retrieval
    run = _latest_run(client, experiment, args.run)
    heldout = _latest_run(client, experiment, args.heldout_run)
    metrics = heldout.data.metrics
    version = register_champion(
        settings,
        run.info.run_id,
        heldout={
            "run_id": heldout.info.run_id,
            "recall_at_5": metrics["overall.recall_at_5"],
            "mrr": metrics["overall.mrr"],
        },
    )
    logger.info("models:/%s@%s -> version %s", MODEL_NAME, ALIAS, version.version)


if __name__ == "__main__":
    main()
