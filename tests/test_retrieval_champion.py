import json

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from raglaw.ingest.evaluate_retrieval import run_params
from raglaw.retrieval.champion import (
    ALIAS,
    MODEL_NAME,
    ChampionError,
    load_champion,
    register_champion,
)
from raglaw.retrieval.retriever import IndexMismatchError
from raglaw.tracking import stage_run

pytestmark = pytest.mark.unit


def evaluation_run(settings, **param_overrides):
    """A finished retrieval run logged with ``settings``' config, as the stage logs it."""
    with stage_run(
        "evaluate_retrieval",
        input_hash="x",
        settings=settings,
        experiment="retrieval",
        run_name="champion",
    ) as run:
        run.log_params(run_params(settings, chunks_sha256="0" * 64) | param_overrides)
        run.log_metrics({"overall.recall_at_5": 0.7})
    return run.run_id


def test_the_champion_is_registered_under_its_alias(champion_settings):
    run_id = evaluation_run(champion_settings)

    version = register_champion(
        champion_settings, run_id, heldout={"recall_at_5": 0.79}
    )

    client = MlflowClient(champion_settings.mlflow_tracking_uri)
    assert (
        client.get_model_version_by_alias(MODEL_NAME, ALIAS).version == version.version
    )
    assert version.run_id == run_id
    assert version.tags["heldout_recall_at_5"] == "0.79"


def test_the_registered_config_is_what_the_run_was_scored_with(
    champion_settings, tmp_path
):
    run_id = evaluation_run(champion_settings)
    register_champion(champion_settings, run_id)

    mlflow.set_tracking_uri(champion_settings.mlflow_tracking_uri)
    local = mlflow.artifacts.download_artifacts(
        f"models:/{MODEL_NAME}@{ALIAS}", dst_path=str(tmp_path / "model")
    )
    config = json.loads(next((tmp_path / "model").rglob("config.json")).read_text())

    assert local
    assert config["search"] == champion_settings.search.model_dump(mode="json")
    assert config["embedding"]["revision"] == champion_settings.embedding.revision


@pytest.mark.parametrize(
    ("param", "value"),
    [("retrieval_mode", "dense"), ("chunk_size", "1"), ("strategy", "other")],
)
def test_a_run_made_with_another_config_is_not_registered(
    champion_settings, param, value
):
    run_id = evaluation_run(champion_settings, **{param: value})

    with pytest.raises(ChampionError, match=param):
        register_champion(champion_settings, run_id)


@pytest.mark.parametrize(
    ("param", "value"), [("chunks_sha256", "f" * 64), ("index_device", "cuda")]
)
def test_a_run_that_searched_another_index_is_not_registered(
    champion_settings, param, value
):
    run_id = evaluation_run(champion_settings, **{param: value})

    with pytest.raises(ChampionError, match=param):
        register_champion(champion_settings, run_id)


def test_the_champion_declares_what_it_imports(champion_settings, tmp_path):
    register_champion(champion_settings, evaluation_run(champion_settings))

    mlflow.set_tracking_uri(champion_settings.mlflow_tracking_uri)
    mlflow.artifacts.download_artifacts(
        f"models:/{MODEL_NAME}@{ALIAS}", dst_path=str(tmp_path / "model")
    )
    requirements = (
        next((tmp_path / "model").rglob("requirements.txt")).read_text().split()
    )

    assert "egypt-law-rag" in requirements
    for package in ("langchain-qdrant", "qdrant-client", "bm25s", "torch"):
        assert any(r.startswith(f"{package}==") for r in requirements)


def test_the_champion_answers_with_article_citations(champion_settings, search_index):
    register_champion(champion_settings, evaluation_run(champion_settings))

    with load_champion(
        champion_settings, embeddings=search_index.embeddings
    ) as champion:
        [citations] = champion.predict(["What does Article 4 say about majority?"])

    assert citations[:3] == [
        "Article 4",
        "Article 4",
        "Article 2",
    ]  # both parts, then 2


def test_the_champion_refuses_an_index_built_differently(
    champion_settings, search_index, build_search_index, make_chunk
):
    register_champion(champion_settings, evaluation_run(champion_settings))
    # Rebuild the index on this "machine" from other chunks.
    build_search_index(
        [make_chunk(9, "نص")], {"art-9-p1": [1.0, 0.0, 0.0]}, [1.0, 0.0, 0.0]
    )

    with pytest.raises(IndexMismatchError, match="registered"):
        load_champion(champion_settings, embeddings=search_index.embeddings)


def test_closing_the_champion_releases_the_index(champion_settings, search_index):
    from raglaw.retrieval.dense import qdrant_client

    register_champion(champion_settings, evaluation_run(champion_settings))
    load_champion(champion_settings, embeddings=search_index.embeddings).close()

    qdrant_client(search_index.dense_dir).close()  # would raise if still locked
