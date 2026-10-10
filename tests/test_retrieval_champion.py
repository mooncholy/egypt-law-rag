import json

import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from raglaw.retrieval.champion import (
    CANDIDATE,
    MODEL_NAME,
    PRODUCTION,
    ChampionError,
    load_champion,
    register_champion,
)
from raglaw.retrieval.retriever import IndexMismatchError

pytestmark = pytest.mark.unit


def test_a_registered_run_is_the_candidate_with_its_heldout_scores(
    champion_settings, evaluation_run
):
    run_id = evaluation_run(champion_settings)
    heldout_id = evaluation_run(
        champion_settings, recall_at_5=0.79, mrr=0.8, split="heldout"
    )

    version = register_champion(champion_settings, run_id, heldout_run_id=heldout_id)

    client = MlflowClient(champion_settings.mlflow_tracking_uri)
    candidate = client.get_model_version_by_alias(MODEL_NAME, CANDIDATE)
    assert candidate.version == version.version
    assert PRODUCTION not in version.aliases  # nothing serves it before promotion
    assert version.run_id == run_id
    assert (
        version.tags["heldout_run_id"],
        version.tags["heldout_recall_at_5"],
        version.tags["heldout_mrr"],
    ) == (heldout_id, "0.79", "0.8")


def test_the_heldout_scores_must_come_from_a_heldout_run(
    champion_settings, evaluation_run
):
    tuning = evaluation_run(champion_settings, recall_at_5=0.99)  # split: tuning

    with pytest.raises(ChampionError, match="not the held-out half"):
        register_champion(
            champion_settings, evaluation_run(champion_settings), heldout_run_id=tuning
        )


def test_the_heldout_run_must_have_scored_this_config(
    champion_settings, evaluation_run
):
    other = evaluation_run(
        champion_settings, recall_at_5=0.99, split="heldout", retrieval_mode="dense"
    )

    with pytest.raises(ChampionError, match="retrieval_mode"):
        register_champion(
            champion_settings, evaluation_run(champion_settings), heldout_run_id=other
        )


def test_loading_serves_the_production_version(
    champion_settings, search_index, in_production
):
    version = in_production(champion_settings)

    with load_champion(
        champion_settings, embeddings=search_index.embeddings
    ) as champion:
        assert champion.version == version.version


def test_without_production_nothing_loads(champion_settings, tmp_path, evaluation_run):
    empty = champion_settings.model_copy(
        update={"mlflow_tracking_uri": f"sqlite:///{tmp_path / 'fresh.db'}"}
    )
    register_champion(empty, evaluation_run(empty))  # a candidate only

    with pytest.raises(MlflowException):
        load_champion(empty)


def test_the_registered_config_is_what_the_run_was_scored_with(
    champion_settings, tmp_path, evaluation_run
):
    run_id = evaluation_run(champion_settings)
    register_champion(champion_settings, run_id)

    mlflow.set_tracking_uri(champion_settings.mlflow_tracking_uri)
    local = mlflow.artifacts.download_artifacts(
        f"models:/{MODEL_NAME}@{CANDIDATE}", dst_path=str(tmp_path / "model")
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
    champion_settings, param, value, evaluation_run
):
    run_id = evaluation_run(champion_settings, **{param: value})

    with pytest.raises(ChampionError, match=param):
        register_champion(champion_settings, run_id)


@pytest.mark.parametrize(
    ("param", "value"), [("chunks_sha256", "f" * 64), ("index_device", "cuda")]
)
def test_a_run_that_searched_another_index_is_not_registered(
    champion_settings, param, value, evaluation_run
):
    run_id = evaluation_run(champion_settings, **{param: value})

    with pytest.raises(ChampionError, match=param):
        register_champion(champion_settings, run_id)


def test_the_champion_declares_what_it_imports(
    champion_settings, tmp_path, evaluation_run
):
    register_champion(champion_settings, evaluation_run(champion_settings))

    mlflow.set_tracking_uri(champion_settings.mlflow_tracking_uri)
    mlflow.artifacts.download_artifacts(
        f"models:/{MODEL_NAME}@{CANDIDATE}", dst_path=str(tmp_path / "model")
    )
    requirements = (
        next((tmp_path / "model").rglob("requirements.txt")).read_text().split()
    )

    assert "egypt-law-rag" in requirements
    for package in ("langchain-qdrant", "qdrant-client", "bm25s", "torch"):
        assert any(r.startswith(f"{package}==") for r in requirements)


def test_the_champion_answers_with_article_citations(
    champion_settings, search_index, in_production
):
    in_production(champion_settings)

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
    champion_settings, search_index, build_search_index, make_chunk, in_production
):
    in_production(champion_settings)
    # Rebuild the index on this "machine" from other chunks.
    build_search_index(
        [make_chunk(9, "نص")], {"art-9-p1": [1.0, 0.0, 0.0]}, [1.0, 0.0, 0.0]
    )

    with pytest.raises(IndexMismatchError, match="registered"):
        load_champion(champion_settings, embeddings=search_index.embeddings)


def test_closing_the_champion_releases_the_index(
    champion_settings, search_index, in_production
):
    from raglaw.retrieval.dense import qdrant_client

    in_production(champion_settings)
    load_champion(champion_settings, embeddings=search_index.embeddings).close()

    qdrant_client(search_index.dense_dir).close()  # would raise if still locked


def test_the_champion_retrieves_chunks_with_their_scores(
    champion_settings, search_index, in_production
):
    in_production(champion_settings)

    with load_champion(
        champion_settings, embeddings=search_index.embeddings
    ) as champion:
        documents = champion.retrieve("What does Article 4 say about majority?")

        assert champion.documents_indexed == 5
    assert documents[0].metadata["retrieval"]["lookup"] is True
    assert [d.metadata["article_number"] for d in documents[:3]] == [4, 4, 2]
    with pytest.raises(RuntimeError, match="closed"):
        champion.retrieve("anything")
