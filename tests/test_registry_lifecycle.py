"""U35: the registry lifecycle on a real tracking server (D22)."""

import pytest
from mlflow.tracking import MlflowClient

from raglaw.retrieval.champion import (
    CANDIDATE,
    MODEL_NAME,
    PRODUCTION,
    PromotionError,
    load_champion,
    promote,
    register_champion,
)

pytestmark = pytest.mark.unit


def aliases(settings) -> dict[str, str]:
    """Each alias of the registered model -> the version it names."""
    model = MlflowClient(settings.mlflow_tracking_uri).get_registered_model(MODEL_NAME)
    return dict(model.aliases)


def candidate(settings, evaluation_run, heldout_recall: float | None):
    heldout = None if heldout_recall is None else {"recall_at_5": heldout_recall}
    return register_champion(settings, evaluation_run(settings), heldout=heldout)


def test_u35_a_registered_version_is_the_candidate_until_promoted(
    server_settings, evaluation_run
):
    v1 = candidate(server_settings, evaluation_run, 0.79)
    assert aliases(server_settings) == {CANDIDATE: v1.version}

    promoted = promote(server_settings)

    assert promoted.version == v1.version
    assert aliases(server_settings) == {PRODUCTION: v1.version}


def test_u35_a_lower_scoring_candidate_leaves_production_unchanged(
    server_settings, evaluation_run
):
    v1 = candidate(server_settings, evaluation_run, 0.79)
    promote(server_settings)
    worse = candidate(server_settings, evaluation_run, 0.57)

    with pytest.raises(PromotionError, match="below production"):
        promote(server_settings)

    assert aliases(server_settings) == {
        PRODUCTION: v1.version,
        CANDIDATE: worse.version,
    }


def test_u35_a_candidate_scoring_the_same_is_promoted(server_settings, evaluation_run):
    candidate(server_settings, evaluation_run, 0.79)
    promote(server_settings)
    same = candidate(server_settings, evaluation_run, 0.79)

    assert promote(server_settings).version == same.version


def test_nothing_is_promoted_without_a_scored_candidate(
    server_settings, evaluation_run
):
    with pytest.raises(PromotionError, match="no candidate"):
        promote(server_settings)

    candidate(server_settings, evaluation_run, None)  # registered without scores

    with pytest.raises(PromotionError, match="heldout_recall_at_5"):
        promote(server_settings)


def test_moving_production_changes_what_loads_without_code(
    server_settings, evaluation_run, search_index
):
    v1 = candidate(server_settings, evaluation_run, 0.79)
    promote(server_settings)
    v2 = candidate(server_settings, evaluation_run, 0.80)
    promote(server_settings)

    with load_champion(server_settings, embeddings=search_index.embeddings) as served:
        assert served.version == v2.version
    MlflowClient(server_settings.mlflow_tracking_uri).set_registered_model_alias(
        MODEL_NAME, PRODUCTION, v1.version
    )  # a rollback: move the alias back, nothing else
    with load_champion(server_settings, embeddings=search_index.embeddings) as served:
        assert served.version == v1.version


def test_the_server_proxies_the_models_artifacts(server_settings, evaluation_run):
    """Clients hand artifacts to the server, which stores them (S3 in compose),
    so neither a client nor the API container needs storage credentials."""
    version = candidate(server_settings, evaluation_run, 0.79)

    run = MlflowClient(server_settings.mlflow_tracking_uri).get_run(version.run_id)
    assert run.info.artifact_uri.startswith("mlflow-artifacts:")
