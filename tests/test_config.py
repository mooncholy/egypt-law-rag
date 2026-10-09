import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from raglaw.config import (
    Embedding,
    Evaluation,
    Experiments,
    Paths,
    Retrieval,
    RootHeading,
    Search,
    Settings,
)

pytestmark = pytest.mark.unit


def env_example_keys(path: Path) -> set[str]:
    lines = path.read_text("utf-8").splitlines()
    return {
        line.split("=", 1)[0]
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def params_keys(path: Path) -> set[str]:
    return set(yaml.safe_load(path.read_text("utf-8")))


def test_each_setting_has_exactly_one_home(repo_root):
    in_params = params_keys(repo_root / "params.yaml")
    in_env = {
        name
        for name in Settings.model_fields
        if f"RAGLAW_{name.upper()}" in env_example_keys(repo_root / ".env.example")
    }

    assert in_params | in_env == set(Settings.model_fields)
    assert in_params & in_env == set()


def test_env_example_holds_only_settings_fields(repo_root):
    fields = {f"RAGLAW_{name.upper()}" for name in Settings.model_fields}

    assert env_example_keys(repo_root / ".env.example") <= fields


def test_params_yaml_parses_as_settings(clean_env):
    settings = Settings(_env_file=None)

    assert settings.paths.raw_pdf == Path("data/raw/civil_code.pdf")
    assert settings.root_heading.ar == "باب تمهيدي / أحكام عامة"


def test_environment_cannot_override_params_yaml(clean_env, monkeypatch):
    monkeypatch.setenv(
        "RAGLAW_PATHS",
        json.dumps({**Settings().paths.model_dump(mode="json"), "raw_pdf": "x"}),
    )

    assert Settings(_env_file=None).paths.raw_pdf == Path("data/raw/civil_code.pdf")


def test_blank_env_values_fall_back_to_defaults(tmp_path, clean_env):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RAGLAW_LLM_API_KEY=\nRAGLAW_MLFLOW_TRACKING_URI=\n", encoding="utf-8"
    )

    settings = Settings(_env_file=env_file)

    assert settings.llm_api_key is None
    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"


def test_env_example_parses_as_settings(repo_root, clean_env):
    settings = Settings(_env_file=repo_root / ".env.example")

    assert settings.llm_api_key is None
    assert settings.mlflow_artifact_root.startswith("s3://")


def test_missing_params_yaml_fails_loudly(tmp_path, monkeypatch, clean_env):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="paths"):
        Settings(_env_file=None)


VALID_PATHS = {
    "raw_pdf": "data/raw/civil_code.pdf",
    "errata": "data/errata.yaml",
    "interim_dir": "data/interim",
    "corpus_dir": "data/corpus",
    "gold_dir": "data/gold",
    "metrics_dir": "docs/metrics",
    "reports_dir": "docs/reports",
    "analysis_dir": "docs/analysis/source_pdf",
    "index_dir": "data/index",
    "logs_dir": "logs",
}


@pytest.mark.parametrize("bad", ["/data/raw/civil_code.pdf", "../outside.pdf"])
def test_paths_must_stay_inside_the_repo(bad):
    with pytest.raises(ValidationError, match="relative to the repo root"):
        Paths.model_validate({**VALID_PATHS, "raw_pdf": bad})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("s3_bucket", "Has_Upper_And_Underscore"),
        ("s3_bucket", "ab"),
        ("mlflow_artifact_root", "gs://elsewhere/mlflow"),
        ("aws_profile", "  "),
        ("mlflow_tracking_uri", ""),
    ],
)
def test_env_settings_reject_invalid_values(clean_env, field, bad):
    with pytest.raises(ValidationError, match=field):
        Settings(_env_file=None, **{field: bad})


@pytest.mark.parametrize("bad", ["Corpus Build", "-leading-dash", ""])
def test_experiment_name_must_be_safe_as_an_s3_prefix(bad):
    with pytest.raises(ValidationError, match="corpus"):
        Experiments(corpus=bad)


def test_root_heading_must_not_be_blank():
    with pytest.raises(ValidationError, match="en"):
        RootHeading(ar="باب تمهيدي / أحكام عامة", en="   ")


def test_one_embedding_block_serves_chunking_and_index(clean_env):
    settings = Settings(_env_file=None)

    assert settings.embedding.model == "BAAI/bge-m3"
    assert "model" not in settings.chunking.semantic.model_dump()


@pytest.mark.parametrize(
    "bad", ["main", "5617a9f", "5617A9F61B028005A4858FDAC845DB406AEFB181"]
)
def test_embedding_revision_must_be_a_full_commit_sha(bad):
    with pytest.raises(ValidationError, match="revision"):
        Embedding(model="BAAI/bge-m3", revision=bad, batch_size=32)


def test_search_needs_at_least_top_k_candidates():
    with pytest.raises(ValidationError, match="candidates"):
        Search(mode="hybrid", article_lookup=True, candidates=5, rrf_k=60, top_k=10)


def test_stopwords_apply_only_to_word_tokenizers():
    """Subword pieces aren't words, so no stopword list can filter them."""
    with pytest.raises(ValidationError, match="bm25_stopwords"):
        Retrieval(
            document_text="both_with_headings",
            bm25_tokenizer="model_subwords",
            bm25_stopwords="nltk",
        )


def test_the_variant_switches_default_to_off():
    retrieval = Retrieval(document_text="both_with_headings", bm25_tokenizer="words")

    assert retrieval.bm25_stopwords == "none"
    assert Settings(_env_file=None).chunking.repealed == "per_article"


@pytest.mark.parametrize("bad", [0, 1.5])
def test_the_recall_target_is_a_share(bad):
    with pytest.raises(ValidationError, match="target_recall_at_5"):
        Evaluation(split="tuning", target_recall_at_5=bad)


def test_the_new_search_switches_default_to_off():
    search = Settings(_env_file=None).search

    assert (search.rerank, search.cite_expansion) == (False, False)
    assert Settings(_env_file=None).retrieval.repealed_text == "note"


def test_the_reranker_is_pinned_like_the_embedder():
    reranker = Settings(_env_file=None).reranker

    assert reranker.model == "BAAI/bge-reranker-v2-m3"
    assert len(reranker.revision) == 40


def test_rerank_depth_must_cover_top_k():
    with pytest.raises(ValidationError, match="rerank_depth"):
        Search(
            mode="hybrid",
            article_lookup=True,
            candidates=50,
            rrf_k=60,
            top_k=10,
            rerank=True,
            rerank_depth=5,
        )


def test_the_device_is_per_machine_and_defaults_to_cpu(clean_env, monkeypatch):
    assert Settings(_env_file=None).device == "cpu"
    monkeypatch.setenv("RAGLAW_DEVICE", "cuda")
    assert Settings(_env_file=None).device == "cuda"
