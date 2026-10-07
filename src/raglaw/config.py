from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, SecretStr, StringConstraints
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def _inside_repo(path: Path) -> Path:
    """Reject absolute paths and ``..``: dvc.yaml names the same paths from the root."""
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("must be relative to the repo root and stay inside it")
    return path


RepoPath = Annotated[Path, AfterValidator(_inside_repo)]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Paths(BaseModel):
    """Where the pipeline reads and writes, relative to the repo root."""

    raw_pdf: RepoPath = Field(
        description="The source PDF, placed by `dvc pull` (DVC-tracked)."
    )
    errata: RepoPath = Field(
        description="Owner-approved fixes for one-off source errors (git-tracked YAML)."
    )
    interim_dir: RepoPath = Field(
        description="Row-level outputs of `extract` and `repair` (DVC-tracked)."
    )
    corpus_dir: RepoPath = Field(
        description="`articles.json` and `chunks.json` (DVC-tracked)."
    )
    gold_dir: RepoPath = Field(
        description="The 30 hand-corrected gold articles (DVC-tracked)."
    )
    metrics_dir: RepoPath = Field(
        description="Stage metrics as JSON, read by `dvc metrics` (git-tracked)."
    )
    reports_dir: RepoPath = Field(
        description="Generated and hand-written reports (git-tracked)."
    )
    analysis_dir: RepoPath = Field(
        description="Evidence behind the source analysis report, written by "
        "`profile` (gitignored; pinned by its `.sha256` file)."
    )
    logs_dir: RepoPath = Field(
        description="One JSONL log per stage run (gitignored; attached to MLflow)."
    )


class RootHeading(BaseModel):
    """The page 1 heading that opens every heading path (P7)."""

    ar: NonEmptyStr = Field(description="The heading as printed on page 1.")
    en: NonEmptyStr = Field(
        description="English label; the source has none, so the owner supplies it "
        "(D7). `TODO` until then."
    )


# An experiment's name is also its S3 prefix under the artifact root.
ExperimentName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]*$")]


class Experiments(BaseModel):
    """One MLflow experiment per kind of work, since runs are compared within one.

    Later kinds (retrieval and answer evaluation) add a field each here.
    """

    corpus: ExperimentName = Field(
        description="Every `dvc repro` stage run (`stage_run`). Lowercase "
        "letters, digits, `.`, `_` and `-` only."
    )


class Semantic(BaseModel):
    """The ``structural_semantic`` strategy's embedding model and threshold."""

    model: NonEmptyStr = Field(description="Hugging Face model id.")
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")] = Field(
        description="The model's pinned commit, so a rebuild embeds the same way."
    )
    breakpoint_percentile: float = Field(
        gt=0,
        lt=100,
        description="Neighbouring paragraphs split where their distance is above "
        "this percentile of all such distances in the corpus.",
    )
    batch_size: int = Field(gt=0, description="Paragraphs embedded per batch.")


class Chunking(BaseModel):
    """How articles become retrievable chunks (Phase 5)."""

    strategy: Literal["structural", "structural_semantic"] = Field(
        description="`structural` splits at paragraph markers only; "
        "`structural_semantic` also splits where meaning shifts."
    )
    max_chars: int = Field(
        gt=0, description="Longest Arabic text in one chunk; a paragraph is never cut."
    )
    semantic: Semantic = Field(description="Settings for `structural_semantic`.")


class Tracking(BaseModel):
    """How MLflow groups the project's runs."""

    experiments: Experiments = Field(
        description="The experiment for each kind of work."
    )


class Settings(BaseSettings):
    """Every configurable value in the project, from two sources.

    - ``params.yaml`` (tracked): project structure and parameters, shared with
      ``dvc.yaml``. Changing one is a reviewed commit, never a local override.
    - The environment or ``.env`` (untracked): per-machine and secret values,
      under the ``RAGLAW_`` prefix (``llm_api_key`` ← ``RAGLAW_LLM_API_KEY``).

    ``params.yaml`` ranks above the environment, so a path can't be changed
    for one machine only. Both files are read from the working directory:
    the repo root for ``dvc repro`` and tests, the app directory in Docker.
    """

    # env_ignore_empty: a blank ``KEY=`` in .env means "use the default", so a
    # copied .env.example never turns a value into "".
    model_config = SettingsConfigDict(
        env_prefix="RAGLAW_",
        env_file=".env",
        env_ignore_empty=True,
        yaml_file="params.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    # From params.yaml
    paths: Paths = Field(description="Pipeline inputs and outputs.")
    root_heading: RootHeading = Field(description="The fixed root of heading paths.")
    tracking: Tracking = Field(description="MLflow run grouping.")
    chunking: Chunking = Field(description="Chunking strategy and limits.")

    # From the environment or .env
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="Key for the OpenAI-compatible LLM backend. Optional so the "
        "service still starts without it: /health reports the gap and /ask "
        "answers 503.",
    )
    s3_bucket: (
        Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")]
        | None
    ) = Field(
        default=None,
        description="The project bucket (prefixes `dvc/`, `mlflow/`, `eval/`). S3 "
        "naming rules: 3–63 lowercase letters, digits, dots and hyphens.",
    )
    mlflow_tracking_uri: NonEmptyStr = Field(
        default="sqlite:///mlflow.db",
        description="Where MLflow keeps run records: a local, gitignored sqlite file.",
    )
    mlflow_artifact_root: (
        Annotated[str, StringConstraints(pattern=r"^(s3|file)://\S+$")] | None
    ) = Field(
        default=None,
        description="Root for run artifacts: `s3://<bucket>/mlflow`, or `file://` in "
        "tests. Each experiment writes under `<root>/<experiment>`. Unset, MLflow "
        "keeps artifacts beside its database.",
    )
    aws_profile: NonEmptyStr | None = Field(
        default=None,
        description="The ~/.aws profile boto3 uses for MLflow's S3 artifacts. DVC "
        "reads its own from `.dvc/config.local`, which boto3 doesn't see.",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Rank sources: explicit arguments, then params.yaml, then env and .env."""
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
