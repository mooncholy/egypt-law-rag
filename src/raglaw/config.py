from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Paths(BaseModel):
    """Where the pipeline reads and writes, relative to the repo root."""

    raw_pdf: Path
    errata: Path
    interim_dir: Path
    corpus_dir: Path
    gold_dir: Path
    metrics_dir: Path
    reports_dir: Path
    logs_dir: Path


class RootHeading(BaseModel):
    """The page 1 heading that opens every heading path (P7)."""

    ar: str
    en: str


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
    paths: Paths
    root_heading: RootHeading

    # From the environment or .env. The key is optional so the service still
    # starts without it: /health reports the gap and /ask answers 503.
    llm_api_key: SecretStr | None = None
    s3_bucket: str | None = None
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    # s3://<bucket>/mlflow; unset, MLflow keeps artifacts beside its database.
    mlflow_artifact_root: str | None = None

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
