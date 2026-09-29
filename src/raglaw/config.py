from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every configurable value in the project, read from the environment or ``.env``.

    Variables carry the ``RAGLAW_`` prefix, so ``llm_api_key`` is read from
    ``RAGLAW_LLM_API_KEY``. Pipeline paths, the S3 bucket and chunk sizes join
    this class as their phases add them; nothing is hardcoded elsewhere.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGLAW_", env_file=".env", extra="ignore"
    )

    # Optional so the service still starts without it: /health reports the gap
    # and /ask answers 503 instead of the process refusing to boot.
    llm_api_key: SecretStr | None = None
