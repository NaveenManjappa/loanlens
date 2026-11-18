"""Application settings, loaded from environment variables and an optional `.env` file.

Every value LoanLens reads from its environment is declared here, once. The variable names are
the field names in upper case (`database_url` -> `DATABASE_URL`), so existing `.env` files and the
Azure container app settings keep working. Values in the real environment win over `.env`.

LangSmith's `LANGSMITH_*` variables are not listed: the LangSmith library reads them itself.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SQLITE_PATH = "loanlens_cases.db"


class Settings(BaseSettings):
    """All configuration for LoanLens."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Azure OpenAI v1 API (Microsoft Foundry). Optional here so the app and tests can build
    # Settings without them; get_model() raises a clear error if any is missing.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    # The name YOU gave the deployment in Foundry, not the model name.
    azure_openai_deployment_name: str | None = None

    # Without DATABASE_URL, cases are kept in a local SQLite file at sqlite_path.
    database_url: SecretStr | None = None
    sqlite_path: str = DEFAULT_SQLITE_PATH

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_case_log_level(cls, value: object) -> object:
        """Accept LOG_LEVEL=info as well as INFO."""
        return value.upper() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    """The settings, read once per process."""
    return Settings()
