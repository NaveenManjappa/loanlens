"""Settings: read from the environment, with safe defaults."""

import pytest

from loanlens.config.settings import DEFAULT_SQLITE_PATH, Settings
from loanlens.integrations.llm import get_model


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ignore the developer's own environment, so these tests see only what they set."""
    for name in ("LOG_LEVEL", "DATABASE_URL", "SQLITE_PATH"):
        monkeypatch.delenv(name, raising=False)
    for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"):
        monkeypatch.delenv(name, raising=False)


def test_defaults_without_any_environment():
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"
    assert settings.database_url is None
    assert settings.sqlite_path == DEFAULT_SQLITE_PATH


def test_reads_the_existing_variable_names(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "my-deployment")
    settings = Settings(_env_file=None)
    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == "postgresql://u:p@host/db"
    assert settings.azure_openai_deployment_name == "my-deployment"


def test_database_url_is_hidden_when_printed(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:secret@host/db")
    assert "secret" not in repr(Settings(_env_file=None))


def test_log_level_accepts_lower_case(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert Settings(_env_file=None).log_level == "DEBUG"


def test_get_model_names_every_missing_variable():
    settings = Settings(_env_file=None, azure_openai_endpoint="https://x.openai.azure.com")
    with pytest.raises(RuntimeError) as error:
        get_model(settings)
    message = str(error.value)
    assert "AZURE_OPENAI_API_KEY" in message
    assert "AZURE_OPENAI_DEPLOYMENT_NAME" in message
    assert "AZURE_OPENAI_ENDPOINT" not in message
