import re

import pydantic
import pytest

from src.config import Settings, bedrock_client_config, get_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    # Settings reads the developer's real .env and shell env by default;
    # strip anything that could leak into these tests.
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    monkeypatch.delenv("STAGING_DB_PATH", raising=False)
    monkeypatch.delenv("BEDROCK_CONNECT_TIMEOUT", raising=False)
    monkeypatch.delenv("BEDROCK_READ_TIMEOUT", raising=False)
    monkeypatch.delenv("BEDROCK_MAX_TOKENS", raising=False)


def test_defaults_load():
    settings = Settings(_env_file=None)
    assert settings.bedrock_region == "us-east-1"
    assert settings.bedrock_model_id == "us.anthropic.claude-sonnet-4-6"
    assert settings.staging_db_path == "data/staging.db"
    assert settings.bedrock_connect_timeout == 5
    assert settings.bedrock_read_timeout == 60
    assert settings.bedrock_max_tokens == 1024


def test_invalid_bedrock_region_raises(monkeypatch):
    monkeypatch.setenv("BEDROCK_REGION", "us-eats1")
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None)


def test_empty_bedrock_model_id_raises(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "")
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None)


def test_staging_db_path_missing_folder_raises(tmp_path, monkeypatch):
    missing_dir = tmp_path / "missing"
    db_path = str(missing_dir / "staging.db")
    monkeypatch.setenv("STAGING_DB_PATH", db_path)

    with pytest.raises(pydantic.ValidationError, match=re.escape(str(missing_dir))):
        Settings(_env_file=None)


def test_staging_db_path_bare_filename_is_valid(monkeypatch):
    monkeypatch.setenv("STAGING_DB_PATH", "staging.db")
    settings = Settings(_env_file=None)
    assert settings.staging_db_path == "staging.db"


def test_staging_db_path_inside_existing_folder_is_valid(tmp_path, monkeypatch):
    db_path = str(tmp_path / "staging.db")
    monkeypatch.setenv("STAGING_DB_PATH", db_path)
    settings = Settings(_env_file=None)
    assert settings.staging_db_path == db_path


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("BEDROCK_REGION", "eu-west-1")
    settings = Settings(_env_file=None)
    assert settings.bedrock_region == "eu-west-1"


@pytest.mark.parametrize(
    ("env_var", "value", "expected"),
    [
        ("BEDROCK_CONNECT_TIMEOUT", "10", 10.0),
        ("BEDROCK_READ_TIMEOUT", "30", 30.0),
        ("BEDROCK_MAX_TOKENS", "2048", 2048),
    ],
)
def test_bedrock_timeout_and_max_tokens_env_overrides(
    monkeypatch, env_var, value, expected
):
    monkeypatch.setenv(env_var, value)
    settings = Settings(_env_file=None)
    assert getattr(settings, env_var.lower()) == expected


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("BEDROCK_CONNECT_TIMEOUT", "0"),
        ("BEDROCK_CONNECT_TIMEOUT", "-1"),
        ("BEDROCK_CONNECT_TIMEOUT", "60.1"),
        ("BEDROCK_READ_TIMEOUT", "0"),
        ("BEDROCK_READ_TIMEOUT", "-1"),
        ("BEDROCK_READ_TIMEOUT", "120.1"),
        ("BEDROCK_MAX_TOKENS", "0"),
        ("BEDROCK_MAX_TOKENS", "-1"),
        ("BEDROCK_MAX_TOKENS", "4097"),
    ],
)
def test_bedrock_timeout_and_max_tokens_reject_out_of_range(
    monkeypatch, env_var, value
):
    monkeypatch.setenv(env_var, value)
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None)


def test_bedrock_client_config_has_single_attempt_and_configured_timeouts(monkeypatch):
    monkeypatch.setenv("BEDROCK_CONNECT_TIMEOUT", "7")
    monkeypatch.setenv("BEDROCK_READ_TIMEOUT", "42")

    config = bedrock_client_config()

    assert config.retries == {"total_max_attempts": 1}
    assert config.connect_timeout == get_settings().bedrock_connect_timeout == 7
    assert config.read_timeout == get_settings().bedrock_read_timeout == 42
