import re

import pydantic
import pytest

from src.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    # Settings reads the developer's real .env and shell env by default;
    # strip anything that could leak into these tests.
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    monkeypatch.delenv("STAGING_DB_PATH", raising=False)


def test_defaults_load():
    settings = Settings(_env_file=None)
    assert settings.bedrock_region == "us-east-1"
    assert settings.bedrock_model_id == "us.anthropic.claude-sonnet-4-6"
    assert settings.staging_db_path == "data/staging.db"


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
