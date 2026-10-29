"""Runtime configuration read from the environment (.env in development), with
non-secret defaults so importing this module never requires a .env file.

Settings is validated once, at startup, via the cached get_settings(): a bad
value stops the app with a ValidationError naming every bad field, instead of
failing later wherever that setting happens to be used. Extra .env keys that
aren't declared here (e.g. AWS_PROFILE, read directly by boto3) are ignored
rather than rejected.
"""

import os
from functools import lru_cache

from botocore.config import Config
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Declares every setting this app reads, with its default and validation."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    bedrock_region: str = Field(default="us-east-1", pattern=r"^[a-z]{2}(-[a-z]+)+-\d$")
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-6", min_length=1
    )
    staging_db_path: str = Field(default="data/staging.db", min_length=1)
    bedrock_connect_timeout: float = Field(default=5, gt=0, le=60)
    bedrock_read_timeout: float = Field(default=60, gt=0, le=120)
    bedrock_max_tokens: int = Field(default=1024, ge=1, le=4096)

    @field_validator("staging_db_path")
    @classmethod
    def folder_must_exist(cls, value):
        """Reject a path whose folder doesn't exist; a bare file name means the current folder."""
        directory = os.path.dirname(value) or "."
        if not os.path.isdir(directory):
            raise ValueError(
                f"Invalid staging_db_path {value!r}: folder {directory!r} does not exist"
            )

        return value


@lru_cache(maxsize=1)
def get_settings():
    """Return the cached Settings, built (and validated) on first call.

    Cached so the app reads and validates the environment once. Call
    get_settings.cache_clear() to force a re-read; tests do this
    automatically (see conftest.py).
    """
    return Settings()


def bedrock_client_config() -> Config:
    """Return the botocore Config shared by every Bedrock client.

    Retries are hard-coded to 1 total attempt (never a setting): our own
    MAX_ATTEMPTS loop in agent.py's invoke_with_retries is the only retry
    layer. If botocore retried underneath it too, the two would stack and
    a single observation could make MAX_ATTEMPTS x botocore's own retry
    count real requests instead of MAX_ATTEMPTS.
    """
    settings = get_settings()
    return Config(
        retries={"total_max_attempts": 1},
        connect_timeout=settings.bedrock_connect_timeout,
        read_timeout=settings.bedrock_read_timeout,
    )
