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
