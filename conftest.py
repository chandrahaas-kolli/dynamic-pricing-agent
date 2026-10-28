"""Root conftest: its presence makes pytest add the repo root to sys.path, so tests can import src."""

import pytest

from src.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    # Prevents cached settings from leaking between tests.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
