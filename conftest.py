"""Root conftest: its presence makes pytest add the repo root to sys.path, so tests can import src."""

import pytest

from src.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    # Prevents cached settings from leaking between tests.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _force_tracing_off(request, monkeypatch):
    """No test ever sends a real Langfuse trace, even on a machine whose
    local .env has real keys: force both keys to "" (empty), which beats
    the .env file -- pydantic-settings reads environment variables before
    the dotenv file, so this wins regardless of what's on disk. Skipped for
    tests marked @pytest.mark.live, which keep real settings so they can
    exercise the real tracing path.
    """
    if "live" in request.keywords:
        yield
        return
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    yield
