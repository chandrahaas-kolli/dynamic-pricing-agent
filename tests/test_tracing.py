import pytest

from src.config import Settings, get_settings
from src.tracing import (
    get_langfuse_client,
    get_trace_config,
    shutdown_tracing,
    tracing_enabled,
)


@pytest.fixture(autouse=True)
def _clear_langfuse_client_cache():
    get_langfuse_client.cache_clear()
    yield
    get_langfuse_client.cache_clear()


class _FakeHandler:
    def __init__(self, *, public_key=None, trace_context=None):
        self.public_key = public_key
        self.trace_context = trace_context


class _FakeLangfuseClient:
    def __init__(
        self, *, public_key=None, secret_key=None, base_url=None, environment=None
    ):
        self.public_key = public_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.environment = environment
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


# --- tracing_enabled ---


@pytest.mark.parametrize(
    ("public_key", "secret_key"),
    [
        (None, "sk-real"),
        ("pk-real", None),
        (None, None),
        ("", "sk-real"),
        ("pk-real", ""),
        ("", ""),
    ],
)
def test_tracing_enabled_false_when_either_key_absent_or_empty(public_key, secret_key):
    settings = Settings(
        _env_file=None, langfuse_public_key=public_key, langfuse_secret_key=secret_key
    )
    assert tracing_enabled(settings) is False


def test_tracing_enabled_true_when_both_keys_present_and_nonempty():
    settings = Settings(
        _env_file=None, langfuse_public_key="pk-real", langfuse_secret_key="sk-real"
    )
    assert tracing_enabled(settings) is True


# --- get_trace_config: off ---


def test_get_trace_config_off_when_keys_absent_returns_empty_dict(monkeypatch):
    settings = Settings(_env_file=None)
    monkeypatch.setattr("src.tracing.get_settings", lambda: settings)
    assert get_trace_config("run-1", "g4") == {}


def test_get_trace_config_off_when_keys_empty_returns_empty_dict(monkeypatch):
    settings = Settings(_env_file=None, langfuse_public_key="", langfuse_secret_key="")
    monkeypatch.setattr("src.tracing.get_settings", lambda: settings)
    assert get_trace_config("run-1", "g4") == {}


# --- get_trace_config: on (client/handler mocked -- no network) ---


def test_get_trace_config_on_returns_one_callback_handler_and_metadata(monkeypatch):
    settings = Settings(
        _env_file=None,
        langfuse_public_key="pk-real",
        langfuse_secret_key="sk-real",
        langfuse_base_url="https://us.cloud.langfuse.com",
    )
    monkeypatch.setattr("src.tracing.get_settings", lambda: settings)
    monkeypatch.setattr("src.tracing.Langfuse", _FakeLangfuseClient)
    monkeypatch.setattr("src.tracing.CallbackHandler", _FakeHandler)

    config = get_trace_config("run-1", "g4")

    assert config["metadata"] == {
        "run_id": "run-1",
        "product_id": "g4",
        "langfuse_session_id": "run-1",
    }
    assert len(config["callbacks"]) == 1
    assert isinstance(config["callbacks"][0], _FakeHandler)
    assert config["callbacks"][0].public_key == "pk-real"


def test_get_langfuse_client_builds_from_settings_only(monkeypatch):
    settings = Settings(
        _env_file=None,
        langfuse_public_key="pk-real",
        langfuse_secret_key="sk-real",
        langfuse_base_url="https://custom.langfuse.example",
        app_env="prod",
    )
    monkeypatch.setattr("src.tracing.get_settings", lambda: settings)
    monkeypatch.setattr("src.tracing.Langfuse", _FakeLangfuseClient)

    client = get_langfuse_client()

    assert client.public_key == "pk-real"
    assert client.secret_key == "sk-real"
    assert client.base_url == "https://custom.langfuse.example"
    assert client.environment == "prod"


# --- shutdown_tracing ---


def test_shutdown_tracing_is_noop_when_no_client_was_ever_created():
    shutdown_tracing()  # must not raise, and must not create a client
    assert get_langfuse_client.cache_info().currsize == 0


def test_shutdown_tracing_shuts_down_the_cached_client(monkeypatch):
    settings = Settings(
        _env_file=None, langfuse_public_key="pk-real", langfuse_secret_key="sk-real"
    )
    monkeypatch.setattr("src.tracing.get_settings", lambda: settings)
    monkeypatch.setattr("src.tracing.Langfuse", _FakeLangfuseClient)

    client = get_langfuse_client()
    shutdown_tracing()

    assert client.shutdown_calls == 1


# --- the autouse conftest fixture that forces tracing off ---


def test_autouse_fixture_forces_tracing_off_even_with_real_keys_on_disk(
    tmp_path, monkeypatch
):
    # Simulate a developer's local .env (never the project's real, gitignored
    # .env) that carries real-looking Langfuse keys, in a throwaway directory.
    (tmp_path / "data").mkdir()  # staging_db_path's default folder must exist
    env_file = tmp_path / ".env"
    env_file.write_text("LANGFUSE_PUBLIC_KEY=pk-real\nLANGFUSE_SECRET_KEY=sk-real\n")
    monkeypatch.chdir(tmp_path)

    # The conftest.py autouse fixture already forced LANGFUSE_PUBLIC_KEY and
    # LANGFUSE_SECRET_KEY to "" for this test (every test but @pytest.mark.live);
    # since pydantic-settings reads environment variables before the dotenv
    # file, that "" wins over the real-looking keys just written to disk.
    settings = get_settings()

    assert settings.langfuse_public_key == ""
    assert settings.langfuse_secret_key.get_secret_value() == ""
    assert tracing_enabled(settings) is False
