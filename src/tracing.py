"""Optional Langfuse tracing for the pricing graph.

get_trace_config builds the config dict a caller passes to the graph:
app.invoke(state, config=get_trace_config(run_id, product_id)). Tracing is
on only when both Langfuse keys are configured on Settings (src/config.py);
otherwise get_trace_config returns {} and no Langfuse client is ever
created, so importing this module -- or calling it with tracing off --
touches no network and needs no credentials.

Every trace is tagged with Settings' app_env ("dev" or "prod") as the
Langfuse trace environment, and grouped into a Langfuse session keyed by
run_id, so every trace from one run (across every product it processes)
shows up as one session.

Never send full prompts or credentials anywhere except the trace itself:
this module logs nothing and only ever forwards run_id/product_id/app_env
as trace metadata/environment.
"""

from functools import lru_cache

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config import get_settings


def tracing_enabled(settings):
    """True only when both the public and secret Langfuse keys are present and non-empty."""
    if not settings.langfuse_public_key:
        return False
    if settings.langfuse_secret_key is None:
        return False
    return bool(settings.langfuse_secret_key.get_secret_value())


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Create the Langfuse client, once, from Settings' own fields only.

    Keys, base_url and environment are always passed explicitly here, so
    the Langfuse SDK's own os.environ/.env fallback (inside its __init__)
    never triggers -- Settings is the only source of credentials and
    environment. environment=settings.app_env tags every trace from this
    client as "dev" or "prod". Created lazily so importing this module
    never touches the network or requires credentials; registers the
    client that CallbackHandler.__init__ (via langfuse's get_client())
    looks up by public_key, instead of letting it construct its own
    default client straight from the raw environment.
    """
    settings = get_settings()
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.app_env,
    )


def get_trace_config(run_id, product_id):
    """Return the config dict for app.invoke(state, config=get_trace_config(...)).

    {} (no client, no callbacks, no network) when tracing is off. Otherwise
    {"callbacks": [a CallbackHandler], "metadata": {...}}: metadata carries
    run_id/product_id as plain custom fields, plus the special key
    "langfuse_session_id" (read by CallbackHandler, see langfuse.langchain.
    CallbackHandler._parse_langfuse_trace_attributes) set to run_id, so every
    trace from one run groups into one Langfuse session.
    """
    settings = get_settings()
    if not tracing_enabled(settings):
        return {}
    get_langfuse_client()
    handler = CallbackHandler(public_key=settings.langfuse_public_key)
    return {
        "callbacks": [handler],
        "metadata": {
            "run_id": run_id,
            "product_id": product_id,
            "langfuse_session_id": run_id,
        },
    }


def shutdown_tracing():
    """Flush any pending traces and stop Langfuse's background threads.

    Callers must call this once after a run finishes (e.g. a finally block
    around app.invoke, or at process exit) whenever tracing might be on --
    a short-lived process can otherwise exit before its traces are sent.
    No-op when no client was ever created (tracing off for this process).
    """
    if get_langfuse_client.cache_info().currsize == 0:
        return
    get_langfuse_client().shutdown()
