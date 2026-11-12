"""Structured JSON logging for the pricing agent.

Logged: event names, ids (run_id, product_id), durations, error types/codes,
outcomes, and the NAMES of state keys -- enough to see what the agent did and
how long it took, without reconstructing what it decided. NEVER logged: full
prompts, model text, credentials, prices, or any other state value.

configure_logging() must be called once at startup; importing this module
configures nothing. Entry points MUST call it: without it, INFO lines are
silently dropped (the logger's effective level falls back to the root
logger's default, WARNING) and WARNING+ records fall back to Python
logging's own lastResort handler, which prints plain text, not JSON.
"""

import functools
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, TypeVar

LOGGER_NAME = "pricing_agent"

# Bound to Mapping so shell can call state.get(...); a TypeVar (not a fixed
# alias) so logged_node preserves the wrapped node's own state type (e.g.
# PipelineState) instead of widening every node to a plain Mapping.
NodeState = TypeVar("NodeState", bound=Mapping[str, Any])


class JsonFormatter(logging.Formatter):
    """Formats each LogRecord as one line of JSON: time, level, event, plus
    every field log_event was called with. A field value that isn't JSON-
    serializable is converted with str().
    """

    def format(self, record):
        fields = getattr(record, "pricing_fields", {})
        payload = {
            **fields,
            "time": datetime.fromtimestamp(record.created, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        return json.dumps(payload, default=str)


def log_event(event, level="INFO", **fields):
    """Emit one JSON log record through the "pricing_agent" logger: event plus every field in **fields."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.log(getattr(logging, level.upper()), event, extra={"pricing_fields": fields})


class _StdoutJsonHandler(logging.StreamHandler):
    """Marker subclass so configure_logging can recognize its own handler
    and skip re-adding it, without being tripped up by unrelated handlers
    (e.g. a test runner's) that a caller never added themselves.
    """


def configure_logging(level="INFO"):
    """Attach one stdout JSON handler to the "pricing_agent" logger.

    Idempotent: a second call never adds a second handler, so log lines are
    never duplicated. propagate=False, so these records never also go
    through the root logger's handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    if not any(isinstance(h, _StdoutJsonHandler) for h in logger.handlers):
        handler = _StdoutJsonHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)


def logged_node(func: Callable[[NodeState], dict[str, Any]]):
    # No "-> Callable[[NodeState], dict[str, Any]]" here on purpose: with it,
    # mypy widens the result to a plain Callable type that LangGraph's
    # heavily overloaded add_node can no longer match (it matches a def's
    # inferred type, not a Callable[...] alias, against its Protocol
    # overloads). Leaving the return type to be inferred from `shell`'s own
    # annotation keeps add_node(name, logged_node(some_node)) type-checking.
    """Wrap a graph node with one "node_end" (success) or "node_error"
    (exception) log line per call.

    Logs run_id, product_id, node (func.__name__), duration_ms, and on
    success, changed (the NAMES of the keys the node returned -- never
    their values, since a value may be a price or other state we never
    log). On an exception, logs error_type (the exception's class name,
    never its message) instead of changed, then re-raises the exact same
    exception unchanged.

    Known limit: the wrapped node must take only `state`; a node that also
    takes `config` (or anything else) would need *args/**kwargs
    passthrough, which this does not do.
    """

    @functools.wraps(func)
    def shell(state: NodeState) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            result = func(state)
            changed = sorted(result)
        except Exception as err:
            log_event(
                "node_error",
                level="ERROR",
                run_id=state.get("run_id"),
                product_id=state.get("product_id"),
                node=func.__name__,
                error_type=type(err).__name__,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            raise

        log_event(
            "node_end",
            level="INFO",
            run_id=state.get("run_id"),
            product_id=state.get("product_id"),
            node=func.__name__,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
            changed=changed,
        )
        return result

    return shell
