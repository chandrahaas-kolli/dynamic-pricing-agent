import json
import logging
from datetime import UTC, datetime

import pytest

from src.logs import (
    LOGGER_NAME,
    _StdoutJsonHandler,
    configure_logging,
    log_event,
    logged_node,
)


def test_log_event_emits_one_line_of_valid_json_with_time_level_event_and_fields(
    capsys,
):
    configure_logging()
    log_event("priced", level="INFO", run_id="r1", product_id="g4")

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["event"] == "priced"
    assert record["level"] == "INFO"
    assert record["run_id"] == "r1"
    assert record["product_id"] == "g4"
    # UTC ISO 8601 with a literal Z suffix, whole seconds.
    assert datetime.strptime(record["time"], "%Y-%m-%dT%H:%M:%SZ").tzinfo is None


def test_log_event_with_no_fields_still_emits_time_level_event(capsys):
    configure_logging()
    log_event("run_started")

    record = json.loads(capsys.readouterr().out.splitlines()[0])
    assert record == {
        "event": "run_started",
        "level": "INFO",
        "time": record["time"],
    }


def test_log_event_non_serializable_field_value_falls_back_to_str(capsys):
    configure_logging()
    not_serializable = datetime(2026, 1, 1, tzinfo=UTC)
    log_event("observed", duration=not_serializable)

    record = json.loads(capsys.readouterr().out.splitlines()[0])
    assert record["duration"] == str(not_serializable)


def test_configure_logging_twice_does_not_duplicate_the_handler(capsys):
    configure_logging()
    configure_logging()

    # Count only our own handler type: the test runner's own log-capturing
    # handler may also be attached to this logger (it targets any logger
    # with propagate=False, which is unrelated to configure_logging's own
    # idempotency), so it must not be mistaken for a duplicate.
    logger = logging.getLogger(LOGGER_NAME)
    our_handlers = [h for h in logger.handlers if isinstance(h, _StdoutJsonHandler)]
    assert len(our_handlers) == 1

    log_event("startup")
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_configure_logging_sets_propagate_false():
    configure_logging()
    assert logging.getLogger(LOGGER_NAME).propagate is False


def test_level_filtering_drops_records_below_configured_level(capsys):
    configure_logging(level="WARNING")
    log_event("ignored", level="INFO")
    log_event("kept", level="ERROR")

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "kept"


# --- logged_node ---


def _some_node(state):
    return {"b": 2, "a": 1}


def test_logged_node_logs_node_end_and_returns_the_result_unchanged(capsys):
    configure_logging()
    wrapped = logged_node(_some_node)

    result = wrapped({"run_id": "r1", "product_id": "g4"})

    assert result == {"b": 2, "a": 1}
    record = json.loads(capsys.readouterr().out.splitlines()[0])
    assert record["event"] == "node_end"
    assert record["level"] == "INFO"
    assert record["run_id"] == "r1"
    assert record["product_id"] == "g4"
    assert record["node"] == "_some_node"
    assert isinstance(record["duration_ms"], (int, float))
    assert record["changed"] == ["a", "b"]


def test_logged_node_node_returning_none_logs_node_error_and_reraises(capsys):
    configure_logging()
    wrapped = logged_node(lambda state: None)

    with pytest.raises(TypeError):
        wrapped({"run_id": "r1", "product_id": "g4"})

    record = json.loads(capsys.readouterr().out.splitlines()[0])
    assert record["event"] == "node_error"
    assert record["level"] == "ERROR"
    assert record["error_type"] == "TypeError"
    assert "changed" not in record


def test_logged_node_failing_node_logs_node_error_and_reraises_same_exception(capsys):
    configure_logging()

    def _failing_node(state):
        raise ValueError("boom")

    wrapped = logged_node(_failing_node)

    with pytest.raises(ValueError, match="boom"):
        wrapped({"run_id": "r1", "product_id": "g4"})

    out = capsys.readouterr().out
    record = json.loads(out.splitlines()[0])
    assert record["event"] == "node_error"
    assert record["level"] == "ERROR"
    assert record["node"] == "_failing_node"
    assert record["error_type"] == "ValueError"
    # The message text ("boom") is never logged, only the exception's type.
    assert "boom" not in out


def test_logged_node_keeps_the_wrapped_function_name():
    wrapped = logged_node(_some_node)
    assert wrapped.__name__ == "_some_node"
