import json
import sqlite3

import pytest

from src.staging import (
    decision_cost,
    find_decision,
    get_monthly_state,
    has_open_escalation,
    init_db,
    open_escalation,
    record_decision,
    record_llm_calls,
    resolve_escalation,
    run_cost,
    upsert_monthly_state,
)


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    yield connection
    connection.close()


SAMPLE_INPUTS = {
    "comp_prices": [80, 40, 35],
    "comp_ratings": [4.5, 4.0, 3.8],
    "our_rating": 4.0,
}


def test_init_db_twice_raises_nothing(conn):
    init_db(conn)


def test_init_db_creates_llm_calls_with_expected_columns(conn):
    columns = [
        (row[1], row[2], row[3]) for row in conn.execute("PRAGMA table_info(llm_calls)")
    ]
    assert columns == [
        ("id", "INTEGER", 0),
        ("run_id", "TEXT", 1),
        ("product_id", "TEXT", 1),
        ("observed_at", "TEXT", 1),
        ("call", "TEXT", 1),
        ("attempt", "INTEGER", 1),
        ("input_tokens", "INTEGER", 0),
        ("output_tokens", "INTEGER", 0),
        ("outcome", "TEXT", 1),
        ("request_id", "TEXT", 0),
        ("input_price_per_mtok", "REAL", 1),
        ("output_price_per_mtok", "REAL", 1),
        ("estimated_cost_usd", "REAL", 0),
        ("created_at", "TEXT", 1),
    ]


def test_record_llm_calls_round_trip(conn):
    rows = [
        (
            "r1",
            "g4",
            "2018-09-01T10:00:00Z",
            "step",
            1,
            100,
            20,
            "ok",
            "req-1",
            3.0,
            15.0,
            0.0006,
        ),
        (
            "r1",
            "g4",
            "2018-09-01T10:00:00Z",
            "step",
            2,
            None,
            None,
            "invalid_output",
            None,
            3.0,
            15.0,
            None,
        ),
    ]
    record_llm_calls(conn, rows)
    fetched = conn.execute(
        """
        SELECT run_id, product_id, observed_at, call, attempt, input_tokens,
               output_tokens, outcome, request_id, input_price_per_mtok,
               output_price_per_mtok, estimated_cost_usd
        FROM llm_calls
        ORDER BY attempt
        """
    ).fetchall()
    assert fetched == rows


def test_record_llm_calls_empty_list_inserts_nothing(conn):
    record_llm_calls(conn, [])
    count = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    assert count == 0


def test_record_llm_calls_not_null_violation_raises_and_inserts_nothing(conn):
    rows = [
        (
            "r1",
            "g4",
            "2018-09-01T10:00:00Z",
            "step",
            1,
            100,
            20,
            "ok",
            "req-1",
            3.0,
            15.0,
            0.0006,
        ),
        (
            None,
            "g4",
            "2018-09-01T10:00:00Z",
            "step",
            2,
            None,
            None,
            "ok",
            None,
            3.0,
            15.0,
            None,
        ),
    ]
    with pytest.raises(sqlite3.IntegrityError):
        record_llm_calls(conn, rows)
    count = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    assert count == 0


def test_run_cost_sums_only_its_own_run(conn):
    record_llm_calls(
        conn,
        [
            (
                "r1",
                "g4",
                "2018-09-01T10:00:00Z",
                "step",
                1,
                1000,
                200,
                "ok",
                "req-1",
                3.0,
                15.0,
                0.006,
            ),
            (
                "r1",
                "g4",
                "2018-09-01T10:00:00Z",
                "step",
                2,
                500,
                100,
                "ok",
                "req-2",
                3.0,
                15.0,
                0.003,
            ),
            (
                "r2",
                "g5",
                "2018-09-02T10:00:00Z",
                "step",
                1,
                1000,
                200,
                "ok",
                "req-3",
                3.0,
                15.0,
                999.0,
            ),
        ],
    )
    assert run_cost(conn, "r1") == {
        "estimated_cost_usd": pytest.approx(0.009),
        "unknown_cost_calls": 0,
    }


def test_run_cost_counts_null_cost_rows_separately_from_the_sum(conn):
    record_llm_calls(
        conn,
        [
            (
                "r1",
                "g4",
                "2018-09-01T10:00:00Z",
                "step",
                1,
                1000,
                200,
                "ok",
                "req-1",
                3.0,
                15.0,
                0.006,
            ),
            (
                "r1",
                "g4",
                "2018-09-01T10:00:00Z",
                "step",
                2,
                None,
                None,
                "invalid_output",
                None,
                3.0,
                15.0,
                None,
            ),
        ],
    )
    assert run_cost(conn, "r1") == {
        "estimated_cost_usd": pytest.approx(0.006),
        "unknown_cost_calls": 1,
    }


def test_run_cost_empty_run_returns_zero_float_and_zero_count(conn):
    result = run_cost(conn, "missing-run")
    assert result == {"estimated_cost_usd": 0.0, "unknown_cost_calls": 0}
    assert isinstance(result["estimated_cost_usd"], float)


def test_decision_cost_filters_to_one_product_and_observation_within_a_run(conn):
    record_llm_calls(
        conn,
        [
            (
                "r1",
                "g4",
                "2018-09-01T10:00:00Z",
                "step",
                1,
                1000,
                200,
                "ok",
                "req-1",
                3.0,
                15.0,
                0.006,
            ),
            (
                "r1",
                "g4",
                "2018-09-02T10:00:00Z",
                "step",
                1,
                1000,
                200,
                "ok",
                "req-2",
                3.0,
                15.0,
                0.009,
            ),
            (
                "r1",
                "g5",
                "2018-09-01T10:00:00Z",
                "step",
                1,
                1000,
                200,
                "ok",
                "req-3",
                3.0,
                15.0,
                0.012,
            ),
            (
                "r2",
                "g4",
                "2018-09-01T10:00:00Z",
                "step",
                1,
                1000,
                200,
                "ok",
                "req-4",
                3.0,
                15.0,
                0.015,
            ),
        ],
    )
    assert decision_cost(conn, "r1", "g4", "2018-09-01T10:00:00Z") == {
        "estimated_cost_usd": pytest.approx(0.006),
        "unknown_cost_calls": 0,
    }


def test_record_decision_round_trip(conn):
    brief = {"recommendation": "multi_month_path", "rationale": "Gradual is safer."}
    record_decision(
        conn,
        "g4",
        "2018-09-01T10:00:00Z",
        "escalated",
        None,
        "cap_exceeded",
        "medium",
        "Escalated (medium): target beyond the cap.",
        brief,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    row = conn.execute(
        "SELECT product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief, "
        "input_hash, inputs, created_at FROM decisions"
    ).fetchone()
    (
        product_id,
        observed_at,
        outcome,
        price,
        escalation_cause,
        tier,
        reason,
        brief_json,
        input_hash,
        inputs_json,
        created_at,
    ) = row
    assert product_id == "g4"
    assert outcome == "escalated"
    assert price is None
    assert escalation_cause == "cap_exceeded"
    assert tier == "medium"
    assert reason == "Escalated (medium): target beyond the cap."
    assert json.loads(brief_json) == brief
    assert input_hash == "a" * 64
    assert json.loads(inputs_json) == SAMPLE_INPUTS
    assert created_at is not None


def test_record_decision_no_brief_stores_null_and_keeps_price(conn):
    record_decision(
        conn,
        "g4",
        "2018-09-01T10:00:00Z",
        "priced",
        42.5,
        None,
        "low",
        "Priced at 42.50: target is the mid of the competitor band (42.50).",
        None,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    row = conn.execute("SELECT price, brief FROM decisions").fetchone()
    price, brief_json = row
    assert price == 42.5
    assert brief_json is None


def test_record_decision_duplicate_product_and_observed_at_raises(conn):
    record_decision(
        conn,
        "g4",
        "2018-09-01T10:00:00Z",
        "priced",
        42.5,
        None,
        "low",
        "Priced at 42.50.",
        None,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    with pytest.raises(sqlite3.IntegrityError):
        record_decision(
            conn,
            "g4",
            "2018-09-01T10:00:00Z",
            "priced",
            43.0,
            None,
            "low",
            "Priced at 43.00.",
            None,
            "a" * 64,
            SAMPLE_INPUTS,
        )


def test_record_decision_same_product_different_observed_at_inserts_fine(conn):
    record_decision(
        conn,
        "g4",
        "2018-09-01T10:00:00Z",
        "priced",
        42.5,
        None,
        "low",
        "Priced at 42.50.",
        None,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    record_decision(
        conn,
        "g4",
        "2018-09-02T10:00:00Z",
        "priced",
        43.0,
        None,
        "low",
        "Priced at 43.00.",
        None,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    count = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE product_id = ?", ("g4",)
    ).fetchone()[0]
    assert count == 2


def test_record_decision_same_date_different_time_inserts_fine(conn):
    record_decision(
        conn,
        "g4",
        "2018-09-01T10:00:00Z",
        "priced",
        42.5,
        None,
        "low",
        "Priced at 42.50.",
        None,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    record_decision(
        conn,
        "g4",
        "2018-09-01T14:30:00Z",
        "priced",
        43.0,
        None,
        "low",
        "Priced at 43.00.",
        None,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    count = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE product_id = ?", ("g4",)
    ).fetchone()[0]
    assert count == 2


def test_find_decision_round_trip(conn):
    brief = {"recommendation": "multi_month_path", "rationale": "Gradual is safer."}
    record_decision(
        conn,
        "g4",
        "2018-09-01T10:00:00Z",
        "escalated",
        None,
        "cap_exceeded",
        "medium",
        "Escalated (medium): target beyond the cap.",
        brief,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    decision = find_decision(conn, "g4", "2018-09-01T10:00:00Z")
    assert decision["product_id"] == "g4"
    assert decision["outcome"] == "escalated"
    assert decision["brief"] == brief
    assert decision["input_hash"] == "a" * 64
    assert decision["inputs"] == SAMPLE_INPUTS


def test_find_decision_missing_returns_none(conn):
    assert find_decision(conn, "missing", "2018-09-01T10:00:00Z") is None


def test_find_decision_no_brief_returns_none(conn):
    record_decision(
        conn,
        "g4",
        "2018-09-01T10:00:00Z",
        "priced",
        42.5,
        None,
        "low",
        "Priced at 42.50.",
        None,
        "a" * 64,
        SAMPLE_INPUTS,
    )
    decision = find_decision(conn, "g4", "2018-09-01T10:00:00Z")
    assert decision["brief"] is None


def test_upsert_monthly_state_updates_current_price_not_anchor(conn):
    upsert_monthly_state(conn, "g4", "2018-09", 30.0, 30.0)
    upsert_monthly_state(conn, "g4", "2018-09", 999.0, 32.5)
    state = get_monthly_state(conn, "g4", "2018-09")
    assert state["anchor_price"] == 30.0
    assert state["current_price"] == 32.5


def test_get_monthly_state_missing_returns_none(conn):
    assert get_monthly_state(conn, "missing", "2018-09") is None


def test_has_open_escalation_false_with_no_rows(conn):
    assert has_open_escalation(conn, "g4") is False


def test_has_open_escalation_true_after_open_false_after_resolve(conn):
    escalation_id = open_escalation(conn, "g4", "cap_exceeded")
    assert has_open_escalation(conn, "g4") is True
    resolve_escalation(conn, escalation_id)
    assert has_open_escalation(conn, "g4") is False


def test_has_open_escalation_does_not_count_another_product(conn):
    open_escalation(conn, "g4", "cap_exceeded")
    assert has_open_escalation(conn, "other") is False


def test_open_escalation_detail_round_trips_as_json(conn):
    detail = {"triggered_count": 2, "total_count": 3}
    escalation_id = open_escalation(conn, "g4", "competitor_move", detail)
    detail_json = conn.execute(
        "SELECT detail FROM escalations WHERE id = ?", (escalation_id,)
    ).fetchone()[0]
    assert json.loads(detail_json) == detail


def test_open_escalation_no_detail_stores_null(conn):
    escalation_id = open_escalation(conn, "g4", "cap_exceeded")
    detail = conn.execute(
        "SELECT detail FROM escalations WHERE id = ?", (escalation_id,)
    ).fetchone()[0]
    assert detail is None
